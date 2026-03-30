"""
app.py
------
Mixr backend — Flask server with two DSP/analysis upgrades:

1. BEAT-ALIGNED CROSSFADING
   After BPM detection, all beat timestamps are computed. The user's
   requested crossfade start point is snapped to the nearest beat boundary,
   so the transition always lands on the beat — never mid-phrase.

2. ENERGY-BASED TRANSITION POINT DETECTION
   Uses RMS energy curves to find:
     - The optimal moment in Track A to BEGIN fading out (energy dip)
     - The optimal moment in Track B to BEGIN fading in (energy rise)
   Returns these suggestions to the frontend so the user can accept or override.

Endpoints:
  GET  /        → index.html
  POST /analyze → returns BPM, beat times, suggested transition points
  POST /merge   → merges with beat-aligned equal-power crossfade

Dependencies:
    pip install flask librosa numpy soundfile scipy
"""

from flask import Flask, request, send_file, jsonify
import numpy as np
import librosa
import soundfile as sf
import tempfile, os, io

app = Flask(__name__)
TARGET_SR   = 44100
ANALYSIS_SR = 22050


# ═══════════════════════════════════════════════════════
# MODULE 1 — Beat Detection & Alignment
# ═══════════════════════════════════════════════════════

def get_beat_times(y_mono: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """
    Detect BPM and return all beat timestamps using librosa's beat tracker.
    Internally uses onset strength envelope + dynamic programming.
    """
    tempo, beat_frames = librosa.beat.beat_track(
        y=y_mono, sr=sr, hop_length=512, units='frames'
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
    return float(tempo), beat_times


def snap_to_beat(time_sec: float, beat_times: np.ndarray) -> float:
    """Snap a timestamp to the nearest beat boundary."""
    if len(beat_times) == 0:
        return time_sec
    idx = int(np.argmin(np.abs(beat_times - time_sec)))
    return float(beat_times[idx])


def snap_to_bar(time_sec: float, beat_times: np.ndarray) -> float:
    """
    Snap to nearest beat, then advance to next 4-beat bar boundary.
    Crossfades starting on a bar boundary always feel more natural.
    """
    if len(beat_times) == 0:
        return time_sec
    idx     = int(np.argmin(np.abs(beat_times - time_sec)))
    bar_idx = int(np.ceil(idx / 4)) * 4
    bar_idx = min(bar_idx, len(beat_times) - 1)
    return float(beat_times[bar_idx])


# ═══════════════════════════════════════════════════════
# MODULE 2 — Energy-Based Transition Detection
# ═══════════════════════════════════════════════════════

def compute_energy_curve(y_mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute smoothed RMS energy curve.

    RMS (Root Mean Square) measures average signal power per frame.
    High RMS = loud/dense (chorus, drop). Low RMS = quiet (breakdown, outro).
    We smooth with a 1-second window to remove transient spikes.
    """
    hop = 512
    rms = librosa.feature.rms(y=y_mono, frame_length=2048, hop_length=hop)[0]

    # 1-second smoothing window
    smooth_frames = max(1, int(sr / hop))
    kernel        = np.ones(smooth_frames) / smooth_frames
    rms_smooth    = np.convolve(rms, kernel, mode='same')
    rms_norm      = rms_smooth / (rms_smooth.max() + 1e-6)

    times = librosa.frames_to_time(np.arange(len(rms_norm)), sr=sr, hop_length=hop)
    return times, rms_norm


def find_fade_out_point(y_mono: np.ndarray, sr: int, start_offset: float = 0.0) -> dict:
    """
    Find the best moment to begin fading OUT Track A.

    Looks in the final 40% of the track for the first energy drop
    below 70% of the track's average. This mirrors how a human DJ
    reads energy — they fade when the track starts losing steam.

    Returns suggested time (in seconds from file start) + full energy curve.
    """
    times, energy = compute_energy_curve(y_mono, sr)
    duration      = len(y_mono) / sr
    avg_energy    = float(np.mean(energy))

    search_mask   = times >= duration * 0.60
    if not np.any(search_mask):
        suggested = duration * 0.80
    else:
        region_e  = energy[search_mask]
        region_t  = times[search_mask]
        drops     = np.where(region_e < avg_energy * 0.70)[0]
        suggested = float(region_t[drops[0]]) if len(drops) > 0 else duration * 0.80

    _, beats      = get_beat_times(y_mono, sr)
    snapped       = snap_to_beat(suggested, beats)

    return {
        "suggested_sec":  round(snapped + start_offset, 2),
        "energy_times":   [round(float(t) + start_offset, 3) for t in times],
        "energy_values":  [round(float(v), 4) for v in energy],
    }


def find_fade_in_point(y_mono: np.ndarray, sr: int, start_offset: float = 0.0) -> dict:
    """
    Find the best moment to begin fading IN Track B.

    Looks in the first 40% of the track for where energy first rises
    above 60% of average (the track is building up). Then steps back
    one bar (4 beats) so the fade has room to build naturally.
    """
    times, energy = compute_energy_curve(y_mono, sr)
    duration      = len(y_mono) / sr
    avg_energy    = float(np.mean(energy))

    search_mask   = times <= duration * 0.40
    region_e      = energy[search_mask]
    region_t      = times[search_mask]
    rises         = np.where(region_e > avg_energy * 0.60)[0]

    bpm, beats    = get_beat_times(y_mono, sr)
    if len(rises) > 0:
        peak_time       = float(region_t[rises[0]])
        seconds_per_bar = (60.0 / bpm) * 4
        suggested       = max(0.0, peak_time - seconds_per_bar)
    else:
        suggested = 0.0

    snapped = snap_to_beat(suggested, beats)

    return {
        "suggested_sec":  round(snapped + start_offset, 2),
        "energy_times":   [round(float(t) + start_offset, 3) for t in times],
        "energy_values":  [round(float(v), 4) for v in energy],
    }


# ═══════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════

with open("index.html", "r") as f:
    HTML = f.read()

@app.route("/")
def index():
    return HTML


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze both tracks. Returns BPM, beat times, energy curves,
    and AI-suggested transition points for both tracks.
    """
    try:
        file_a  = request.files.get("track_a")
        file_b  = request.files.get("track_b")
        start_a = float(request.form.get("start_a", 0.0))
        start_b = float(request.form.get("start_b", 0.0))

        if not file_a or not file_b:
            return jsonify({"error": "Both tracks required"}), 400

        tmp_a = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_b = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        file_a.save(tmp_a.name)
        file_b.save(tmp_b.name)

        y_a, _ = librosa.load(tmp_a.name, sr=ANALYSIS_SR, mono=True, offset=start_a)
        y_b, _ = librosa.load(tmp_b.name, sr=ANALYSIS_SR, mono=True, offset=start_b)
        os.unlink(tmp_a.name)
        os.unlink(tmp_b.name)

        bpm_a, beats_a = get_beat_times(y_a, ANALYSIS_SR)
        bpm_b, beats_b = get_beat_times(y_b, ANALYSIS_SR)
        trans_a        = find_fade_out_point(y_a, ANALYSIS_SR, start_a)
        trans_b        = find_fade_in_point(y_b,  ANALYSIS_SR, start_b)

        return jsonify({
            "track_a": {
                "bpm":              round(bpm_a, 1),
                "beat_times":       [round(float(t) + start_a, 3) for t in beats_a],
                "suggested_fade_out": trans_a["suggested_sec"],
                "energy_times":     trans_a["energy_times"],
                "energy_values":    trans_a["energy_values"],
            },
            "track_b": {
                "bpm":              round(bpm_b, 1),
                "beat_times":       [round(float(t) + start_b, 3) for t in beats_b],
                "suggested_fade_in":  trans_b["suggested_sec"],
                "energy_times":     trans_b["energy_times"],
                "energy_values":    trans_b["energy_values"],
            },
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/merge", methods=["POST"])
def merge():
    """
    Merge two tracks with beat-aligned equal-power crossfade.

    Form params:
        track_a, track_b      : audio files
        fade_duration         : crossfade length (seconds)
        start_a, start_b      : playback start offsets (seconds)
        fade_start_a          : when to begin fading A out (absolute seconds)
        fade_start_b          : when to begin fading B in (absolute seconds)
        snap_to_beat          : "true" | "false"
    """
    try:
        file_a       = request.files.get("track_a")
        file_b       = request.files.get("track_b")
        fade_sec     = float(request.form.get("fade_duration", 5.0))
        start_a      = float(request.form.get("start_a", 0.0))
        start_b      = float(request.form.get("start_b", 0.0))
        fade_start_a = request.form.get("fade_start_a")
        fade_start_b = request.form.get("fade_start_b")
        do_snap      = request.form.get("snap_to_beat", "true").lower() == "true"

        tmp_a = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_b = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        file_a.save(tmp_a.name)
        file_b.save(tmp_b.name)

        # Full quality stereo for output
        y_a, _ = librosa.load(tmp_a.name, sr=TARGET_SR, mono=False, offset=start_a)
        y_b, _ = librosa.load(tmp_b.name, sr=TARGET_SR, mono=False, offset=start_b)

        # Mono at analysis SR for beat snapping
        y_a_m, _ = librosa.load(tmp_a.name, sr=ANALYSIS_SR, mono=True, offset=start_a)
        y_b_m, _ = librosa.load(tmp_b.name, sr=ANALYSIS_SR, mono=True, offset=start_b)
        os.unlink(tmp_a.name); os.unlink(tmp_b.name)

        def to_stereo(y):
            if y.ndim == 1:
                return np.stack([y, y], axis=1).astype(np.float32)
            return y.T.astype(np.float32)

        y_a = to_stereo(y_a)
        y_b = to_stereo(y_b)
        dur_a = len(y_a) / TARGET_SR
        dur_b = len(y_b) / TARGET_SR

        # ── Beat alignment ────────────────────────────
        if do_snap:
            _, beats_a = get_beat_times(y_a_m, ANALYSIS_SR)
            _, beats_b = get_beat_times(y_b_m, ANALYSIS_SR)
            raw_a  = (float(fade_start_a) - start_a) if fade_start_a else dur_a - fade_sec
            raw_b  = (float(fade_start_b) - start_b) if fade_start_b else 0.0
            snap_a = snap_to_bar(raw_a, beats_a)
            snap_b = snap_to_bar(raw_b, beats_b)
        else:
            snap_a = (float(fade_start_a) - start_a) if fade_start_a else dur_a - fade_sec
            snap_b = (float(fade_start_b) - start_b) if fade_start_b else 0.0

        fade_out_frame = int(np.clip(snap_a, 0, dur_a - fade_sec) * TARGET_SR)
        fade_in_frame  = int(np.clip(snap_b, 0, dur_b - fade_sec) * TARGET_SR)
        fade_frames    = int(fade_sec * TARGET_SR)
        fade_frames    = min(fade_frames,
                             len(y_a) - fade_out_frame,
                             len(y_b) - fade_in_frame)

        # ── Equal-power crossfade ─────────────────────
        t        = np.linspace(0, np.pi / 2, fade_frames)
        fade_out = np.cos(t).reshape(-1, 1)
        fade_in  = np.sin(t).reshape(-1, 1)

        body_a = y_a[:fade_out_frame]
        tail_a = y_a[fade_out_frame: fade_out_frame + fade_frames]
        head_b = y_b[fade_in_frame:  fade_in_frame  + fade_frames]
        body_b = y_b[fade_in_frame  + fade_frames:]

        xfade_len = min(len(tail_a), len(head_b))
        xfade = (tail_a[:xfade_len] * fade_out[:xfade_len] +
                 head_b[:xfade_len] * fade_in[:xfade_len])

        merged = np.concatenate([body_a, xfade, body_b], axis=0)
        merged = np.clip(merged, -1.0, 1.0).astype(np.float32)

        out_buf = io.BytesIO()
        sf.write(out_buf, merged, TARGET_SR, format="WAV", subtype="PCM_16")
        out_buf.seek(0)

        return send_file(out_buf, mimetype="audio/wav",
                         as_attachment=True, download_name="mixr_merged.wav")

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n  Mixr is running → http://localhost:5000\n")
    app.run(debug=True, port=5000)