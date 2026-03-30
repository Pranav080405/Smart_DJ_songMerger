"""
streamlit_app.py
----------------
Mixr — Smart Audio Merger (Streamlit version)

All DSP logic is identical to app.py:
  - BPM detection via onset strength + dynamic programming
  - RMS energy curve analysis
  - Beat-aligned crossfading
  - Equal-power fade (sin/cos)

Run:
    streamlit run streamlit_app.py

Deploy:
    Push to GitHub → connect on share.streamlit.io → done
"""

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile, os, io

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="Mixr — Smart Audio Merger",
    page_icon="🎚️",
    layout="wide"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
  h1, h2, h3 { font-family: 'Space Mono', monospace; }
  .stButton>button {
    width: 100%;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
  }
  .suggestion-box {
    background: rgba(123,108,255,0.1);
    border: 1px solid rgba(123,108,255,0.4);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
  }
</style>
""", unsafe_allow_html=True)

TARGET_SR   = 44100
ANALYSIS_SR = 22050


# ═══════════════════════════════════════════════════════
# DSP FUNCTIONS (identical to app.py)
# ═══════════════════════════════════════════════════════

def get_beat_times(y_mono, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr, hop_length=512)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
    return float(tempo), beat_times

def snap_to_beat(time_sec, beat_times):
    if len(beat_times) == 0:
        return time_sec
    idx = int(np.argmin(np.abs(beat_times - time_sec)))
    return float(beat_times[idx])

def snap_to_bar(time_sec, beat_times):
    if len(beat_times) == 0:
        return time_sec
    idx     = int(np.argmin(np.abs(beat_times - time_sec)))
    bar_idx = int(np.ceil(idx / 4)) * 4
    bar_idx = min(bar_idx, len(beat_times) - 1)
    return float(beat_times[bar_idx])

def compute_energy_curve(y_mono, sr):
    hop  = 512
    rms  = librosa.feature.rms(y=y_mono, frame_length=2048, hop_length=hop)[0]
    k    = np.ones(max(1, int(sr/hop))) / max(1, int(sr/hop))
    rms  = np.convolve(rms, k, mode='same')
    rms  = rms / (rms.max() + 1e-6)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    return times, rms

def find_fade_out_point(y_mono, sr):
    times, energy = compute_energy_curve(y_mono, sr)
    duration      = len(y_mono) / sr
    avg           = float(np.mean(energy))
    mask          = times >= duration * 0.60
    if not np.any(mask):
        suggested = duration * 0.80
    else:
        drops     = np.where(energy[mask] < avg * 0.70)[0]
        suggested = float(times[mask][drops[0]]) if len(drops) > 0 else duration * 0.80
    _, beats  = get_beat_times(y_mono, sr)
    return snap_to_beat(suggested, beats), times, energy

def find_fade_in_point(y_mono, sr):
    times, energy = compute_energy_curve(y_mono, sr)
    duration      = len(y_mono) / sr
    avg           = float(np.mean(energy))
    mask          = times <= duration * 0.40
    rises         = np.where(energy[mask] > avg * 0.60)[0]
    bpm, beats    = get_beat_times(y_mono, sr)
    if len(rises) > 0:
        peak_time = float(times[mask][rises[0]])
        suggested = max(0.0, peak_time - (60.0 / bpm) * 4)
    else:
        suggested = 0.0
    return snap_to_beat(suggested, beats), times, energy

def merge_tracks(y_a, y_b, y_a_mono, y_b_mono,
                 fade_sec, fade_start_a, fade_start_b, do_snap):

    def to_stereo(y):
        if y.ndim == 1:
            return np.stack([y, y], axis=1).astype(np.float32)
        return y.T.astype(np.float32)

    y_a = to_stereo(y_a)
    y_b = to_stereo(y_b)
    dur_a = len(y_a) / TARGET_SR
    dur_b = len(y_b) / TARGET_SR

    if do_snap:
        _, beats_a = get_beat_times(y_a_mono, ANALYSIS_SR)
        _, beats_b = get_beat_times(y_b_mono, ANALYSIS_SR)
        snap_a = snap_to_bar(fade_start_a, beats_a)
        snap_b = snap_to_bar(fade_start_b, beats_b)
    else:
        snap_a = fade_start_a
        snap_b = fade_start_b

    fade_out_frame = int(np.clip(snap_a, 0, dur_a - fade_sec) * TARGET_SR)
    fade_in_frame  = int(np.clip(snap_b, 0, dur_b - fade_sec) * TARGET_SR)
    fade_frames    = int(fade_sec * TARGET_SR)
    fade_frames    = min(fade_frames,
                         len(y_a) - fade_out_frame,
                         len(y_b) - fade_in_frame)

    t        = np.linspace(0, np.pi / 2, fade_frames)
    fade_out = np.cos(t).reshape(-1, 1)
    fade_in  = np.sin(t).reshape(-1, 1)

    body_a = y_a[:fade_out_frame]
    tail_a = y_a[fade_out_frame: fade_out_frame + fade_frames]
    head_b = y_b[fade_in_frame:  fade_in_frame  + fade_frames]
    body_b = y_b[fade_in_frame  + fade_frames:]

    xlen  = min(len(tail_a), len(head_b))
    xfade = tail_a[:xlen] * fade_out[:xlen] + head_b[:xlen] * fade_in[:xlen]

    merged = np.concatenate([body_a, xfade, body_b], axis=0)
    return np.clip(merged, -1.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════

st.title("🎚️ Mixr — Smart Audio Merger")
st.caption("Upload two songs · Analyse energy · Get one seamless beat-aligned track.")
st.divider()

col_a, col_b = st.columns(2)

# ── Track A ───────────────────────────────────────────
with col_a:
    st.subheader("🟢 Track A")
    file_a = st.file_uploader("Upload Track A", type=["mp3","wav","flac"], key="fa")

    if file_a:
        st.audio(file_a)
        st.caption(f"**{file_a.name}**")

# ── Track B ───────────────────────────────────────────
with col_b:
    st.subheader("🔴 Track B")
    file_b = st.file_uploader("Upload Track B", type=["mp3","wav","flac"], key="fb")

    if file_b:
        st.audio(file_b)
        st.caption(f"**{file_b.name}**")

st.divider()

# ── Controls ──────────────────────────────────────────
st.subheader("⚙️ Merge Settings")

ctrl1, ctrl2 = st.columns([2, 1])
with ctrl1:
    fade_sec = st.slider("Crossfade duration (seconds)", 1.0, 30.0, 5.0, 0.5)
with ctrl2:
    do_snap  = st.checkbox("Snap to beat boundary", value=True)
    st.caption("Locks crossfade to the nearest 4-beat bar")

st.divider()

# ── Analyse ───────────────────────────────────────────
if file_a and file_b:

    if st.button("⚡ ANALYSE TRACKS", use_container_width=True):
        with st.spinner("Detecting BPM and energy curves..."):

            # Save to temp files
            tmp_a = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp_b = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp_a.write(file_a.read()); tmp_a.flush()
            tmp_b.write(file_b.read()); tmp_b.flush()
            file_a.seek(0); file_b.seek(0)

            y_a_m, _ = librosa.load(tmp_a.name, sr=ANALYSIS_SR, mono=True)
            y_b_m, _ = librosa.load(tmp_b.name, sr=ANALYSIS_SR, mono=True)
            os.unlink(tmp_a.name); os.unlink(tmp_b.name)

            bpm_a, beats_a     = get_beat_times(y_a_m, ANALYSIS_SR)
            bpm_b, beats_b     = get_beat_times(y_b_m, ANALYSIS_SR)
            sug_a, et_a, ev_a  = find_fade_out_point(y_a_m, ANALYSIS_SR)
            sug_b, et_b, ev_b  = find_fade_in_point(y_b_m,  ANALYSIS_SR)

            st.session_state.bpm_a   = bpm_a
            st.session_state.bpm_b   = bpm_b
            st.session_state.sug_a   = sug_a
            st.session_state.sug_b   = sug_b
            st.session_state.et_a    = et_a
            st.session_state.ev_a    = ev_a
            st.session_state.et_b    = et_b
            st.session_state.ev_b    = ev_b
            st.session_state.dur_a   = len(y_a_m) / ANALYSIS_SR
            st.session_state.dur_b   = len(y_b_m) / ANALYSIS_SR
            st.session_state.analysed = True

    # ── Analysis results ──────────────────────────────
    if st.session_state.get("analysed"):
        res_a, res_b = st.columns(2)

        with res_a:
            st.metric("Track A BPM", f"{st.session_state.bpm_a:.1f}")
            st.markdown(f"""<div class='suggestion-box'>
                🎯 Suggested fade-out at <b>{st.session_state.sug_a:.1f}s</b><br>
                <small>Energy drops below 70% of average — outro detected</small>
            </div>""", unsafe_allow_html=True)
            st.line_chart(
                {"Energy": st.session_state.ev_a},
                height=120, use_container_width=True
            )
            fade_out_time = st.slider(
                "Fade-out start (Track A)",
                0.0, float(st.session_state.dur_a),
                float(st.session_state.sug_a), 0.1,
                key="fo_a"
            )

        with res_b:
            st.metric("Track B BPM", f"{st.session_state.bpm_b:.1f}")
            st.markdown(f"""<div class='suggestion-box'>
                🎯 Suggested fade-in at <b>{st.session_state.sug_b:.1f}s</b><br>
                <small>One bar before energy rises above 60% of average</small>
            </div>""", unsafe_allow_html=True)
            st.line_chart(
                {"Energy": st.session_state.ev_b},
                height=120, use_container_width=True
            )
            fade_in_time = st.slider(
                "Fade-in start (Track B)",
                0.0, float(st.session_state.dur_b),
                float(st.session_state.sug_b), 0.1,
                key="fi_b"
            )

        st.divider()

        # ── Merge ─────────────────────────────────────
        if st.button("🎵 MERGE TRACKS", use_container_width=True):
            with st.spinner("Applying beat-aligned crossfade..."):

                tmp_a = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tmp_b = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tmp_a.write(file_a.read()); tmp_a.flush()
                tmp_b.write(file_b.read()); tmp_b.flush()
                file_a.seek(0); file_b.seek(0)

                y_a, _   = librosa.load(tmp_a.name, sr=TARGET_SR, mono=False)
                y_b, _   = librosa.load(tmp_b.name, sr=TARGET_SR, mono=False)
                y_a_m, _ = librosa.load(tmp_a.name, sr=ANALYSIS_SR, mono=True)
                y_b_m, _ = librosa.load(tmp_b.name, sr=ANALYSIS_SR, mono=True)
                os.unlink(tmp_a.name); os.unlink(tmp_b.name)

                merged = merge_tracks(
                    y_a, y_b, y_a_m, y_b_m,
                    fade_sec,
                    fade_out_time,
                    fade_in_time,
                    do_snap
                )

                out_buf = io.BytesIO()
                sf.write(out_buf, merged, TARGET_SR, format="WAV", subtype="PCM_16")
                out_buf.seek(0)

            st.success("✅ Merged successfully!")
            st.audio(out_buf, format="audio/wav")
            st.download_button(
                label="⬇️ Download merged track (.wav)",
                data=out_buf,
                file_name="mixr_merged.wav",
                mime="audio/wav",
                use_container_width=True
            )

else:
    st.info("Upload both tracks above to get started.")