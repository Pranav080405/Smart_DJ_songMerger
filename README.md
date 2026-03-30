# Smart_DJ_songMerger

# Mixr — Smart Audio Merger

> Upload two songs. Analyse energy. Get one seamless, beat-aligned track.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![librosa](https://img.shields.io/badge/librosa-0.11-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## What is Mixr?

Mixr is a web-based audio signal processing tool that intelligently merges two songs into a single seamless track. Unlike simple audio editors that just cut and paste, Mixr analyses the acoustic properties of both tracks — their tempo, beat grid, and energy curves — to find the most musically natural transition point and execute a perceptually smooth crossfade.

It was built as a portfolio project to demonstrate applied audio DSP (Digital Signal Processing) concepts in a working, interactive product.

---

## Live Demo

🔗 **[mixr.yourdomain.com](https://mixr.yourdomain.com)** ← replace after hosting

---

## Features

- **Waveform visualisation** — full PCM waveform rendered per pixel bucket with beat tick marks overlaid
- **BPM detection** — tempo analysis using onset strength envelope + dynamic programming beat tracking
- **Energy curve analysis** — smoothed RMS power curve computed per frame to identify high and low energy regions
- **Automatic transition suggestion** — AI-suggested fade-out point for Track A and fade-in point for Track B, shown as purple markers on the energy curve
- **Beat-aligned crossfading** — user-chosen transition point snapped to the nearest 4-beat bar boundary so the crossfade always lands on the musical grid
- **Equal-power crossfade** — perceptually smooth fade using sin/cos curves (not linear) so perceived loudness stays constant through the transition
- **Manual override** — click anywhere on waveform or energy canvas to set custom start and transition points
- **Preview playback** — listen to each track from any point before merging
- **WAV export** — download the merged track as a 44.1 kHz 16-bit stereo WAV

---

## DSP Concepts Explained

### 1. Waveform Rendering

The raw PCM (Pulse Code Modulation) audio data is downsampled to one peak amplitude value per pixel column using a max-pooling approach:

```
peak[i] = max(|sample|) for all samples in bucket i
```

This gives an accurate visual representation of the signal's dynamic range without rendering every individual sample (a 3-minute song at 44100 Hz has ~8 million samples).

Beat tick marks are overlaid by converting detected beat frame indices to pixel positions:

```
pixel_x = (beat_time_seconds / total_duration) * canvas_width
```

---

### 2. BPM Detection

BPM is detected using `librosa.beat.beat_track`, which internally runs a two-stage pipeline:

**Stage 1 — Onset Strength Envelope**

The audio is transformed into a mel-frequency spectrogram (perceptually-weighted frequency representation). The onset strength at each frame is computed as the positive spectral flux — how much new energy appears compared to the previous frame. This produces a 1D signal that spikes at beats and transients.

**Stage 2 — Dynamic Programming Beat Tracking**

A tempo hypothesis is generated from the autocorrelation of the onset envelope. A dynamic programming algorithm then finds the sequence of beat positions that best fits the tempo hypothesis while staying consistent with observed onset peaks. This is more robust than simple peak picking because it enforces temporal consistency.

Output: tempo in BPM + array of beat timestamps in seconds.

---

### 3. Energy Curve Analysis & Transition Detection

**RMS Energy Curve**

For each overlapping frame of audio, the Root Mean Square energy is computed:

```
RMS[frame] = sqrt(mean(sample² for each sample in frame))
```

RMS measures the average power of the signal. High RMS = loud/dense section (chorus, drop). Low RMS = quiet section (breakdown, outro, intro).

A 1-second sliding window smoothing is applied to remove transient spikes and reveal the macro energy shape of the track.

The curve is normalised to [0, 1] for display.

**Fade-out suggestion (Track A)**

The algorithm searches the final 40% of Track A for the first frame where smoothed RMS energy drops below 70% of the track's average:

```python
drop = first frame where energy[frame] < mean(energy) * 0.70
       in the region: time > duration * 0.60
```

This is where the outro begins — the track is losing steam. A human DJ would start fading here. The result is snapped to the nearest beat.

**Fade-in suggestion (Track B)**

The algorithm searches the first 40% of Track B for the first frame where energy rises above 60% of average (the track is building up). It then steps back one bar (4 beats) to give the fade-in room to build naturally:

```python
rise    = first frame where energy[frame] > mean(energy) * 0.60
          in the region: time < duration * 0.40
start   = rise_time - (60 / BPM * 4)   # one bar earlier
```

---

### 4. Beat-Aligned Crossfading

Without beat alignment, a crossfade starting at an arbitrary time might cut mid-snare or mid-phrase, which sounds jarring.

Mixr snaps the chosen transition point to the nearest 4-beat bar boundary:

```python
beat_idx = argmin(|beat_times - chosen_time|)
bar_idx  = ceil(beat_idx / 4) * 4        # round up to next bar
snapped  = beat_times[bar_idx]
```

This ensures the transition always begins at a musically natural boundary — the start of a new bar.

---

### 5. Equal-Power Crossfade

A naive linear crossfade computes:

```
gain_A = 1 - t
gain_B = t
```

At the midpoint (t = 0.5), both tracks are at 50% amplitude. However, perceived loudness follows the square of amplitude (power), so the total perceived power dips noticeably in the middle.

Equal-power crossfading uses sin/cos curves instead:

```
angle  = t × π/2
gain_A = cos(angle)     # 1.0 → 0.0
gain_B = sin(angle)     # 0.0 → 1.0
```

At t = 0.5:
- Linear: 0.5 + 0.5 = 1.0 amplitude (sounds quieter)
- Equal-power: cos(π/4) + sin(π/4) ≈ 0.707 + 0.707 → combined power stays constant ✓

The mixed output at each frame is:

```python
xfade[frame] = audio_A[frame] * cos(t) + audio_B[frame] * sin(t)
```

---

## Project Structure

```
smart_DJ_audioMerger/
├── app.py            ← Flask backend: BPM detection, energy analysis, crossfade merge
├── index.html        ← Frontend: waveform renderer, energy curves, upload UI
├── requirements.txt  ← Python dependencies
└── README.md
```

### Backend modules inside `app.py`

| Function | Purpose |
|---|---|
| `get_beat_times()` | BPM detection + beat timestamp array |
| `snap_to_beat()` | Snap a time to nearest beat |
| `snap_to_bar()` | Snap to nearest 4-beat bar boundary |
| `compute_energy_curve()` | Smoothed RMS energy over time |
| `find_fade_out_point()` | Suggest fade-out time for Track A |
| `find_fade_in_point()` | Suggest fade-in time for Track B |
| `/analyze` endpoint | Runs full analysis, returns JSON |
| `/merge` endpoint | Applies beat-aligned crossfade, returns WAV |

---

## Installation & Running Locally

**Requirements:** Python 3.10+, pip, ffmpeg (for mp3 decoding)

```bash
# Install ffmpeg (Mac)
brew install ffmpeg

# Clone the repo
git clone https://github.com/yourusername/mixr.git
cd mixr

# Install Python dependencies
pip install -r requirements.txt

# Run
python app.py
```

Open **http://localhost:5000** in your browser.

---

## How to Use

1. **Upload** Track A and Track B (MP3, WAV, or FLAC)
2. **Click on the waveform** to set where each track starts playing from
3. **Click ⚡ Analyse Tracks** — BPM is detected and energy curves are computed
4. **Purple dashed lines** appear on the energy curves showing the AI-suggested transition points
5. Hit **"Use this"** to accept a suggestion, or click anywhere on the energy canvas to set it manually
6. Adjust the **crossfade duration** slider (1–30 seconds)
7. Toggle **Snap to beat** — keeps the crossfade locked to the musical bar grid
8. Click **Merge Tracks** — download the result as a `.wav` file

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Audio analysis | librosa, numpy, scipy |
| Audio I/O | soundfile |
| Frontend | Vanilla HTML/CSS/JS, Web Audio API |
| Visualisation | Canvas 2D API |

---

## Hosting

Recommended platforms for hosting this project:

| Platform | Notes |
|---|---|
| **Render** | Free tier, easy Flask deployment, recommended |
| **Railway** | Simple git-based deploy, generous free tier |
| **Fly.io** | More control, free tier available |
| **Heroku** | Paid only now, but very stable |

See deployment instructions below.

---

## Deploy to Render (Recommended)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app`
5. Add `gunicorn` to `requirements.txt`
6. Deploy — Render gives you a public URL instantly

Add to `requirements.txt`:
```
gunicorn>=21.0.0
```

---

## Limitations

- Large files (>50MB) may be slow to process on free hosting tiers
- BPM detection works best on music with a clear rhythmic pulse; ambient or classical music may give less accurate results
- Output is WAV (lossless) — convert to MP3 with ffmpeg if needed: `ffmpeg -i mixr_merged.wav -b:a 320k output.mp3`

---

## License

MIT — free to use, modify, and distribute.
