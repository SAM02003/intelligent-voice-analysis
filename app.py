"""Streamlit voice analysis and enhancement application for EC6675."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from dsp import (
    MAX_UPLOAD_BYTES, AudioData, apply_filters, generate_test_signal,
    proxy_scores, read_wav, recommend_with_reference, reference_scores, wav_bytes,
)
from voice_analysis import analyze_voice, spectrogram_figure_data

st.set_page_config(page_title="VoiceLab | Voice Analysis & Enhancement", page_icon="🎙️", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1250px;padding-top:1.2rem}
.hero{background:linear-gradient(115deg,#163344,#0d676c);color:white;padding:1.1rem 1.5rem;border-radius:16px;margin-bottom:1rem}
.hero h1{color:white;margin:.15rem 0;font-size:2rem}
.small-note{color:#64748b;font-size:.9rem}
</style>
<div class="hero"><div style="color:#a7f3d0;font-weight:700;letter-spacing:.12em;font-size:.72rem">NIT ROURKELA · EC6675 · SIGNAL PROCESSING</div>
<h1>Intelligent Voice Analysis & Enhancement</h1><div>Record or upload up to 20 seconds of speech, inspect its acoustic properties, and enhance the recording.</div></div>
""", unsafe_allow_html=True)


def waveform_plot(x: np.ndarray, y: np.ndarray, fs: int):
    a = x[:, 0] if x.ndim == 2 else x
    b = y[:, 0] if y.ndim == 2 else y
    step = max(1, len(a) // 7000)
    t = np.arange(0, len(a), step) / fs
    fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
    ax.plot(t, a[::step], lw=.65, alpha=.75, label="Original")
    ax.plot(t, b[::step], lw=.65, alpha=.8, label="Enhanced")
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title="Voice waveform")
    ax.grid(alpha=.2); ax.legend()
    return fig


def spectrum_plot(x: np.ndarray, y: np.ndarray, fs: int):
    def curve(v):
        v = v[:, 0] if v.ndim == 2 else v
        n = len(v); win = np.hanning(n)
        mag = np.abs(np.fft.rfft(v * win)) / max(np.sum(win), 1)
        return np.fft.rfftfreq(n, 1/fs), 20*np.log10(np.maximum(mag, 1e-7))
    f0, d0 = curve(x); f1, d1 = curve(y)
    fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
    ax.plot(f0, d0, lw=.8, label="Original", alpha=.8)
    ax.plot(f1, d1, lw=.8, label="Enhanced", alpha=.85)
    ax.set_xlim(0, min(8000, fs/2)); ax.set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", title="FFT spectrum")
    ax.grid(alpha=.2); ax.legend()
    return fig


def metric(value, unit="", digits=2):
    return "Not reliably detected" if value is None else f"{value:.{digits}f}{unit}"

with st.sidebar:
    st.header("1 · Voice input")
    input_mode = st.radio("Choose input", ["Record voice in browser", "Upload WAV", "Simulated demo"], index=0)
    audio_data = None
    if input_mode == "Record voice in browser":
        st.caption("Allow microphone access, record naturally, and stop within 20 seconds.")
        recording = st.audio_input("Record up to 20 seconds of speech", sample_rate=16000)
        if recording is not None:
            raw = recording.getvalue()
            if len(raw) > MAX_UPLOAD_BYTES:
                st.error("Recording is larger than 25 MB.")
            else:
                try:
                    audio_data = read_wav(raw, "browser_recording.wav")
                except ValueError as exc:
                    st.error(str(exc))
    elif input_mode == "Upload WAV":
        uploaded = st.file_uploader("Upload a voice recording", type=["wav"])
        if uploaded is not None:
            try:
                if uploaded.size > MAX_UPLOAD_BYTES:
                    raise ValueError("File exceeds 25 MB.")
                audio_data = read_wav(uploaded.getvalue(), uploaded.name)
            except ValueError as exc:
                st.error(str(exc))
    else:
        noise_kind = st.selectbox("Demo noise", ["High-frequency hiss", "White noise", "50 Hz hum + hiss"])
        level = st.slider("Noise RMS", .05, .40, .22, .01)
        audio_data = generate_test_signal(noise_level=level, noise_kind=noise_kind)

    st.divider(); st.header("2 · Enhancement settings")
    if audio_data is not None:
        max_cutoff = min(6000, int(.35 * audio_data.sample_rate))
        min_cutoff = min(200, max_cutoff - 1)
        cutoff = st.slider("Low-pass cutoff (Hz)", min_cutoff, max_cutoff, min(3400, max_cutoff), 100)
        window = st.slider("Moving-average window", 3, 31, 7, 2)
        taps = st.slider("FIR taps", 31, 301, 101, 10)
    else:
        st.info("Record or upload audio to enable analysis.")

if audio_data is None:
    st.info("**Get started:** Record your professor speaking naturally for up to 20 seconds, or upload a WAV recording.")
    st.stop()

if audio_data.duration > 20:
    st.error(f"Recording is {audio_data.duration:.2f} seconds. The project accepts a maximum of 20 seconds. Please record a shorter clip.")
    st.stop()

try:
    outputs = apply_filters(audio_data, float(cutoff), window, taps)
    voice = analyze_voice(audio_data.samples, audio_data.sample_rate)
    rows, recommended, reason = proxy_scores(audio_data, outputs, float(cutoff))
except (ValueError, MemoryError) as exc:
    st.error(f"Analysis failed: {exc}"); st.stop()

st.subheader("Recording overview")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Duration", f"{audio_data.duration:.2f} s")
c2.metric("Sampling rate", f"{audio_data.sample_rate:,} Hz")
c3.metric("RMS energy", f"{voice['rms']:.4f}")
c4.metric("Peak amplitude", f"{voice['peak']:.4f}")

st.subheader("Voice characteristics")
a,b,c,d = st.columns(4)
a.metric("Estimated average pitch", metric(voice['pitch_mean'], " Hz", 1))
b.metric("Pitch range (10–90%)", (f"{voice['pitch_min']:.0f}–{voice['pitch_max']:.0f} Hz" if voice['pitch_min'] is not None else "Not detected"))
c.metric("Speech activity", f"{voice['speech_activity_pct']:.1f}%")
d.metric("Silence / pauses", f"{voice['silence_pct']:.1f}%")
e,f,g,h = st.columns(4)
e.metric("Zero-crossing rate", f"{voice['zcr']:.4f}")
f.metric("Spectral centroid", f"{voice['spectral_centroid']:.0f} Hz")
g.metric("Spectral bandwidth", f"{voice['spectral_bandwidth']:.0f} Hz")
h.metric("Clipped samples", f"{voice['clipping_pct']:.3f}%")

if voice['snr_est'] is not None:
    st.info(f"**Estimated speech-to-background ratio: {voice['snr_est']:.1f} dB.** This is a rough estimate from low-energy frames within the recording, not a laboratory SNR measurement.")
else:
    st.warning("No clear low-energy region was detected, so a background-noise estimate is unavailable. Record a short pause before speaking for a more useful estimate.")

st.subheader("Noise reduction results")
st.dataframe(rows, use_container_width=True, hide_index=True)
st.success(f"**Suggested filter: {recommended}.** {reason}")
st.caption("For a natural voice recording, these are frequency-band preservation/reduction proxies. They are not true SNR or MSE because no clean reference speech signal is available.")
selected = st.selectbox("Select output to listen to", list(outputs), index=list(outputs).index(recommended))
cleaned = outputs[selected]

sig_tab, spec_tab, audio_tab = st.tabs(["Waveform & spectrum", "STFT spectrogram", "Listen & download"])
with sig_tab:
    fig = waveform_plot(audio_data.samples, cleaned, audio_data.sample_rate); st.pyplot(fig, use_container_width=True); plt.close(fig)
    fig = spectrum_plot(audio_data.samples, cleaned, audio_data.sample_rate); st.pyplot(fig, use_container_width=True); plt.close(fig)
with spec_tab:
    # Compare original and selected enhanced audio using the same STFT colour scale.
    f_orig, t_orig, z_orig = spectrogram_figure_data(audio_data.samples, audio_data.sample_rate)
    f_enh, t_enh, z_enh = spectrogram_figure_data(cleaned, audio_data.sample_rate)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, constrained_layout=True)
    vmin, vmax = -100, -20
    mesh = axes[0].pcolormesh(t_orig, f_orig, z_orig, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original voice · STFT spectrogram")
    axes[1].pcolormesh(t_enh, f_enh, z_enh, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Enhanced voice · {selected}")
    for ax in axes:
        ax.set_ylim(0, min(8000, audio_data.sample_rate / 2))
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=axes, label="Magnitude (dB)", shrink=0.9)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with audio_tab:
    left,right=st.columns(2)
    with left:
        st.markdown("**Original recording**"); st.audio(wav_bytes(audio_data.samples,audio_data.sample_rate),format="audio/wav")
    with right:
        st.markdown(f"**Enhanced recording · {selected}**")
        cleaned_bytes=wav_bytes(cleaned,audio_data.sample_rate); st.audio(cleaned_bytes,format="audio/wav")
        st.download_button("Download enhanced WAV",cleaned_bytes,file_name="enhanced_voice.wav",mime="audio/wav",type="primary")

st.subheader("Recording-specific interpretation")
notes = []

# Signal level and recording quality
if voice["clipping_pct"] >= 1.0:
    notes.append(f"Clipping is noticeable ({voice['clipping_pct']:.2f}% of samples). Reduce microphone/input gain and record again to avoid distortion.")
elif voice["clipping_pct"] > 0.05:
    notes.append(f"A small amount of clipping was detected ({voice['clipping_pct']:.2f}%). The recording is close to the maximum digital level in a few places.")
else:
    notes.append(f"Clipping is negligible ({voice['clipping_pct']:.3f}%), so there is little evidence of peak-level digital distortion.")

if voice["rms"] < 0.02:
    notes.append(f"The overall signal level is low (RMS {voice['rms']:.4f}); a closer microphone position or modest input-gain increase may help.")
elif voice["rms"] > 0.25:
    notes.append(f"The recording has relatively high average signal energy (RMS {voice['rms']:.4f}). Check playback for distortion or excessive loudness.")
else:
    notes.append(f"The overall signal energy is moderate (RMS {voice['rms']:.4f}).")

# Speech activity and pauses
activity = voice["speech_activity_pct"]
silence = voice["silence_pct"]
if activity >= 85:
    notes.append(f"Speech/activity occupies most of the clip ({activity:.1f}%), with relatively few pauses ({silence:.1f}%).")
elif activity >= 50:
    notes.append(f"The clip contains a mixture of active speech and pauses: activity {activity:.1f}%, pauses/low-energy frames {silence:.1f}%.")
else:
    notes.append(f"A large part of the clip is classified as low-energy or pause time (activity {activity:.1f}%, pauses {silence:.1f}%). This may reflect long pauses, quiet speech, or background conditions.")

# Pitch characteristics: describe measured pitch, without inferring identity or emotion.
if voice["pitch_mean"] is None:
    notes.append("A reliable periodic pitch estimate was not obtained. This can occur with unvoiced speech, weak/quiet voice, or noisy recordings.")
else:
    pitch_span = voice["pitch_max"] - voice["pitch_min"]
    notes.append(f"The estimated median voiced pitch is {voice['pitch_mean']:.1f} Hz, with a 10th–90th percentile range of {voice['pitch_min']:.1f}–{voice['pitch_max']:.1f} Hz (spread {pitch_span:.1f} Hz). This describes periodic voiced frames only.")

# Spectral characteristics
centroid = voice["spectral_centroid"]
bandwidth = voice["spectral_bandwidth"]
if centroid < 1000:
    centroid_desc = "energy is weighted toward lower frequencies"
elif centroid < 2500:
    centroid_desc = "the spectral energy is centred in a lower-to-mid frequency region"
else:
    centroid_desc = "the spectrum has a comparatively high-frequency weighting"
notes.append(f"The spectral centroid is {centroid:.0f} Hz, indicating that {centroid_desc}. The spectral bandwidth is {bandwidth:.0f} Hz, describing the spread of spectral energy.")

if voice["zcr"] >= 0.15:
    notes.append(f"The zero-crossing rate is relatively high ({voice['zcr']:.3f}); this can be associated with noisy, unvoiced, or high-frequency content and should be interpreted alongside the spectrum.")
elif voice["zcr"] <= 0.03:
    notes.append(f"The zero-crossing rate is low ({voice['zcr']:.3f}), indicating relatively few sign changes between adjacent samples.")
else:
    notes.append(f"The zero-crossing rate is {voice['zcr']:.3f}, a measure of how frequently the waveform changes sign.")

if voice["noise_dbfs_est"] is not None:
    notes.append(f"The estimated low-energy background level is {voice['noise_dbfs_est']:.1f} dBFS. The within-recording speech-to-background estimate is {voice['snr_est']:.1f} dB; treat it as approximate because quiet speech may be counted as background.")
else:
    notes.append("A separate low-energy background region was not detected, so a background-noise estimate could not be made reliably. A short room-tone pause before speaking can improve this estimate.")

for note in notes:
    st.write("• " + note)
st.caption("These observations are calculated from this recording's acoustic measurements. They are signal-quality descriptions, not medical, emotional, or speaker-identity assessments.")

with st.expander("Methods and project notes"):
    st.markdown("""
- **RMS and peak:** describe signal energy and maximum absolute amplitude.
- **Zero-crossing rate:** proportion of adjacent samples that change sign.
- **Pitch:** estimated from autocorrelation on active short-time frames; unvoiced segments are excluded.
- **FFT:** displays the recording's frequency content.
- **STFT:** shows how frequency content changes over time.
- **Silence/activity:** estimated using a short-time RMS threshold; quiet speech may be mistaken for silence.
- **Filtering:** moving average, fourth-order Butterworth IIR, Hamming-window FIR, and optional 50/60 Hz notch + FIR.
- **No-reference limitation:** without a clean copy of the same speech, true SNR and MSE cannot be measured. The filter table uses band-energy proxies only.
""")
