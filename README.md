# Intelligent Voice Analysis & Enhancement (EC6675)

A Streamlit application for analyzing and enhancing short speech recordings. It retains the original EC6675 DSP filtering work and adds browser microphone recording, voice measurements, FFT/STFT visualizations, and before/after playback.

## Features

- Record speech directly in the browser with Streamlit's microphone input, or upload a WAV file.
- Accepts recordings from 0.25 to 20 seconds, mono/stereo, 8–48 kHz, up to 25 MB.
- Browser recordings are received as WAV audio; no manual conversion is required.
- Measures duration, RMS, peak amplitude, clipping percentage, zero-crossing rate, speech activity/silence, estimated pitch, spectral centroid, bandwidth and roll-off.
- Shows waveform, FFT spectrum and STFT spectrogram.
- Runs Moving Average, Butterworth IIR, Hamming-window FIR, and optional 50/60 Hz Notch + FIR filtering.
- Plays original and enhanced audio and allows WAV download.
- Uses no-reference frequency-band metrics for uploaded speech. It does **not** claim true SNR/MSE without a clean reference.

## Setup (Windows)

1. Extract the ZIP file.
2. Open Command Prompt or PowerShell in the extracted `intelligent-audio-analyzer` folder.
3. Create and activate a virtual environment (recommended):

```bash
py -m venv .venv
.venv\Scripts\activate
```

4. Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Run:

```bash
streamlit run app.py
```

6. Open the local URL printed in the terminal. Allow microphone permission when prompted.

## Recording protocol for the demonstration

Ask the speaker to record a natural utterance lasting no more than 20 seconds. A short pause before speaking is useful because low-energy frames can help estimate the background-noise floor. The app rejects recordings longer than 20 seconds after recording is submitted.

## Important interpretation limits

- Pitch estimation is approximate and may fail on unvoiced, breathy, very quiet or noisy segments.
- Silence detection uses an RMS threshold; quiet speech can be classified as silence.
- The displayed speech-to-background ratio is only a rough estimate from low-energy frames, not a calibrated laboratory SNR.
- Without a clean reference recording of the same utterance, true SNR and MSE cannot be computed.
- The current build does not perform speech transcription or claim emotion/health analysis. The project prioritizes dependable DSP analysis and enhancement within the short implementation window.
- Low-pass filtering can remove useful speech detail. Listen to the output before deciding it is better.

## Main files

- `app.py`: Streamlit user interface and plots.
- `dsp.py`: Audio validation, filters, FFT-related scoring and WAV handling.
- `voice_analysis.py`: Voice-specific acoustic measurements and STFT data.
- `ai.py`: Original synthetic-noise KNN demonstration module (not used to label real speech recordings as a reliable noise class).
- `tests/test_dsp.py`: Original DSP regression tests.
