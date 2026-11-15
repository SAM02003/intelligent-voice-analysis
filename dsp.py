"""Small, explicit DSP functions for the EC6675 mini-project."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import warnings

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile


MIN_RATE = 8_000
MAX_RATE = 48_000
MIN_SECONDS = 0.25
MAX_SECONDS = 20.0
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
METHODS = ("Moving average", "Butterworth", "FIR", "Wiener filter", "Spectral gating")


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray  # float64; shape = (frames, channels)
    sample_rate: int
    name: str
    clean_reference: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return self.samples.shape[0] / self.sample_rate


def validate_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Check finite mono/stereo samples, rate, and bounded duration."""
    if not isinstance(sample_rate, (int, np.integer)) or not MIN_RATE <= sample_rate <= MAX_RATE:
        raise ValueError(f"Sample rate must be {MIN_RATE:,}–{MAX_RATE:,} Hz.")
    data = np.asarray(samples)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[1] not in (1, 2):
        raise ValueError("Only mono and stereo WAV files are supported.")
    duration = data.shape[0] / sample_rate
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        raise ValueError(f"Audio must be {MIN_SECONDS}–{MAX_SECONDS:g} seconds long (received {duration:.2f} s).")
    if not np.issubdtype(data.dtype, np.number) or not np.all(np.isfinite(data)):
        raise ValueError("Audio contains invalid or non-finite sample values.")
    data = data.astype(np.float64, copy=False)
    if np.max(np.abs(data)) > 1.0001:
        raise ValueError("Audio samples exceed the supported ±1 range.")
    return data


def read_wav(raw: bytes, name: str = "uploaded.wav") -> AudioData:
    """Decode uncompressed PCM/IEEE float WAV; integer types scale to ±1."""
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a non-empty WAV file no larger than 25 MB.")
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("This is not a standard RIFF/WAVE file. Export it as uncompressed WAV.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", wavfile.WavFileWarning)
            sample_rate, data = wavfile.read(BytesIO(raw))
    except (ValueError, OSError, EOFError, wavfile.WavFileWarning) as exc:
        raise ValueError("Could not read WAV. Export as uncompressed PCM or float WAV.") from exc
    if data.dtype.kind == "u" and data.dtype.itemsize == 1:
        normalized = (data.astype(np.float64) - 128.0) / 128.0
    elif data.dtype.kind == "i" and data.dtype.itemsize in (2, 4, 8):
        # SciPy left-justifies 24-bit PCM in int32, so scaling by 2^31 works.
        normalized = data.astype(np.float64) / float(1 << (8 * data.dtype.itemsize - 1))
    elif data.dtype.kind == "f" and data.dtype.itemsize in (4, 8):
        normalized = data.astype(np.float64)
    else:
        raise ValueError("Unsupported WAV encoding. Use 8/16/24/32-bit PCM or float WAV.")
    normalized = validate_audio(normalized, int(sample_rate))
    if float(np.sqrt(np.mean(normalized**2))) < 1e-5:
        raise ValueError("Audio is essentially silent; choose a recording with audible content.")
    return AudioData(normalized, int(sample_rate), name)


def generate_test_signal(
    duration: float = 3.0,
    sample_rate: int = 16_000,
    noise_level: float = 0.25,
    noise_kind: str = "High-frequency hiss",
    seed: int = 7,
) -> AudioData:
    """Return a deterministic synthetic melody-like waveform plus known noise."""
    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    envelope = 0.65 + 0.35 * np.sin(2 * np.pi * 1.3 * t) ** 2
    clean = envelope * (
        0.42 * np.sin(2 * np.pi * 220 * t)
        + 0.23 * np.sin(2 * np.pi * 440 * t)
        + 0.11 * np.sin(2 * np.pi * 880 * t)
    )
    rng = np.random.default_rng(seed)
    if noise_kind == "High-frequency hiss":
        white = rng.standard_normal(n)
        noise = signal.sosfilt(signal.butter(3, 1_900, btype="highpass", fs=sample_rate, output="sos"), white)
    elif noise_kind == "White noise":
        noise = rng.standard_normal(n)
    elif noise_kind == "50 Hz hum + hiss":
        white = rng.standard_normal(n)
        hiss = signal.sosfilt(signal.butter(3, 1_900, btype="highpass", fs=sample_rate, output="sos"), white)
        noise = hiss + np.sin(2 * np.pi * 50 * t)
    else:
        raise ValueError("Unknown noise type.")
    noise = noise / np.sqrt(np.mean(noise**2)) * noise_level
    noisy = clean + noise
    if np.max(np.abs(noisy)) > 1:
        # A common scale avoids clipping while leaving SNR/MSE comparisons valid.
        scale = 0.98 / np.max(np.abs(noisy))
        clean, noisy = clean * scale, noisy * scale
    clean, noisy = clean[:, None], noisy[:, None]
    return AudioData(validate_audio(noisy, sample_rate), sample_rate, "simulated.wav", clean)


def validate_settings(sample_rate: int, cutoff_hz: float, moving_window: int, fir_taps: int) -> None:
    if not 100 <= cutoff_hz < 0.45 * sample_rate:
        raise ValueError("Cutoff must be at least 100 Hz and below 45% of sample rate.")
    if moving_window < 3 or moving_window % 2 != 1 or moving_window > 101:
        raise ValueError("Moving-average window must be odd, from 3 to 101 samples.")
    if fir_taps < 31 or fir_taps % 2 != 1 or fir_taps > 401:
        raise ValueError("FIR length must be odd, from 31 to 401 taps.")


def detect_hum(data: AudioData) -> int | None:
    """Find a prominent narrow peak near mains frequency; otherwise return None."""
    x = data.samples[: 4 * data.sample_rate]
    n = len(x)
    window = np.hanning(n)[:, None]
    power = np.mean(np.abs(np.fft.rfft(x * window, axis=0)) ** 2, axis=1)
    freq = np.fft.rfftfreq(n, 1 / data.sample_rate)
    low_band = (freq >= 30) & (freq <= 90)
    total = float(np.sum(power[low_band]))
    for target in (50, 60):
        near = np.abs(freq - target) <= max(2, 1.5 * data.sample_rate / n)
        neighbors = (np.abs(freq - target) >= 5) & (np.abs(freq - target) <= 15)
        if np.any(near) and np.any(neighbors):
            peak = float(np.max(power[near]))
            baseline = float(np.median(power[neighbors]))
            if peak > 16 * max(baseline, 1e-15) and peak > 0.18 * max(total, 1e-15):
                return target
    return None



def _stft_denoise_channel(x: np.ndarray, sample_rate: int, mode: str) -> np.ndarray:
    """STFT-domain Wiener or soft spectral-gating denoising for one channel."""
    x = np.asarray(x, dtype=np.float64)
    original_length = len(x)
    nperseg = min(512, max(128, 2 ** int(np.floor(np.log2(max(128, original_length // 4))))))
    nperseg = min(nperseg, original_length)
    if nperseg < 32:
        return x.copy()
    noverlap = min(3 * nperseg // 4, nperseg - 1)

    freqs, times, spectrum = signal.stft(
        x, fs=sample_rate, window="hann", nperseg=nperseg,
        noverlap=noverlap, boundary="zeros", padded=True
    )
    magnitude = np.abs(spectrum)
    power = magnitude ** 2

    # Estimate stationary background from the quietest 20% of time frames.
    frame_energy = np.mean(power, axis=0)
    quiet_count = max(1, int(np.ceil(0.20 * len(frame_energy))))
    quiet_indices = np.argsort(frame_energy)[:quiet_count]
    noise_power = np.mean(power[:, quiet_indices], axis=1, keepdims=True)
    eps = np.finfo(np.float64).eps

    if mode == "wiener":
        # Wiener gain: estimated clean-signal power / total observed power.
        gain = np.maximum(power - noise_power, 0.0) / (power + eps)
        gain = np.clip(gain, 0.05, 1.0)
        # Light smoothing reduces musical-noise fluctuations.
        gain = ndimage.gaussian_filter(gain, sigma=(0.8, 0.6))
    elif mode == "spectral":
        # Soft spectral gate with a conservative floor to avoid chopping speech.
        noise_magnitude = np.sqrt(noise_power)
        threshold = 1.5 * noise_magnitude
        gain = 1.0 - threshold / (magnitude + eps)
        gain = np.clip(gain, 0.10, 1.0)
        gain = ndimage.gaussian_filter(gain, sigma=(1.0, 0.8))
    else:
        raise ValueError("Unknown STFT denoising mode.")

    _, reconstructed = signal.istft(
        spectrum * gain, fs=sample_rate, window="hann",
        nperseg=nperseg, noverlap=noverlap, input_onesided=True,
        boundary=True
    )
    if len(reconstructed) < original_length:
        reconstructed = np.pad(reconstructed, (0, original_length - len(reconstructed)))
    return np.asarray(reconstructed[:original_length], dtype=np.float64)


def wiener_filter(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply conservative STFT Wiener filtering independently to each channel."""
    return np.column_stack([
        _stft_denoise_channel(x[:, channel], sample_rate, "wiener")
        for channel in range(x.shape[1])
    ])


def spectral_gating(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply soft STFT spectral gating independently to each channel."""
    return np.column_stack([
        _stft_denoise_channel(x[:, channel], sample_rate, "spectral")
        for channel in range(x.shape[1])
    ])

def apply_filters(data: AudioData, cutoff_hz: float, moving_window: int = 9, fir_taps: int = 101) -> dict[str, np.ndarray]:
    """Run classical filters and STFT denoisers independently on each recording."""
    validate_settings(data.sample_rate, cutoff_hz, moving_window, fir_taps)
    x = data.samples
    moving = ndimage.uniform_filter1d(x, size=moving_window, axis=0, mode="reflect")
    sos = signal.butter(4, cutoff_hz, btype="lowpass", fs=data.sample_rate, output="sos")
    butter = signal.sosfiltfilt(sos, x, axis=0)
    taps = signal.firwin(fir_taps, cutoff_hz, fs=data.sample_rate, window="hamming")
    fir = signal.filtfilt(taps, [1.0], x, axis=0)

    outputs = {
        "Moving average": moving,
        "Butterworth": butter,
        "FIR": fir,
        "Wiener filter": wiener_filter(x, data.sample_rate),
        "Spectral gating": spectral_gating(x, data.sample_rate),
    }
    hum_hz = detect_hum(data)
    if hum_hz is not None:
        b, a = signal.iirnotch(hum_hz, Q=30, fs=data.sample_rate)
        without_hum = signal.filtfilt(b, a, x, axis=0)
        outputs[f"Notch + FIR ({hum_hz} Hz)"] = signal.filtfilt(taps, [1.0], without_hum, axis=0)
    return outputs


def mse(clean: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean((clean - estimate) ** 2))


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    error_power = mse(clean, estimate)
    signal_power = float(np.mean(clean**2))
    if error_power == 0:
        return float("inf")
    if signal_power == 0:
        return float("-inf")
    return float(10 * np.log10(signal_power / error_power))


def reference_scores(data: AudioData, outputs: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    if data.clean_reference is None:
        raise ValueError("SNR/MSE require a known clean reference.")
    rows = []
    for name, y in {"Original noisy": data.samples, **outputs}.items():
        rows.append({"Method": name, "SNR (dB)": snr_db(data.clean_reference, y), "MSE": mse(data.clean_reference, y)})
    return rows


def _band_powers(x: np.ndarray, sample_rate: int, cutoff_hz: float, hum_hz: int | None = None) -> tuple[float, float, np.ndarray, float]:
    """Welch powers and passband magnitudes, averaged across channels."""
    f, power = signal.welch(x, fs=sample_rate, nperseg=min(4096, len(x)), axis=0)
    power = np.mean(power, axis=1)
    low = f <= 0.8 * cutoff_hz
    high = f >= 1.2 * cutoff_hz
    hum = np.abs(f - hum_hz) <= 4 if hum_hz is not None else np.zeros_like(f, dtype=bool)
    if hum_hz is not None:
        low &= np.abs(f - hum_hz) > 12  # also omit nearby spectral leakage
    if not np.any(low) or not np.any(high):
        raise ValueError("Cutoff leaves too little frequency range for proxy analysis.")
    return float(np.sum(power[low])), float(np.sum(power[high])), power[low], float(np.sum(power[hum]))


def proxy_scores(data: AudioData, outputs: dict[str, np.ndarray], cutoff_hz: float) -> tuple[list[dict[str, float | str]], str, str]:
    """No-reference diagnostics; never claim estimated SNR or MSE."""
    hum_hz = detect_hum(data)
    input_low, input_high, input_low_bins, input_hum = _band_powers(data.samples, data.sample_rate, cutoff_hz, hum_hz)
    floor = 1e-15
    rows = []
    for name, y in outputs.items():
        output_low, output_high, output_low_bins, output_hum = _band_powers(y, data.sample_rate, cutoff_hz, hum_hz)
        # Cap at 80 dB so a numerical floor is not reported as meaningful attenuation.
        reduction = min(80.0, 10 * np.log10(max(input_high, floor) / max(output_high, input_high * 1e-8, floor)))
        retention = 100 * output_low / max(input_low, floor)
        # Difference in power-spectrum magnitudes is only a preservation proxy.
        passband_change = 100 * np.linalg.norm(np.sqrt(output_low_bins) - np.sqrt(input_low_bins)) / max(np.linalg.norm(np.sqrt(input_low_bins)), floor)
        row = {"Method": name, "High-band reduction (dB)": float(reduction), "Low-band power retained (%)": float(retention), "Low-band spectrum change (%)": float(passband_change)}
        if hum_hz is not None:
            row["Hum reduction (dB)"] = float(min(80, 10 * np.log10(max(input_hum, floor) / max(output_hum, input_hum * 1e-8, floor))))
        rows.append(row)
    if input_high <= floor or input_high < 1e-5 * max(input_low, floor):
        winner = min(rows, key=lambda r: r["Low-band spectrum change (%)"])["Method"]
        reason = "Very little energy lies in the assumed noise band. The least-changing filter is shown; filtering may be unnecessary."
    else:
        eligible = [r for r in rows if r["Low-band spectrum change (%)"] <= 10]
        if eligible:
            winner = max(eligible, key=lambda r: r["High-band reduction (dB)"] + r.get("Hum reduction (dB)", 0))["Method"]
            reason = "Among filters changing the protected lower band by at most 10%, this gives the strongest measured noise-band reduction."
        else:
            winner = min(rows, key=lambda r: r["Low-band spectrum change (%)"])["Method"]
            reason = "No filter met the 10% lower-band-change limit, so this is the least-changing option."
    return rows, str(winner), reason


def recommend_with_reference(rows: list[dict[str, float | str]]) -> tuple[str, str]:
    original_mse = float(rows[0]["MSE"])
    best = min(rows[1:], key=lambda r: r["MSE"])
    if float(best["MSE"]) >= original_mse:
        return str(best["Method"]), "This is the best available method by MSE, but none improves on the noisy original."
    return str(best["Method"]), "Lowest MSE against the known clean signal; equivalently, highest output SNR."


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Export 16-bit PCM. Clip only at the final conversion, and report clipping in UI."""
    data = np.clip(samples, -1, 1)
    pcm = np.round(data * 32767).astype(np.int16)
    if pcm.ndim == 2 and pcm.shape[1] == 1:
        pcm = pcm[:, 0]
    buffer = BytesIO()
    wavfile.write(buffer, sample_rate, pcm)
    return buffer.getvalue()
