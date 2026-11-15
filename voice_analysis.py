"""Voice-specific measurements for short speech recordings (no model training)."""
from __future__ import annotations

import numpy as np
from scipy import signal


def mono_signal(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return x


def analyze_voice(samples: np.ndarray, sample_rate: int) -> dict:
    """Return transparent acoustic measurements; these are not medical scores."""
    x = mono_signal(samples)
    eps = 1e-12
    duration = len(x) / sample_rate
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    clipped = float(np.mean(np.abs(x) >= 0.999)) * 100

    frame_len = max(1, int(0.025 * sample_rate))
    hop = max(1, int(0.010 * sample_rate))
    if len(x) < frame_len:
        frames = x[None, :]
    else:
        starts = np.arange(0, len(x) - frame_len + 1, hop)
        frames = np.stack([x[s:s + frame_len] for s in starts])
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + eps)
    threshold = max(float(np.max(frame_rms)) * 0.035, 0.002)
    active = frame_rms > threshold
    active_ratio = float(np.mean(active)) if len(active) else 0.0
    silence_pct = 100.0 * (1.0 - active_ratio)
    active_seconds = duration * active_ratio

    # Short-time zero-crossing rate.
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    mean_zcr = float(np.mean(zcr))

    # Robust pitch estimate from autocorrelation of voiced 25 ms frames.
    pitches = []
    min_lag = max(1, int(sample_rate / 350))
    max_lag = min(frame_len - 2, int(sample_rate / 70))
    if max_lag > min_lag:
        for frame, is_active in zip(frames, active):
            if not is_active:
                continue
            frame = frame - np.mean(frame)
            frame = frame * np.hanning(len(frame))
            if np.sqrt(np.mean(frame ** 2)) < 0.005:
                continue
            corr = signal.correlate(frame, frame, mode="full", method="fft")[len(frame)-1:]
            if corr[0] <= eps:
                continue
            corr = corr / corr[0]
            lag = min_lag + int(np.argmax(corr[min_lag:max_lag + 1]))
            confidence = float(corr[lag])
            if confidence >= 0.30:
                pitches.append(sample_rate / lag)
    pitch_mean = float(np.median(pitches)) if pitches else None
    pitch_min = float(np.percentile(pitches, 10)) if pitches else None
    pitch_max = float(np.percentile(pitches, 90)) if pitches else None

    # Spectrum and spectral descriptors.
    nperseg = min(2048, len(x))
    f, pxx = signal.welch(x, fs=sample_rate, nperseg=max(256, nperseg) if len(x) >= 256 else len(x))
    if len(pxx) > 1:
        power_sum = float(np.sum(pxx)) + eps
        centroid = float(np.sum(f * pxx) / power_sum)
        bandwidth = float(np.sqrt(np.sum(((f - centroid) ** 2) * pxx) / power_sum))
        cumulative = np.cumsum(pxx)
        rolloff = float(f[min(np.searchsorted(cumulative, 0.85 * cumulative[-1]), len(f)-1)])
        dominant = float(f[1 + np.argmax(pxx[1:])]) if len(f) > 1 else 0.0
    else:
        centroid = bandwidth = rolloff = dominant = 0.0

    # Estimate a noise floor from low-energy frames only; if none exist, say unavailable.
    if np.any(~active):
        noise_rms = float(np.median(frame_rms[~active]))
        noise_dbfs = 20 * np.log10(max(noise_rms, eps))
        snr_est = 20 * np.log10(max(float(np.median(frame_rms[active])) if np.any(active) else rms, eps) / max(noise_rms, eps))
    else:
        noise_rms = noise_dbfs = snr_est = None

    return {
        "duration": duration, "rms": rms, "peak": peak, "clipping_pct": clipped,
        "silence_pct": silence_pct, "speech_activity_pct": active_ratio * 100,
        "active_seconds": active_seconds, "zcr": mean_zcr,
        "pitch_mean": pitch_mean, "pitch_min": pitch_min, "pitch_max": pitch_max,
        "spectral_centroid": centroid, "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff, "dominant_frequency": dominant,
        "noise_rms_est": noise_rms, "noise_dbfs_est": noise_dbfs,
        "snr_est": snr_est, "frame_rms": frame_rms, "frame_active": active,
    }


def spectrogram_figure_data(samples: np.ndarray, sample_rate: int):
    x = mono_signal(samples)
    nperseg = min(512, len(x))
    if nperseg < 16:
        return np.array([0.0]), np.array([0.0]), np.zeros((1, 1))
    f, t, z = signal.stft(x, fs=sample_rate, nperseg=nperseg, noverlap=int(nperseg * 0.75))
    db = 20 * np.log10(np.maximum(np.abs(z), 1e-7))
    return f, t, db
