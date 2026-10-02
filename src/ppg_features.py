"""Shared preprocessing and 16-feature morphological extraction used for both datasets.

preprocess_ppg: drop first 1 s, mean-centre, zero-phase 3rd-order Butterworth 0.5-8 Hz,
automatic polarity (orientation with the more regular rhythm), min-max to [0,1].
extract_features: see Table 2 of the manuscript for definitions.
Area features are integrated in samples; divide by fs to obtain seconds.
"""
import numpy as np
import scipy.signal as ss
from scipy.stats import skew, kurtosis as kurt

def preprocess_ppg(sig, fs, lo=0.5, hi=8.0, order=3, auto_polarity=False):
    trim = int(1.0 * fs)
    sig = sig[trim:].copy()
    sig = sig - np.mean(sig)
    nyq = fs / 2
    b, a = ss.butter(order, [lo / nyq, hi / nyq], btype='band')
    filt = ss.filtfilt(b, a, sig)
    # polarity: pick orientation giving the more regular rhythm (STU-BA only)
    best_sign, best_cv = 1.0, np.inf
    for sign in ((1.0, -1.0) if auto_polarity else ()):
        x = sign * filt
        pk, _ = ss.find_peaks(x, distance=int(0.33 * fs), prominence=0.3 * np.std(x))
        if len(pk) < 4:
            continue
        ibi = np.diff(pk) / fs
        cv = np.std(ibi) / np.mean(ibi)
        if cv < best_cv:
            best_cv, best_sign = cv, sign
    filt = best_sign * filt
    filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-9)
    return filt


def extract_features(sig_raw, fs, auto_polarity=False):
    sig = preprocess_ppg(sig_raw, fs=fs, auto_polarity=auto_polarity)
    n = len(sig)
    feats = {}
    adaptive_height = np.percentile(sig, 60)
    min_distance = int(fs * 0.4)
    peaks_idx, _ = ss.find_peaks(sig, distance=min_distance, height=adaptive_height, prominence=0.05)
    if len(peaks_idx) < 3:
        adaptive_height = np.percentile(sig, 50)
        peaks_idx, _ = ss.find_peaks(sig, distance=min_distance, height=adaptive_height)
    if len(peaks_idx) < 2:
        return None
    ibi = np.diff(peaks_idx) / fs
    ibi = ibi[(ibi > 0.4) & (ibi < 1.5)]
    if len(ibi) < 1:
        return None
    feats['hr_bpm'] = np.mean(60 / ibi)
    feats['ibi_mean'] = ibi.mean()
    feats['ibi_std'] = ibi.std()
    feats['ibi_rmssd'] = np.sqrt(np.mean(np.diff(ibi) ** 2)) if len(ibi) > 1 else 0

    sys_amps, dias_amps, pulse_widths = [], [], []
    rise_times, fall_times, aucs_sys, aucs_dias = [], [], [], []
    for i in range(len(peaks_idx) - 1):
        p_sys, p_nxt = peaks_idx[i], peaks_idx[i + 1]
        cycle_len = p_nxt - p_sys
        if cycle_len < 10:
            continue
        search_back = max(0, p_sys - int(0.35 * fs))
        if search_back >= p_sys:
            continue
        foot = search_back + np.argmin(sig[search_back:p_sys])
        amp_sys = sig[p_sys] - sig[foot]
        if amp_sys <= 0.02:
            continue
        sys_amps.append(amp_sys)
        rise_times.append((p_sys - foot) / fs)
        search_fwd = min(n - 1, p_sys + int(0.7 * fs))
        next_foot = p_sys + np.argmin(sig[p_sys:search_fwd])
        fall_times.append((next_foot - p_sys) / fs)
        d_start = p_sys + int(0.35 * cycle_len)
        d_end = p_sys + int(0.75 * cycle_len)
        if d_end < n and d_start < d_end:
            notch = d_start + np.argmin(sig[d_start:d_end])
            dias_amps.append(sig[notch] - sig[foot])
        half_amp = sig[foot] + 0.5 * amp_sys
        above = sig[foot:p_nxt] >= half_amp
        trans = np.where(np.diff(above.astype(int)))[0]
        if len(trans) >= 2:
            pulse_widths.append((trans[-1] - trans[0]) / fs)
        if p_sys > foot:
            aucs_sys.append(np.trapezoid(sig[foot:p_sys + 1] - sig[foot]))
        if next_foot > p_sys:
            aucs_dias.append(np.trapezoid(sig[p_sys:next_foot + 1] - sig[foot]))

    def sm(lst):
        arr = [x for x in lst if x is not None and not np.isnan(x)]
        return np.mean(arr) if arr else np.nan

    feats['sys_amp'] = sm(sys_amps)
    feats['dias_amp'] = sm(dias_amps)
    feats['dias_ratio'] = sm([d / s for d, s in zip(dias_amps, sys_amps) if s > 0 and not np.isnan(d)])
    feats['pulse_width'] = sm(pulse_widths)
    feats['rise_time'] = sm(rise_times)
    feats['fall_time'] = sm(fall_times)
    feats['auc_sys'] = sm(aucs_sys)
    feats['auc_dias'] = sm(aucs_dias)
    feats['auc_ratio'] = (sm(aucs_sys) / (sm(aucs_dias) + 1e-9) if aucs_sys and aucs_dias else np.nan)
    feats['skewness'] = skew(sig)
    feats['kurtosis_'] = kurt(sig)
    f, pxx = ss.welch(sig, fs=fs, nperseg=min(4096, n))
    feats['spectral_ratio'] = pxx[(f >= 0.5) & (f < 2)].sum() / (pxx[(f >= 2) & (f < 8)].sum() + 1e-9)
    return feats


