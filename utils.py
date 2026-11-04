import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import maximum_filter



def to_spectrogram(audio_file, sr, hop_length, window_length, n_fft):
    """
    Time-frequency representation
    Args:
        audio_file: np.ndarray
        sr: int
        hop_length: int
        window_length: int
        n_fft = int
    Returns:
        np.ndarray
    """
    y, sr = librosa.load(audio_file, sr=sr)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=window_length, window="hann", center=False)
    spectrogram = librosa.amplitude_to_db((np.abs(stft) + 1e-10)) # Log magnitude

    return spectrogram


def constellation(spectrogram, level_threshold, neighbourhood_size):
    """
    Extract prominent peaks from spectrogram
    """
    local_max = maximum_filter(spectrogram, size=neighbourhood_size)
    is_peak = (spectrogram == local_max) & (spectrogram > level_threshold)
    peaks = np.argwhere(is_peak)

    return peaks


def fingerprints(hashes, song_id, fingerprint_index):
    for hash_val, t_anchor in hashes:
        if hash_val not in fingerprint_index:
            fingerprint_index[hash_val] = []
        fingerprint_index[hash_val].append((song_id, t_anchor))
    return fingerprint_index


def generate_hashes(peaks, fan_out, min_time_delta, max_time_delta, max_freq_delta):
    """
    Combanatorial Hashing aka peak pairing
    """
    peaks = peaks[np.argsort(peaks[:, 1])] # Sort peaks by time
    
    hashes = []
    for i in range(len(peaks)): 
        f_anchor, t_anchor = peaks[i]
        pairs_found = 0       
        
        # Takes on F that are closest in time to the anchor 
        for j in range(i + 1, len(peaks)):
            f_target, t_target = peaks[j]
            delta_t = t_target - t_anchor 

            if delta_t > max_time_delta:
                break
            
            # Skip peaks that are too close to the anchor
            if delta_t < min_time_delta:
                continue

            # Skips peaks outside the target zone's frequency range
            if abs(int(f_target) - int(f_anchor)) > max_freq_delta:
                continue

            # create hash: 32 bit unsigned int
            hash_val = (int(f_anchor & 0x3FF) << 20) | (int(f_target & 0x3FF) << 10) | int(delta_t & 0x3FF)
            hash_val_32 = np.uint32(hash_val)
            hashes.append((int(hash_val_32), int(t_anchor)))

            pairs_found += 1

            if pairs_found >= fan_out:
                break
    
    return hashes
    
def find_matches(query_hashes, reference_fingerprints):
    """
    Use query to find matches to reference files via reverse index
    """
    matches = []
    for q_hash, query_time in query_hashes:
        if q_hash in reference_fingerprints:
            for song_id, ref_time in reference_fingerprints[q_hash]:
                matches.append((song_id, query_time, ref_time))        
    return matches

def time_diff_histogram(matches):
    """
    Count matches per (song_id, offset). The highest count is the best alignment.
    """
    counts = {}
    for song_id, query_time, ref_time in matches:
        delta_tk = ref_time - query_time
        key = (song_id, delta_tk)
        if key not in counts:
            counts[key] = 0
        counts[key] += 1
    return counts



def score_and_align(counts, top_n):
    """
    Rank songs by their best offset count (strongest diagonal).
    """
    best = {}
    for (song_id, offset), count in counts.items():
        if song_id not in best or count > best[song_id][0]:
            best[song_id] = (count, offset)

    ranking = [(song_id, count, offset) for song_id, (count, offset) in best.items()]
    ranking.sort(key=lambda r: r[1], reverse=True)
    return ranking[:top_n]



def plot_match(query, reference, sr, hop_length, n_fft):
    """
    Show which peaks produced a match and the offset histogram behind its score.
    query and reference are (peaks, hashes) tuples.
    """
    query_peaks, query_hashes = query
    reference_peaks, reference_hashes = reference

    counts = time_diff_histogram(find_matches(query_hashes, fingerprints(reference_hashes, 0, {})))
    _, score, offset = score_and_align(counts, top_n=1)[0]

    # Hashes in the winning offset bin: these are the score
    reference_lookup = set(reference_hashes)
    matched = [(h, t) for h, t in query_hashes if (h, t + offset) in reference_lookup]

    to_seconds = hop_length / sr
    to_hz = sr / n_fft
    fig, (ax_peaks, ax_hist) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [2, 1]})

    # Left: both sets of peaks, with the query slid to the winning offset
    query_times = (query_peaks[:, 1] + offset) * to_seconds
    ax_peaks.scatter(reference_peaks[:, 1] * to_seconds, reference_peaks[:, 0] * to_hz,
                     s=30, color='#a3a29c', label='Reference peaks')
    ax_peaks.scatter(query_times, query_peaks[:, 0] * to_hz,
                     s=45, marker='+', color='#2a78d6', label='Query peaks (shifted by offset)')

    for i, (h, t_anchor) in enumerate(matched):
        f_anchor, f_target, delta_t = h >> 20, (h >> 10) & 0x3FF, h & 0x3FF
        start = t_anchor + offset
        ax_peaks.plot([start * to_seconds, (start + delta_t) * to_seconds],
                      [f_anchor * to_hz, f_target * to_hz],
                      color='#eb6834', linewidth=2, marker='o', markersize=8,
                      label='Matched hash pairs' if i == 0 else None)

    ax_peaks.set_xlim(query_times.min() - 0.5, query_times.max() + 0.5)
    ax_peaks.set(xlabel='Time in reference (s)', ylabel='Frequency (Hz)')
    ax_peaks.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)

    # Right: a real match has one tall bar above the rest
    ax_hist.bar([key[1] * to_seconds for key in counts], list(counts.values()), width=0.15, color='#2a78d6')
    ax_hist.set(xlabel='Offset (s)', ylabel='Matching hashes')
    ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    plt.show()
