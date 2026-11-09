import numpy as np
import librosa
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

            # Find ratio of anchor to target frequency
            f_relative = f_anchor / f_target
            # Encode float to int
            # 6 significant figures fits into 20 bits safely
            f_relative_encoded = int(round(f_relative * 10_000))
            f_relative_encoded_masked = int(f_relative_encoded & 0xFFFFF)
            assert(f_relative_encoded == f_relative_encoded_masked) # Check the encoding did not lose information

            # create hash: 32 bit unsigned int
            hash_val = (f_relative_encoded_masked << 10) | int(delta_t & 0x3FF)
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

def score_and_align(counts, top_n, min_score=0):
    """
    Rank songs by their best offset count (strongest diagonal).
    """
    best = {}
    for (song_id, offset), count in counts.items():
        if song_id not in best or count > best[song_id][0]:
            best[song_id] = (count, offset)

    ranking = [(song_id, count, offset) for song_id, (count, offset) in best.items()
               if count >= min_score]
    ranking.sort(key=lambda r: r[1], reverse=True)
    return ranking[:top_n]
