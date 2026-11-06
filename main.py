import pickle
from pathlib import Path
from utils import (to_spectrogram, constellation, fingerprints, 
                   generate_hashes, find_matches, time_diff_histogram, score_and_align)
                    

"""
Reference: 
An Industrial-Strength Audio Search Algorithm
Avery Li-Chun Wang
Published: 2003

"""

# Configuration parameters
SR = 44100
WINDOW_LENGTH = 2048
N_FFT = 2048
HOP_LENGTH = 1024
TOP_N = 5


# peak feature filter parameters
LEVEL_THRESHOLD_DB = 2
NEIGHBOURHOOD_SIZE = 20

# pairing filter parameters
FAN_OUT = 10
MIN_TIME_DELTA = 1 # 23ms
MAX_TIME_DELTA = 200 # ~4s
MAX_FREQ_DELTA_HZ= 2000 #Hz
MAX_FREQ_DELTA = round(MAX_FREQ_DELTA_HZ * N_FFT / SR)  # bins



def get_peaks_and_hashes(audio_file):
    spectrogram = to_spectrogram(audio_file=audio_file, sr=SR, hop_length=HOP_LENGTH,
                                 window_length=WINDOW_LENGTH, n_fft=N_FFT)
    # get peaks, constellation, create_fingerprint
    peaks = constellation(spectrogram, level_threshold=LEVEL_THRESHOLD_DB, neighbourhood_size=NEIGHBOURHOOD_SIZE)
    hashes = generate_hashes(peaks, fan_out=FAN_OUT, min_time_delta = MIN_TIME_DELTA, max_time_delta = MAX_TIME_DELTA, max_freq_delta=MAX_FREQ_DELTA)
    return peaks, hashes

def process_file(audio_file):
    return get_peaks_and_hashes(audio_file)[1]


if __name__ == "__main__":
    # Process reference audio files
    reference_audio_dir = Path("./audio_data/database")
    ref_audio_files = sorted(reference_audio_dir.glob("*.mp3") )
    reference_fingerprints = Path("./audio_fingerprint_data/fingerprint.pkl")

    song_names = []
    # Create a database of reference hashes
    if not reference_fingerprints.exists():
        ref_fingerprint_idx = {}
        for song_id, ref_audio in enumerate(ref_audio_files):
            print(ref_audio.name)
            song_names.append(ref_audio.stem)
            ref_hashes = process_file(ref_audio)
            fingerprints(ref_hashes, song_id, ref_fingerprint_idx)


        reference_fingerprints.parent.mkdir(parents=True, exist_ok=True)
        with open(reference_fingerprints, "wb") as f:
            pickle.dump((song_names, dict(ref_fingerprint_idx)), f)

    else:
        with open(reference_fingerprints, "rb") as f:
            song_names, ref_fingerprint_idx = pickle.load(f)
    
    # Process query audio files
    query_audio_dir = Path("./audio_data/query")
    query_audio_files = sorted(query_audio_dir.glob("*.mp3") )

    correct_top1 = 0
    correct_top5 = 0
    false_positive_count = 0
    false_positive_tracks = []

    for song_id, query_audio in enumerate(query_audio_files):
        query_hashes = process_file(query_audio)
        matches = find_matches(query_hashes, ref_fingerprint_idx)
        counts = time_diff_histogram(matches)
        rank = score_and_align(counts, top_n=TOP_N)

        print(f"Query song: {query_audio.name}")
        print("Match: \n")
        for i, (ref_id, score, offset) in enumerate(rank, start=1):
            song_start = offset * HOP_LENGTH / SR
            print(f"  {i}. {song_names[ref_id]}.mp3  (score: {score}, starts at: {song_start:.2f}s) \n")

        expected_song_name = query_audio.stem.replace("_query", "")
        ranked_names = [song_names[ref_id] for ref_id, _, _ in rank]

        if ranked_names[:1] == [expected_song_name]:
            correct_top1 += 1
        elif ranked_names:
            false_positive_tracks.append((query_audio.name, ranked_names[0]))

        if expected_song_name in ranked_names:
            correct_top5 += 1

  

    print(f"Top-1: {correct_top1}/{len(query_audio_files)}")
    print(f"Top-5: {correct_top5}/{len(query_audio_files)}")
    print(f"Accuracy: {correct_top1 / len(query_audio_files) * 100:.2f}%")
    # print(f"False positive: {len(false_positive_tracks)}/{len(query_audio_files)}")
    # print("False positive tracks:")
    # for query_name, returned_song in false_positive_tracks:
    #     print(f"  - {query_name} returned {returned_song}.mp3")

