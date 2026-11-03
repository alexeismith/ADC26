import numpy as np
import glob
import pickle
from pathlib import Path
from utils import to_spectrogram, constellation, generate_hashes, find_matches, time_diff_histogram, score_and_align
                    

"""
Reference: 
An Industrial-Strength Audio Search Algorithm
Avery Li-Chun Wang
Published: 2003

"""

SR = 44100
WINDOW_LENGTH = 2048
N_FFT = 2048
HOP_LENGTH = 1024
TOP_N = 5


def process_file(audio_file):
    spectrogram = to_spectrogram(audio_file=audio_file, sr=SR, hop_length=HOP_LENGTH,
                                 window_length=WINDOW_LENGTH, n_fft=N_FFT)
    # get peaks, constellation, create_fingerprint
    peaks = constellation(spectrogram)
    hashes = generate_hashes(peaks)
    return hashes

def fingerprints(hashes, song_id, fingerprint_index):
    for hash_val, t_anchor in hashes:
        if hash_val not in fingerprint_index:
            fingerprint_index[hash_val] = []
        fingerprint_index[hash_val].append((song_id, t_anchor))
    return fingerprint_index



if __name__ == "__main__":
    # Process reference audio files
    reference_audio_dir = Path("./audio_data/database")
    ref_audio_files = sorted(reference_audio_dir.glob("*.mp3") )
    reference_fingerprints = Path("./fingerprint/fingerprint.pkl")

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

    correct_match = 0
    for song_id, query_audio in enumerate(query_audio_files):
        query_hashes = process_file(query_audio)
        matches = find_matches(query_hashes, ref_fingerprint_idx)
        counts = time_diff_histogram(matches)
        rank = score_and_align(counts, top_n=TOP_N)

        print(f"Query song: {query_audio.name}")
        print("Match:")
        for i, (ref_id, score, offset) in enumerate(rank, start=1):
            print(f"  {i}. {song_names[ref_id]}.mp3  (score: {score})")
        print()

        # Query files are named <reference>_query.mp3, so the filename gives the right answer
        if rank and song_names[rank[0][0]] == query_audio.stem.replace("_query", ""):
            correct_match += 1

    print(f"Top-1: {correct_match}/{len(query_audio_files)}")
