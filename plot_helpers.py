import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from matplotlib.ticker import MaxNLocator
from utils import time_diff_histogram, fingerprints, score_and_align, find_matches

def plot_match_overview(query, reference, sr, hop_length, n_fft):
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
    fig, (ax_peaks, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))

    # Left: both sets of peaks, with the query slid to the winning offset
    query_times = (query_peaks[:, 1] + offset) * to_seconds
    ax_peaks.scatter(reference_peaks[:, 1] * to_seconds, reference_peaks[:, 0] * to_hz, color='grey', label='Reference peaks')
    ax_peaks.scatter(query_times, query_peaks[:, 0] * to_hz,
                    marker='+', color='blue', label='Query peaks (shifted by offset)')

    for i, (h, t_anchor) in enumerate(matched):
        f_anchor, f_target, delta_t = h >> 20, (h >> 10) & 0x3FF, h & 0x3FF
        start = t_anchor + offset
        ax_peaks.plot([start * to_seconds, (start + delta_t) * to_seconds],
                      [f_anchor * to_hz, f_target * to_hz],
                      color='red', marker='o',
                      label='Matched hash pairs' if i == 0 else None)

    ax_peaks.set_xlim(query_times.min() - 0.5, query_times.max() + 0.5)
    ax_peaks.set(xlabel='Time in reference', ylabel='Frequency')
    ax_peaks.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3)

    # Right: a real match has one tall bar above the rest
    ax_hist.bar([key[1] * to_seconds for key in counts], list(counts.values()), width=0.15, color='blue')
    ax_hist.set(xlabel='Offset (s)', ylabel='Matching hashes')
    ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_spectrogram(spectrogram, hop_length, sample_rate, n_fft):
    time_axis = np.arange(spectrogram.shape[1] + 1) * 2 * hop_length / sample_rate
    frequency_axis = np.arange(spectrogram.shape[0] + 1) * sample_rate / n_fft

    fig, ax = plt.subplots(1, figsize=(14, 6))
    mesh = ax.pcolormesh(time_axis, frequency_axis, spectrogram, cmap="magma")
    cbar = fig.colorbar(mesh)
    cbar.set_label("level [dB]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_ylim(top=20000)


def plot_spectrogram_and_constellation(spectrogram, constellation_x, constellation_y):
    fig, ax = plt.subplots(1, figsize=(14, 6))
    mesh = ax.pcolormesh(spectrogram, cmap="magma")
    ax.scatter(constellation_x, constellation_y, marker="x", color="w")
    cbar = fig.colorbar(mesh)
    cbar.set_label("level [dB]")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_yticks([])
    ax.set_xticks([])


def plot_constellations(
    constellation_x1, constellation_y1, constellation_x2, constellation_y2
):
    fig, ax = plt.subplots(1, figsize=(12, 6))
    ax.scatter(constellation_x1, constellation_y1)
    ax.scatter(constellation_x2, constellation_y2, marker="x", color="r")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_yticks([])
    ax.set_xticks([])

def plot_hashes_sets(hashes_sets, markers, min_value, max_value):
    def to_hex(x, pos):
        return "%X" % int(x)

    assert len(hashes_sets) == len(markers), (
        f"the number of sets of hashes: {len(hashes_sets)} needs to equal the number of markers: {len(markers)}"
    )

    fig, ax = plt.subplots(1, figsize=(14, 2))
    ax.axis((min_value, max_value, -1, 1))
    ax.set_xlabel("Hash Values")
    ax.set_yticks([])
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(to_hex))
    for hashes in hashes_sets:
        marker = markers.pop(0)
        ax.scatter(
            np.array(hashes, dtype=np.int32),
            np.zeros_like(hashes, dtype=np.int8),
            s=100,
            alpha=0.6,
            marker=marker,
        )

def plot_histogram(x_values, counts):
    fig, ax = plt.subplots(1, figsize=(12,6))
    ax.bar(x_values, counts)
    ax.set_xlabel("t⚓ reference - t⚓ query [scaling_factor * s]")
    ax.set_ylabel("Counts")
    ax.get_yaxis().set_major_formatter(ticker.FormatStrFormatter('%d'))

def plot_anchor_times(delta_reference, delta_query):
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.scatter(delta_reference, delta_query)
    ax.set_xlabel("t⚓ reference")
    ax.set_ylabel("t⚓ query")
    ax.set_xticks([])
    ax.set_yticks([])
