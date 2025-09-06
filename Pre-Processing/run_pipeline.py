import warnings
warnings.filterwarnings("ignore")


import soundfile as sf
import time
import os
import librosa
import numpy as np
import madmom
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

from PIL import Image
import matplotlib.pyplot as plt







def make_fixed_size_spectrogram(y, sr, n_seconds=3.0, width=128, n_mels=128, hop_length=512):
    """
    Create fixed-size mel spectrogram images (not beat aligned) using
    a single spectrogram computation + interpolation.
    
    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    n_seconds : float
        Duration (in seconds) of each spectrogram window.
    width : int
        Number of time bins (pixels) in the spectrogram.
    n_mels : int
        Number of mel frequency bins.
    hop_length : int
        Hop length used to compute the base spectrogram.
    
    Returns
    -------
    specs : list of np.ndarray
        Each spectrogram of shape (n_mels, width).
    """
    # Use fixed hop_length (512 by default)
    n_fft = hop_length * 4

    # Compute mel spectrogram once
    mel = librosa.feature.melspectrogram(y=y, sr=sr,
                                         n_fft=n_fft,
                                         hop_length=hop_length,
                                         n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Window length in frames
    window_length_frames = int(round(n_seconds * sr / hop_length))

    specs = []
    for start in range(0, mel_db.shape[1] - window_length_frames + 1, window_length_frames):
        end = start + window_length_frames
        segment = mel_db[:, start:end]

        if segment.shape[1] < 2:
            continue

        # Interpolate to fixed width
        x_old = np.linspace(0, 1, segment.shape[1])
        x_new = np.linspace(0, 1, width)
        segment_resampled = np.vstack([
            np.interp(x_new, x_old, row) for row in segment
        ])

        specs.append(segment_resampled)

    return specs



def make_fixed_size_chromagrams(y, sr, n_seconds=3.0, width=128, hop_length=512):
    """
    Create fixed-size chromagram images (not beat aligned) using
    a single chromagram computation + interpolation.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    sr : int
        Sampling rate.
    n_seconds : float
        Duration (in seconds) of each window.
    width : int
        Number of chroma frames allocated per window (output width).
    hop_length : int
        Hop length used to compute the base chromagram.

    Returns
    -------
    images : list of np.ndarray
        Each image has shape (12, width).
    """
    # Compute base chromagram once
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # Get frame times
    frame_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
    )

    # Window length in frames
    window_length_frames = int(round(n_seconds * sr / hop_length))

    images = []
    for start in range(0, chroma.shape[1] - window_length_frames + 1, window_length_frames):
        end = start + window_length_frames
        segment = chroma[:, start:end]

        if segment.shape[1] < 2:
            continue

        # Resample to fixed width
        x_old = np.linspace(0, 1, segment.shape[1])
        x_new = np.linspace(0, 1, width)
        segment_resampled = np.vstack([
            np.interp(x_new, x_old, row) for row in segment
        ])

        images.append(segment_resampled)

    return images





def make_downbeat_aligned_spectrogram(y, sr, downbeat_times, time_bins_per_downbeat=32, downbeats_in_image=4):
    """
    Create log-mel spectrogram images aligned to downbeats.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    sr : int
        Sampling rate.
    downbeat_times : list of float
        Times (in seconds) of detected downbeats.
    time_bins_per_downbeat : int
        Number of spectrogram frames allocated between two downbeats.
    downbeats_in_image : int
        How many downbeats per image.

    Returns
    -------
    images : list of np.ndarray
        Each image has shape (128, time_bins_per_downbeat * downbeats_in_image).
    """
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Map downbeat times -> spectrogram frames
    hop_length = 512
    frame_times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
    downbeat_frames = np.searchsorted(frame_times, downbeat_times)

    segments = []
    for i in range(len(downbeat_frames) - 1):
        start, end = downbeat_frames[i], downbeat_frames[i + 1]
        segment = S_db[:, start:end]

        if segment.shape[1] < 2:
            continue

        # Resample along time axis to exactly time_bins_per_downbeat
        x_old = np.linspace(0, 1, segment.shape[1])
        x_new = np.linspace(0, 1, time_bins_per_downbeat)
        segment_resampled = np.vstack([
            np.interp(x_new, x_old, row) for row in segment
        ])

        segments.append(segment_resampled)

    # Group consecutive segments into images
    images = []
    for i in range(0, len(segments) - downbeats_in_image + 1, downbeats_in_image):
        image = np.hstack(segments[i:i + downbeats_in_image])
        images.append(image)

    return images



def make_downbeat_aligned_chromagrams(y, sr, downbeat_times, time_bins_per_downbeat=128, downbeats_in_image=1):
    """
    Create chromagram images aligned to downbeats.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    sr : int
        Sampling rate.
    downbeat_times : list of float
        Times (in seconds) of detected downbeats.
    time_bins_per_downbeat : int
        Number of chroma frames allocated between two downbeats.
    downbeats_in_image : int
        How many downbeats per image.

    Returns
    -------
    images : list of np.ndarray
        Each image has shape (12, time_bins_per_downbeat * downbeats_in_image).
    """
    # Compute chromagram
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # Map downbeat times -> chroma frames
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    downbeat_frames = np.searchsorted(frame_times, downbeat_times)

    segments = []
    for i in range(len(downbeat_frames) - 1):
        start, end = downbeat_frames[i], downbeat_frames[i + 1]
        segment = chroma[:, start:end]

        if segment.shape[1] < 2:
            continue

        # Resample along time axis to exactly time_bins_per_downbeat
        x_old = np.linspace(0, 1, segment.shape[1])
        x_new = np.linspace(0, 1, time_bins_per_downbeat)
        segment_resampled = np.vstack([
            np.interp(x_new, x_old, row) for row in segment
        ])

        segments.append(segment_resampled)

    # Group consecutive segments into images
    images = []
    for i in range(0, len(segments) - downbeats_in_image + 1, downbeats_in_image):
        image = np.hstack(segments[i:i + downbeats_in_image])
        images.append(image)

    return images



# Map note names to chroma indices
NOTE_TO_INDEX = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

def rotate_chroma_to_C_major(chroma_image, key_str):
    """
    Rotate a chromagram so that the song is aligned to C major (or A minor).

    Parameters
    ----------
    chroma_image : np.ndarray
        A chromagram image with shape (12, width).
    key_str : str
        Detected key, e.g. 'G major' or 'A minor'.

    Returns
    -------
    rotated : np.ndarray
        Chromagram rotated so tonic aligns with C major.
    """

    # Parse key
    parts = key_str.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Unexpected key format: {key_str}")
    tonic, mode = parts[0], parts[1].lower()

    if tonic not in NOTE_TO_INDEX:
        raise ValueError(f"Unrecognized tonic: {tonic}")

    tonic_index = NOTE_TO_INDEX[tonic]

    if mode == "major":
        offset = tonic_index
    elif mode == "minor":
        # shift to relative major (3 semitones up)
        offset = (tonic_index + 3) % 12
    else:
        raise ValueError(f"Mode must be 'major' or 'minor', got: {mode}")

    # Rotate so tonic maps to C (index 0)
    rotated = np.roll(chroma_image, -offset, axis=0)

    return rotated



def downloadImagesArray(imgs, prefix, folder):
    
    os.makedirs(folder, exist_ok=True)
    
    for i, arr in enumerate(imgs):
        # If the array is float (e.g. values in [0,1] or arbitrary floats), scale to 0-255
        if np.issubdtype(arr.dtype, np.floating):
            arr = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype(np.uint8)
        
        pil_img = Image.fromarray(arr)
        pil_img.save(f"{folder}/{prefix}_{i}.png")


def find_audio_files(root_dir, extensions=(".wav", ".mp3")):
    """
    Recursively search a directory for audio files.

    Parameters
    ----------
    root_dir : str
        The top-level directory to search.
    extensions : tuple of str, optional
        File extensions to look for (default: (".wav", ".mp3")).

    Returns
    -------
    list of str
        Full file paths for all matching audio files.
    """
    audio_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(extensions):
                audio_files.append(os.path.join(dirpath, f))
    return audio_files



def processFile(filename, output_folder):
    """
    Process a single audio file using globally instantiated ML models.
    """

    # Use global models instead of re-instantiating them every time
    global key_proc, downbeat_rnn, downbeat_tracker

    # Extract title and load audio
    
    y, sr = librosa.load(filename, sr=44100, mono=True)

    # Downbeat detection (reuse global RNN + tracker)
    downbeat_proc = downbeat_rnn(filename)
    beats = downbeat_tracker(downbeat_proc)
    downbeat_times = beats[beats[:, 1] == 1, 0]

    # Key recognition (reuse global CNN model)
    prediction = key_proc(filename)
    key = key_prediction_to_label(prediction)


    ## Settings for data generation
    fixed_window_n_seconds = 3.0


    fixed_spectros = make_fixed_size_spectrogram(y, sr, n_seconds=fixed_window_n_seconds, width=128, n_mels=128)
    downbeat_spectros = make_downbeat_aligned_spectrogram(y, sr, downbeat_times, time_bins_per_downbeat=128, downbeats_in_image=1)

    fixed_chromas = make_fixed_size_chromagrams(y, sr, n_seconds=fixed_window_n_seconds, width=128, hop_length=512)
    downbeat_chromas = make_downbeat_aligned_chromagrams(y, sr, downbeat_times, time_bins_per_downbeat=128, downbeats_in_image=1)
    rotated_fixed_chromas = [rotate_chroma_to_C_major(chroma_image, key) for chroma_image in fixed_chromas]
    rotated_downbeat_chromas = [rotate_chroma_to_C_major(chroma_image, key) for chroma_image in downbeat_chromas]


    ## Save results
    
    downloadImagesArray(fixed_spectros, "unaligned_spectrogram", output_folder)
    downloadImagesArray(downbeat_spectros, "aligned_spectrogram", output_folder)
    downloadImagesArray(fixed_chromas, "unaligned_chromagram", output_folder)
    downloadImagesArray(downbeat_chromas, "aligned_chromagram", output_folder)
    downloadImagesArray(rotated_fixed_chromas, "rotated_unaligned_chromagram", output_folder)
    downloadImagesArray(rotated_downbeat_chromas, "rotated_aligned_chromagram", output_folder)
    
    


def checkDataExists(folder):
    """
    Checks if processed data exists in the given results folder.
    Returns True if all conditions are satisfied, otherwise False.
    """

    required_prefixes = [
        "unaligned_spectrogram",
        "aligned_spectrogram",
        "unaligned_chromagram",
        "aligned_chromagram",
        "rotated_unaligned_chromagram",
        "rotated_aligned_chromagram"
    ]

    if not os.path.exists(folder):
        return False

    files = os.listdir(folder)
    if not files:
        return False

    # only allow .png files
    png_files = [f for f in files if f.endswith(".png")]
    if len(png_files) != len(files):
        return False
    if not png_files:
        return False

    # group by prefix
    prefix_to_indexes = {prefix: [] for prefix in required_prefixes}

    for f in png_files:
        try:
            name, ext = os.path.splitext(f)
            prefix, idx_str = name.rsplit("_", 1)
            if prefix in prefix_to_indexes:
                idx = int(idx_str)
                prefix_to_indexes[prefix].append(idx)
        except Exception:
            return False  # bad naming pattern

        # check file size > 0
        if os.path.getsize(os.path.join(folder, f)) == 0:
            return False

    # check every required prefix has files
    for prefix in required_prefixes:
        indexes = prefix_to_indexes[prefix]
        if not indexes:
            return False

        # check for missing indexes
        max_idx = max(indexes)
        expected = set(range(0, max_idx + 1))
        if set(indexes) != expected:
            return False

    # store maxes
    aligned_maxes = [
        max(prefix_to_indexes["aligned_spectrogram"]),
        max(prefix_to_indexes["aligned_chromagram"]),
        max(prefix_to_indexes["rotated_aligned_chromagram"]),
    ]
    if len(set(aligned_maxes)) != 1:
        return False

    unaligned_maxes = [
        max(prefix_to_indexes["unaligned_spectrogram"]),
        max(prefix_to_indexes["unaligned_chromagram"]),
        max(prefix_to_indexes["rotated_unaligned_chromagram"]),
    ]
    if len(set(unaligned_maxes)) != 1:
        return False

    return True



if __name__ == "__main__":
    
    start = time.perf_counter()
    last_lap = start
    
    
    # --- instantiate heavy ML models once ---
    key_proc = CNNKeyRecognitionProcessor()
    downbeat_rnn = RNNDownBeatProcessor()  # reuse this
    downbeat_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    
    data_dir = "../Songs/gtzan/jazz"
    files = find_audio_files(data_dir)
    files.sort() # in place alphabetical sort
    n = len(files)
    
    
    for i, filename in enumerate(files):
        
        song_title = filename.rsplit('/', 1)[-1][:-4]
        results_folder = "../Results/Dev/" + song_title + "/"
        if checkDataExists(results_folder): 
            print(f"skipping file ({i+1} of {n}): ", filename)
            continue
            
        print(f"processing file ({i+1} of {n}): ", filename)
        processFile(filename, results_folder)
        
        current = time.perf_counter()
        print(f"Elapsed time: {current - last_lap:.6f} seconds")
        last_lap = time.perf_counter()


    current = time.perf_counter()
    print(f"Total execution time for main: {current - start:.6f} seconds")