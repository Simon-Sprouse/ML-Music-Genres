import os
import numpy as np
from PIL import Image


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
    # if len(png_files) != len(files):
    #     return False
    # if not png_files:
    #     return False

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

