import os
import glob
import random
from collections import defaultdict

def getFilenames(base_dir, prefix):
    """
    Search recursively under base_dir for files of the form 
    aligned_spectrogram_<n>.png. 
    Returns a list of all matches sorted first by folder, 
    then numerically by <n>.
    """
    pattern = os.path.join(base_dir, "*", f"{prefix}_*.png")
    files = glob.glob(pattern)

    def sort_key(path):
        folder = os.path.dirname(path)
        idx = int(os.path.splitext(path)[0].split('_')[-1])
        return (folder, idx)

    files.sort(key=sort_key)
    return files


def getTrainTestSplit(filenames, test_split=0.2, seed=42):
    song_to_files = defaultdict(list)
    for f in filenames:
        song = os.path.basename(os.path.dirname(f))  # parent folder name
        song_to_files[song].append(f)

    all_songs = sorted(song_to_files.keys())
    print(f"Found {len(all_songs)} songs")

    # Reproducible shuffle
    random.seed(seed)
    random.shuffle(all_songs)

    # Split songs into train/test
    num_test = int(len(all_songs) * test_split)
    test_songs = set(all_songs[:num_test])
    train_songs = set(all_songs[num_test:])

    # Flatten into filename lists
    train_files = [f for song in train_songs for f in song_to_files[song]]
    test_files  = [f for song in test_songs  for f in song_to_files[song]]

    print(f"Train: {len(train_files)} files from {len(train_songs)} songs")
    print(f"Test:  {len(test_files)} files from {len(test_songs)} songs")
    
    return train_files, test_files


