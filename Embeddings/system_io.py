import os
import glob

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