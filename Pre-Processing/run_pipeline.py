import warnings
warnings.filterwarnings("ignore")

import time
import librosa
import numpy as np
import madmom
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from PIL import Image


import transform
import system_io



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


    ## Transform wave -> images
    fixed_spectros = transform.make_fixed_size_spectrogram(y, sr, n_seconds=fixed_window_n_seconds, width=128, n_mels=128)
    downbeat_spectros = transform.make_downbeat_aligned_spectrogram(y, sr, downbeat_times, time_bins_per_downbeat=128, downbeats_in_image=1)
    fixed_chromas = transform.make_fixed_size_chromagrams(y, sr, n_seconds=fixed_window_n_seconds, width=128, hop_length=512)
    downbeat_chromas = transform.make_downbeat_aligned_chromagrams(y, sr, downbeat_times, time_bins_per_downbeat=128, downbeats_in_image=1)
    rotated_fixed_chromas = [transform.rotate_chroma_to_C_major(chroma_image, key) for chroma_image in fixed_chromas]
    rotated_downbeat_chromas = [transform.rotate_chroma_to_C_major(chroma_image, key) for chroma_image in downbeat_chromas]


    ## Save results
    system_io.downloadImagesArray(fixed_spectros, "unaligned_spectrogram", output_folder)
    system_io.downloadImagesArray(downbeat_spectros, "aligned_spectrogram", output_folder)
    system_io.downloadImagesArray(fixed_chromas, "unaligned_chromagram", output_folder)
    system_io.downloadImagesArray(downbeat_chromas, "aligned_chromagram", output_folder)
    system_io.downloadImagesArray(rotated_fixed_chromas, "rotated_unaligned_chromagram", output_folder)
    system_io.downloadImagesArray(rotated_downbeat_chromas, "rotated_aligned_chromagram", output_folder)
    
    



if __name__ == "__main__":
    
    start = time.perf_counter()
    last_lap = start
    
    
    # --- instantiate heavy ML models once ---
    key_proc = CNNKeyRecognitionProcessor()
    downbeat_rnn = RNNDownBeatProcessor()  # reuse this
    downbeat_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    
    data_dir = "../Songs/gtzan/jazz"
    files = system_io.find_audio_files(data_dir)
    files.sort() # in place alphabetical sort
    n = len(files)
    
    
    for i, filename in enumerate(files):
        
        
        song_title = filename.rsplit('/', 1)[-1][:-4]
        results_folder = "../Results/Dev/" + song_title + "/"
        if system_io.checkDataExists(results_folder): 
            print(f"skipping file ({i+1} of {n}): ", filename)
            continue
            
        print(f"processing file ({i+1} of {n}): ", filename)
        processFile(filename, results_folder)
        
        current = time.perf_counter()
        print(f"Elapsed time: {current - last_lap:.6f} seconds")
        last_lap = time.perf_counter()


    current = time.perf_counter()
    print(f"Total execution time for main: {current - start:.6f} seconds")