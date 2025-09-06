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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os

# Global variables for each worker process
key_proc = None
downbeat_rnn = None
downbeat_tracker = None

def init_worker():
    """Initialize ML models in each worker process"""
    global key_proc, downbeat_rnn, downbeat_tracker
    print(f"Initializing models in process {os.getpid()}")
    key_proc = CNNKeyRecognitionProcessor()
    downbeat_rnn = RNNDownBeatProcessor()
    downbeat_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)

def processFile(args):
    """
    Process a single audio file using globally instantiated ML models.
    Args is a tuple of (filename, output_folder, file_index, total_files)
    """
    filename, output_folder, file_index, total_files = args
    
    # Use global models (initialized once per worker process)
    global key_proc, downbeat_rnn, downbeat_tracker
    
    process_start = time.perf_counter()
    
    try:
        # Check if models are properly initialized
        if key_proc is None or downbeat_rnn is None or downbeat_tracker is None:
            raise RuntimeError("ML models not properly initialized in worker process")
        
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
        
        process_time = time.perf_counter() - process_start
        return {
            'filename': filename,
            'file_index': file_index,
            'total_files': total_files,
            'process_time': process_time,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        process_time = time.perf_counter() - process_start
        return {
            'filename': filename,
            'file_index': file_index,
            'total_files': total_files,
            'process_time': process_time,
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    start = time.perf_counter()
    
    # Get list of files to process
    data_dir = "../Songs/gtzan/jazz"
    files = system_io.find_audio_files(data_dir)
    files.sort()  # in place alphabetical sort
    n = len(files)
    
    # Filter files that need processing (resumable logic)
    files_to_process = []
    for i, filename in enumerate(files):
        song_title = filename.rsplit('/', 1)[-1][:-4]
        results_folder = "../Results/Dev/" + song_title + "/"
        
        if system_io.checkDataExists(results_folder):
            ## print(f"skipping file ({i+1} of {n}): ", filename)
            continue
        else:
            files_to_process.append((filename, results_folder, i+1, n))
    
    if not files_to_process:
        print("No files to process - all results already exist!")
        exit(0)
    
    print(f"Found {len(files_to_process)} files to process out of {n} total files")
    
    # Determine number of worker processes
    # Use CPU count minus 1 to leave one core free, or at least 1 process
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} worker processes")
    
    # Process files concurrently
    completed_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        # Submit all jobs
        future_to_args = {executor.submit(processFile, args): args for args in files_to_process}
        
        # Process completed jobs as they finish
        for future in as_completed(future_to_args):
            result = future.result()
            completed_count += 1
            
            if result['success']:
                print(f"✓ Completed file ({result['file_index']} of {result['total_files']}): {result['filename']}")
                print(f"  Process time: {result['process_time']:.6f} seconds")
            else:
                failed_count += 1
                print(f"✗ Failed file ({result['file_index']} of {result['total_files']}): {result['filename']}")
                print(f"  Error: {result['error']}")
                print(f"  Process time: {result['process_time']:.6f} seconds")
            
            print(f"  Progress: {completed_count}/{len(files_to_process)} files completed")
            print()
    
    # Final summary
    total_time = time.perf_counter() - start
    
    # Calculate completion percentage
    successfully_processed = completed_count - failed_count
    total_completed = n - len(files_to_process) + successfully_processed  # previously done + newly processed
    completion_percentage = (total_completed / n) * 100
    
    print(f"=== PROCESSING COMPLETE ===")
    print(f"Total files processed: {completed_count}")
    print(f"Failed files: {failed_count}")
    print(f"Successfully processed: {successfully_processed}")
    print(f"Total execution time: {total_time:.6f} seconds")
    if len(files_to_process) > 0:
        print(f"Average time per file: {total_time / len(files_to_process):.6f} seconds")
    print(f"Used {num_workers} worker processes")
    print(f"Overall progress: {total_completed}/{n} files completed ({completion_percentage:.1f}%)")