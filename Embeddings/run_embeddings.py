import warnings
warnings.filterwarnings("ignore")

import time

import system_io
import load_models
import train_models




if __name__ == "__main__":
    start = time.perf_counter()
    
    
    data_dir = "../Results/Dev/"
    embeddings_save_dir = "../Results/EmbeddingData"
    model_save_dir = "../Results/Models"
    OVERRIDE_TRAINED_MODELS = True
    
    prefix = "aligned_spectrogram"
    filenames = system_io.getFilenames(data_dir, prefix)
    

    ## check override_trained_models
    ## if yes, train pca, AE, convAE, convVAE immediately
    ## if no, try to load model
    ## if model load fails -> train
    
    
    if OVERRIDE_TRAINED_MODELS: 
        
        train_models.fitPCA(filenames)
    


    
    