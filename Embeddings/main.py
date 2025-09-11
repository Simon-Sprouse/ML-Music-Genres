import warnings
warnings.filterwarnings("ignore")

import time

import system_io
import load_models
import run_models
import train_models




if __name__ == "__main__":
    start = time.perf_counter()
    
    
    data_dir = "../Results/Dev/"
    embeddings_save_dir = "../Results/EmbeddingData"
    model_save_dir = "../Results/Models"
    OVERRIDE_TRAINED_MODELS = False
    
    prefix = "aligned_spectrogram"
    filenames = system_io.getFilenames(data_dir, prefix)
    

    ## check override_trained_models
    ## if yes, train pca, AE, convAE, convVAE immediately
    ## if no, try to load model
    ## if model load fails -> train
    
    
    if OVERRIDE_TRAINED_MODELS: 
        
        train_models.fitPCA(filenames, model_save_dir)
        
        
    pca_path=f"{model_save_dir}/pixel_pca.npz"
    pca = load_models.loadPCA(pca_path)
    if pca is None:
        print("No PCA found, training new weights...")
        train_models.fitPCA(filenames, model_save_dir)
        pca = load_models.loadPCA(pca_path)
        components, mean, n_components = pca
    else:
        # no training needed
        components, mean, n_components = pca
        print("Loaded PCA weights from: ", pca_path)
        
        
    batch_filenames = filenames[:100] ## TODO implement per song batching


    pca_tensor = run_models.runPCA(batch_filenames, components, mean, n_components)
    print(pca_tensor.shape)
    
    