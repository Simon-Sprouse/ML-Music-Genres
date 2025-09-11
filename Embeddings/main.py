import warnings
warnings.filterwarnings("ignore")

import time
import os
from torch.utils.data import DataLoader

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
    
    
    train_files, test_files = system_io.getTrainTestSplit(filenames)
    img_size = 128
    batch_size = 32

    train_dataset = train_models.AE_Dataset(train_files, img_size=img_size)
    test_dataset = train_models.AE_Dataset(test_files, img_size=img_size)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    

    ## check override_trained_models
    ## if yes, train pca, AE, convAE, convVAE immediately
    ## if no, try to load model
    ## if model load fails -> train
    
    
    if OVERRIDE_TRAINED_MODELS: 
        
        ## train_models.fitPCA(filenames, model_save_dir)
        
        
        

        
        train_models.trainSimpleAE(train_dataloader, test_dataloader, model_save_dir)
        
        
        
        

        
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
        
        
        
        
        
    ae_path = os.path.join(model_save_dir, "autoencoder.pth")
    ae = load_models.loadSimpleAE(ae_path)
    if ae is None:
        print("No AE found, training new weights...")
        train_models.trainSimpleAE(train_dataloader, test_dataloader, model_save_dir)
        ae = load_models.loadSimpleAE(ae_path)
    else:
        print("Loaded AE weights from: ", ae_path)
    
        
        
    batch_filenames = filenames[:100] ## TODO implement per song batching


    pca_tensor = run_models.runPCA(batch_filenames, components, mean, n_components)
    print("pca tensor shape: ", pca_tensor.shape)
    
    ae_tensor = run_models.runAE(batch_filenames, ae)
    print("ae tensor shape: ", ae_tensor.shape)
    
    