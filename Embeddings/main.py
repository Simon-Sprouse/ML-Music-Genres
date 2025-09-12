import warnings
warnings.filterwarnings("ignore")

import time
import os
import torch
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
        
        train_models.fitPCA(filenames, model_save_dir)
        
        train_models.trainSimpleAE(train_dataloader, test_dataloader, model_save_dir)
        train_models.trainConvAE(train_dataloader, test_dataloader, model_save_dir)
        train_models.trainConvVAE(train_dataloader, test_dataloader, model_save_dir)
        
        


        
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
        
        
    conv_ae_path = os.path.join(model_save_dir, "conv_autoencoder.pth")
    conv_ae = load_models.loadConvAE(conv_ae_path)
    if conv_ae is None:
        print("No Conv AE found, training new weights...")
        train_models.trainConvAE(train_dataloader, test_dataloader, model_save_dir)
        conv_ae = load_models.loadConvAE(conv_ae_path)
    else:
        print("Loaded Conv AE weights from: ", conv_ae_path)
    
        
        
    conv_vae_path = os.path.join(model_save_dir, "conv_variational_autoencoder.pth")
    conv_vae = load_models.loadConvVAE(conv_vae_path)
    if conv_vae is None:
        print("No Conv VAE found, training new weights...")
        train_models.trainConvVAE(train_dataloader, test_dataloader, model_save_dir)
        conv_vae = load_models.loadConvVAE(conv_vae_path)
    else:
        print("Loaded Conv VAE weights from: ", conv_vae_path)
        
        
        
        
        
        
    ### load cnn models
    cnn_models = load_models.load_Cnn_Models()
    print("Loaded CNN models")
        
        
        
    
    
    




    
    batches = system_io.group_by_song(filenames)
    print(f"Generated {len(batches)} batches")
        

    os.makedirs(embeddings_save_dir, exist_ok=True)

    for i, batch_filenames in enumerate(batches):
        
        if len(batch_filenames) < 1:
            continue
        
        
        song_name = os.path.basename(os.path.dirname(batch_filenames[0]))
        
        # Run models
        pca_tensor = run_models.runPCA(batch_filenames, components, mean, n_components)
        ae_tensor = run_models.runAE(batch_filenames, ae)
        conv_ae_tensor = run_models.runAE(batch_filenames, conv_ae)
        conv_vae_tensor = run_models.runVAE(batch_filenames, conv_vae)
        cnn_embeddings_dict = run_models.runCNN(batch_filenames, cnn_models)

        # Collect everything into one dictionary
        batch_outputs = {
            "pca": pca_tensor,
            "ae": ae_tensor,
            "conv_ae": conv_ae_tensor,
            "conv_vae": conv_vae_tensor,
        }
        for model_name, tensor in cnn_embeddings_dict.items():
            batch_outputs[model_name] = tensor

        # Save one .pt file per batch
        save_path = os.path.join(embeddings_save_dir, f"{song_name}.pt")
        torch.save(batch_outputs, save_path)
        print(f"Saved batch {i} to {save_path}")