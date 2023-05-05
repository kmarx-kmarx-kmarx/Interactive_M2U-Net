from interactive_m2unet import M2UnetInteractiveModel
import cv2
import numpy as np
import gcsfs
import torch
import albumentations as A  
import os
from tqdm.contrib.itertools import product
import imageio
import json
from tqdm import trange, tqdm
import random

# In this demo, we load images and masks from google cloud and use them to train and validate a model
# We only have enough RAM for one image at a time - we will randomize paths and get images on the fly
def main():
    n_epochs = 10000
    sz = 1024
    center_blank = 400
    imag_dir = 'gs://octopi-tb-data/20221002'
    mask_dir = 'gs://octopi-tb-data-processing/20230301'
    exp_id   = ['SP1+_2022-10-02_20-10-50.529345', 'SP2+_2022-10-02_20-41-41.247131', 'SP3+_2022-10-02_21-13-35.929780', 'SP4+_2022-10-02_21-47-12.117750', 'SP5+_2022-10-02_22-16-53.962897']
    key = '/home/prakashlab/Documents/kmarx/tb_key.json'
    gcs_project = 'soe-octopi'
    flr_correct = '/home/prakashlab/Documents/kmarx/tb_segment/pipeline/illumination correction/flatfield_fluorescence.npy'
    flatfield_fluorescence = np.load(flr_correct)
    fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)
    erode_mask = 0
    proportion_train = 0.5
    n_im_per_cycle = 50
    random.seed(7)

    all_data_paths = get_im_paths(imag_dir, mask_dir, exp_id, fs, debug=False)
    random.shuffle(all_data_paths)

    # Load the model
    torch.cuda.empty_cache()
    transform = A.Compose(
        [
            A.Rotate(limit=(-180, 180), p=1), # set to -10, 10 for dpc
            A.RandomCrop(1500, 1500),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CenterCrop(sz, sz),  
        ]
    )
    model_config = {
        "input_channels": 1,
        "type": "m2unet",
        "activation": "sigmoid",
        "output_channels": 1,
        "loss": { "name": "BCELoss", "kwargs": {} },
        "optimizer": {
            "name": "RMSprop",
            "kwargs": { "lr": 0.01, "weight_decay": 1e-8, "momentum": 0.9 }
        },
        "augmentation":A.to_dict(transform),
    }
    model = M2UnetInteractiveModel(
        model_dir="tb_models_erosion_ccrop_+",          # Directory to save trained models
        model_name="TB_flr_only.pth", # Name of model to save
        model_config=model_config,# M2UNet config
        pretrained_model=None,    # Path to pretrained model
        save_freq=171,            # Number of epochs between model saves
        save_intermediate = True, # Set False to overwrite, set True to keep 
        save_if_lower = True,     # Set True to save if validation loss is lower than what was previously seen
        use_gpu=True,             # Use GPU for training and inferece
        run_width=sz,             # Image width for training/inference, limited by GPU RAM
        run_height=sz,            # Image height for training/inference, limited by GPU RAM
    )
    
    idx = 0
    for _ in trange(n_epochs):
        end_idx = idx + n_im_per_cycle
        end_idx = min(end_idx, len(all_data_paths)-1)
        data_paths = all_data_paths[idx:end_idx]
        n_train = int(np.floor(proportion_train * len(data_paths)))
        n_valid = len(data_paths) - n_train
        # split into training and validation
        training, validation = torch.utils.data.random_split(data_paths, [n_train, n_valid])
        for dirs in tqdm(training):
            # get the imgs and train one at a time
            training_dat, training_msk = get_ims([dirs], flatfield_fluorescence, erode_mask,center_blank, fs)
            model.train_on_stack(training_dat, training_msk)
        
        # validate in batch
        valid_dat, valid_msk=get_ims(validation, flatfield_fluorescence, erode_mask,center_blank, fs)
        model.validate(valid_dat, valid_msk)

        if end_idx >= (len(all_data_paths)-1):
            idx = 0
        else:
            idx = end_idx

def get_ims(dirs, flatfield_fluorescence, erode_mask, center_blank, fs):
    # Take a list of (im_path, mask_path) tuples as argument
    # Return a tuple of a list of images and masks in [n_im, n_chan, x, y] format
    # Only for monochrome images
    ims = []
    masks = []

    if erode_mask >= 0:
        shape = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(shape, (2 * erode_mask + 1, 2 * erode_mask + 1), (erode_mask, erode_mask))
        
    for img_path, mask_path in dirs:
        im_flr = np.array(imread_gcsfs(fs, img_path), dtype=float)
        im_flr = im_flr /  flatfield_fluorescence
        image = np.zeros(im_flr.shape)
        image[center_blank:-center_blank, center_blank:-center_blank] = im_flr[center_blank:-center_blank, center_blank:-center_blank]
        ims.append(image)
        im_mask = np.array(imread_gcsfs(fs, mask_path), dtype=float)
        im_mask = np.array(cv2.erode(im_mask, element))
        masks.append(im_mask)
    ims = np.expand_dims(ims, axis=1)
    masks = np.expand_dims(masks, axis=1)
    return ims, masks


def get_im_paths(imag_dir, mask_dir, exp_id, fs, debug=False):
    # Get paths: list of (image_path, mask_path) tuples
    paths = []
    print("Reading images")
    for id in tqdm(exp_id):
        # get the acquisition params
        json_file = fs.cat(os.path.join(imag_dir, id, "acquisition parameters.json"))
        acquisition_params = json.loads(json_file)
        if debug:
            acquisition_params['Ny'] = min(acquisition_params['Ny'], 2)
            acquisition_params['Nx'] = min(acquisition_params['Nx'], 2)
            acquisition_params['Nz'] = min(acquisition_params['Nz'], 2)
        xrng = range(acquisition_params['Ny']) # note - ny and nx should be flipped
        yrng = range(acquisition_params['Nx'])
        zrng = range(acquisition_params['Nz'])
        for x_id, y_id, z_id in product(xrng, yrng, zrng):
            x_id = str(x_id)
            y_id = str(y_id)
            z_id = str(z_id)
            img_path = os.path.join(imag_dir, id, '0', f"{x_id}_{y_id}_{z_id}_Fluorescence_488_nm_Ex.bmp")
            mask_path = os.path.join(mask_dir, id, f"{x_id}_{y_id}_{z_id}_mask.bmp")
            
            paths.append((img_path, mask_path))

    print("Done reading paths")
    return np.array(paths)



def imread_gcsfs(fs,file_path):
    '''
    imread_gcsfs gets the image bytes from the remote filesystem and convets it into an image
    
    Arguments:
        fs:         a GCSFS filesystem object
        file_path:  a string containing the GCSFS path to the image (e.g. 'gs://data/folder/image.bmp')
    Returns:
        I:          an image object
    
    This code has no side effects
    '''
    img_bytes = fs.cat(file_path)
    im_type = file_path.split('.')[-1]
    I = imageio.core.asarray(imageio.v2.imread(img_bytes, im_type))
    return I

if __name__ ==  "__main__":
    main()