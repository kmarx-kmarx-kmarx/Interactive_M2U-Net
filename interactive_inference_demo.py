from interactive_m2unet import M2UnetInteractiveModel
import cv2
import numpy as np
import glob
import torch
import os
# In this demo, we load mages from .bmp files and run inference

def main():
    data_source = 'newdata'
    data_save = 'result_wdecay'
    model_pth = 'model_wdecay5/model_97_13.pth'
    os.makedirs(data_save, exist_ok=True)
    # Load images into np array
    img_paths = glob.glob(data_source + '/**_DPC.bmp', recursive=True)
    imgs = []
    img_paths = img_paths[:10]
    for img_path in img_paths:
        print(img_path)
        try:
            img = cv2.imread(img_path)
            img = img.transpose(2,0,1)
            # take only one channel - we are using the greyscale model
            img = img[0,:,:]
            img = np.expand_dims(img, 0)
            imgs.append(img)
        except:
            pass
    
    # predict_images must be in [n_images, n_channels, x, y]
    predict_images = np.array(imgs)
    print(predict_images.shape)

    # Load the model
    torch.cuda.empty_cache()
    model = M2UnetInteractiveModel(pretrained_model=model_pth)
    outs = model.predict_on_images(predict_images)
    print(outs.shape)
    outs = outs.transpose(0,2,3,1)
    outs = (225 * outs / np.max(outs)).astype(np.uint8)

    for i, result in enumerate(outs):
        cv2.imwrite(os.path.join(data_save, str(i) + ".png"), result)
if __name__ ==  "__main__":
    main()