from interactive_m2unet import M2UnetInteractiveModel
import cv2
import numpy as np
import glob
import torch
import os
# In this demo, we load mages from .bmp files and run inference

def main():
    data_source = '/home/prakashlab/Documents/kmarx/train_m2unet_cellpose_cloud/data_sbc_validation'
    data_save = 'result_wdecay6'
    model_pth = 'model_wdecay6/model_22_10.pth'
    os.makedirs(data_save, exist_ok=True)
    # Load images into np array
    img_paths = glob.glob(data_source+'/**.npz', recursive=True)
    imgs = []
    masks = []
    img_paths = img_paths[:10]
    for img_path in img_paths:
        print(img_path)
        try:
            img = np.load(img_path)
            mask = img["mask"]
            img = img["img"]
            img = np.expand_dims(img, 0)
            imgs.append(img)
            masks.append(mask)
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

    for i, result in enumerate(outs):
        result = result[:,:,0]
        pred_mask = np.zeros(result.shape).astype(np.uint8)
        pred_mask[result > 0.5] = 255

        probs = (225 * result / np.max(result)).astype(np.uint8)

        diff = pred_mask - masks[i]
        color_diff = np.zeros((result.shape[0], result.shape[1], 3))
        color_diff[:,:,0] = (255 * (diff < 0)).astype(np.uint8)
        color_diff[:,:,2] = (255 * (diff > 0)).astype(np.uint8)
        # cv2.imwrite(os.path.join(data_save, str(i) + "_im.png"), imgs[i,0,:,:])
        cv2.imwrite(os.path.join(data_save, str(i) + "_probs.png"), probs)
        cv2.imwrite(os.path.join(data_save, str(i) + "_pred_mask.png"), pred_mask)
        cv2.imwrite(os.path.join(data_save, str(i) + "_diff.png"), color_diff)
        cv2.imwrite(os.path.join(data_save, str(i) + "_mask.png"), masks[i])
if __name__ ==  "__main__":
    main()