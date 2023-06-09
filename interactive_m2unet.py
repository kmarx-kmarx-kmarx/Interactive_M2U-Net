import os
import json
import numpy as np
import albumentations as A
from m2unet import m2unet
import torch2trt
import torch
import torch.nn as nn
import torch.optim as optim

class M2UnetInteractiveModel:
    def __init__(
        self,
        model_dir=None,           # Directory to save trained models
        model_name=None,          # Name of model to save
        model_config=None,        # M2UNet config
        pretrained_model=None,    # Path to pretrained model
        save_freq=300,            # Number of epochs between model saves
        save_intermediate = True, # Set False to overwrite, set True to keep 
        save_if_lower = True,     # Set True to save if validation loss is lower than what was previously seen
        use_gpu=True,             # Use GPU for training and inferece
        use_trt=False,            # Set False for Torch, True for trt
        target_sz=16,             # Image height/width will be set to a multiple of target_sz
        run_width=1024,           # Image width for training/inference, limited by GPU RAM
        run_height=1024,          # Image height for training/inference, limited by GPU RAM
        batch_sz=1,               # Images per batch, limited by GPU RAM
        overlap=16,               # Number of pixels of overlap when tiling images
        **kwargs,
    ):
        # initialize directories
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model_name = model_name
        self._config = {
            "model_dir": model_dir,
            "model_name": model_name,
            "model_config": model_config,
        }
        assert (model_dir is not None) or (pretrained_model is not None), "need either model_dir or pretrained_model"
        # set image processing params
        assert (overlap <= run_width) and (overlap <= run_height), "need width, height larger than overlap"
        self.target_sz=target_sz
        self.batch_sz=batch_sz
        overlap += overlap % 2
        self.overlap=overlap
        run_width -= run_width % target_sz
        run_height -= run_height % target_sz
        self.run_width=run_width
        self.run_height=run_height
        self.use_trt=use_trt
        # initialize device
        gpu = torch.cuda.is_available()
        if use_gpu and gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # during training, how often should the model be saved?
        self.save_freq = int(save_freq)
        self.save_intermediate = save_intermediate
        self.save_if_lower = save_if_lower
        # Load a pretrained model
        if pretrained_model:
            assert os.path.exists(pretrained_model), "Pretrained model path not found."
            self.load(pretrained_model)
            if model_config and model_dir and model_name: # If we have a model config and we want to continue training, ensure config matches
                # make sure model_config has not changed
                for k in model_config:
                    if k in ["loss", "optimizer", "augmentation"]:
                        continue
                    if k in self.model_config:
                        assert (
                            self.model_config[k] == model_config[k]
                        ), "Model config has changed, please make sure you used the same model_config as before or set `resume=False`."
        # Or initialize a new model
        else:
            self.init_model(model_config)

        # Track training
        self._iterations = 0
        self._valid_loss = np.Inf

    def init_model(self, model_config):
        """initialize the model
        Parameters
        --------------
        None

        Returns
        ------------------
        None
        """
        assert model_config is not None

        model_kwargs = {}
        model_kwargs.update(model_config)
        augmentation_config = model_kwargs.get("augmentation")
        del model_kwargs["loss"]
        del model_kwargs["type"]
        del model_kwargs["optimizer"]
        del model_kwargs["augmentation"]
        if self.use_trt == False:
            self.model = m2unet(**model_kwargs).to(self.device)
        else:
            self.model = torch2trt.TRTModule()
        self.model_config = model_config
        self.transform = augmentation_config and A.from_dict(augmentation_config)
        loss_class = getattr(nn, self.model_config["loss"]["name"])
        optimizer_class = getattr(
            optim, self.model_config["optimizer"]["name"]
        )
        loss_instance = (
            loss_class(**self.model_config["loss"]["kwargs"])
            if "kwargs" in self.model_config["loss"]
            else loss_class
        )
        if self.use_trt:
            self.optimizer = None
        else:
            self.optimizer = optimizer_class(self.model.parameters(), **self.model_config["optimizer"]["kwargs"])
        self.criterion = loss_instance

    def crop_ims_to_size(self, images):
        '''ensures image width, height are multiples of target_sz
        Parameters
        --------------
        images: array [n_images, channel, width, height]
            a batch of input images
        
        Returns
        --------------
        images: array [n_images, channel, new_width, new_height]
            a batch of input images where new_width, new_height are multiples of target_sz
        '''
        __, __, width, height = images.shape
        new_width = width - (width % self.target_sz)
        new_height = height - (height % self.target_sz)

        return images[:,:,0:new_width,0:new_height]
    
    def augment(self, images, targets):
        """augment the images and labels
        Parameters
        --------------
        images: array [n_images, channel, width, height]
            a batch of input images

        targets: array [n_images, channel, width, height]
            a batch of labels

        Returns
        ------------------
        (images, targets) both are: array [n_images, width, height, channel]
            augmented images and labels
        """
        transformed_images = []
        transformed_targets = []
        # transpose from [n_images, channel, width, height] to [n_images, width, height, channel]
        images_transposed = images.transpose(0,2,3,1)
        targets_transposed = targets.transpose(0,2,3,1)
        for i in range(len(images)):
            if self.transform:
                transformed = self.transform(image=images_transposed[i], mask=targets_transposed[i])
            else:
                transformed = {"image": images_transposed[i], "mask": targets_transposed[i]}
            transformed_images.append(transformed["image"])
            transformed_targets.append(transformed["mask"])
        
        transformed_images = np.stack(transformed_images, axis=0)
        transformed_targets = np.stack(transformed_targets, axis=0)
        # transpose from [n_images, width, height, channel] to [n_images, channel, width, height]
        transformed_images = transformed_images.transpose(0,3,1,2)
        transformed_targets = transformed_targets.transpose(0,3,1,2)

        return transformed_images, transformed_targets

    def train(self, images, targets):
        """Train the model for a number of iterations
            note that width and height must be multiples of self.target_sz; 
            the images will be cropped to enforce this
        Parameters
        --------------
        images: array [batch_size, channel, width, height]
            a batch of input images

        targets: array [batch_size, channel, width, height]
            a batch of labels

        Returns
        ------------------
        train_loss: an array of loss during training
        """
        assert self.use_trt == False, "must use torch for training"
        imgi, tgs = self.augment(images, targets)
        imgi = self.crop_ims_to_size(imgi)
        tgs = self.crop_ims_to_size(tgs)

        imgi = torch.from_numpy(imgi)
        tgs = torch.from_numpy(tgs.astype(np.float32))
        imgi = imgi.to(device=self.device, dtype=torch.float32)
        tgs = tgs.to(device=self.device, dtype=torch.float32)
        masks_pred = self.model(imgi)
        train_loss = self.criterion(masks_pred, tgs)
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        if self._iterations % self.save_freq >= (self._iterations+len(images)) % self.save_freq or self._iterations == 0:
            if self.save_intermediate:
                # If save intermediate, give the new file a unique name
                name = os.path.join(self.model_dir, f"{self._iterations}_"+self.model_name)
                self.save(file_path=name)
            else:
                self.save()
        
        self._iterations += len(images)
        return train_loss.item()

    def get_loss(self, images, targets):
        """Get the loss between an image and the target
            note that width and height must be multiples of self.target_sz; 
            the images will be cropped to enforce this
        Parameters
        --------------
        images: array [batch_size, channel, width, height]
            a batch of input images

        targets: array [batch_size, channel, width, height]
            a batch of labels

        Returns
        ------------------
        valid_loss: array [batch_size]
            an array of loss during training
        masks_pred: array [batch_size, output_channels, width, height]
            predicted mask
        """
        imgi = self.crop_ims_to_size(images)
        tgs = self.crop_ims_to_size(targets)
        imgi = torch.from_numpy(imgi)
        tgs = torch.from_numpy(tgs.astype(np.float32))
        imgi = imgi.to(device=self.device, dtype=torch.float32)
        tgs = tgs.to(device=self.device, dtype=torch.float32)
        masks_pred = self.model(imgi)
        valid_loss = self.criterion(masks_pred, tgs)
        return valid_loss.item(), masks_pred.cpu().detach().numpy()
    
    def reshape_images_stack(self, images):
        ''' convert from [num_images, channel, width, height] to 
        [n_slices, channel, new_width, new_height]
        with overlap between slices. 
        This is necessary because our full images may be too large for the model to process.

        Parameters
        --------------
        images: array [n_images, channel, width, height]
            a batch of input images
        
        Returns
        --------------
        images: array [n_slices, channel, run_width, run_height]
            a batch of input images where run_width, run_height are multiples of target_sz and are small enough to run
        nx, ny, dx, dy, im_w, im_h: integers for reconstructiong the original images
        '''
        n_im, im_c, im_w, im_h = images.shape

        nx = int(np.ceil((im_w-self.overlap)/(self.run_width-self.overlap)))
        ny = int(np.ceil((im_h-self.overlap)/(self.run_height-self.overlap)))
        n_slices = int(n_im * nx * ny)
        # slice images down to size and put them in the stack
        image_stack = np.zeros((n_slices, im_c, self.run_width, self.run_height), dtype=np.uint8)
        dx, dy = (0, 0)

        for i in range(n_slices):
            x = i % nx
            y = int(np.floor(i/nx))
            z = int(np.floor(y/ny))
            y = y % ny

            # handle case x=(nx-1), y=(ny-1) separately 
            if x == (nx-1):
                x_0 = im_w-self.run_width
                x_1 = im_w
                dx = ((self.run_width-self.overlap)*x)-x_0
            else:
                x_0 = (self.run_width-self.overlap)*x 
                x_1 = x_0 + self.run_width
            
            if y == (ny-1):
                y_0 = im_h-self.run_height
                y_1 = im_h
                dy = ((self.run_height-self.overlap)*y)-y_0
            else:
                y_0 = (self.run_height-self.overlap)*y 
                y_1 = y_0 + self.run_height
            image_stack[i, :, :, :] = images[z, :, x_0:x_1, y_0:y_1]
        
        return image_stack, nx, ny, dx, dy, im_w, im_h, n_im

    def reshape_stack_images(self, image_stack, nx, ny, dx, dy, im_w, im_h, n_im):
        ''' convert from [num_images, channel, width, height] to 
        [n_slices, channel, new_width, new_height]
        with overlap between slices. 
        This is necessary because our full images may be too large for the model to process.

        Parameters
        --------------
        images: array [n_slices, channel, run_width, run_height]
            a batch of input images
        nx, ny, dx, dy, im_w, im_h: integers for reconstructing the original images

        Returns
        --------------
        images: array [n_images, channel, width, height]
            a stack of masks
        '''
        # print((image_stack.shape, nx, ny, dx, dy, im_w, im_h, n_im))
        output = np.zeros((n_im, image_stack.shape[1], im_w, im_h))
        d = int(self.overlap/2)

        for i in range(len(image_stack)):
            # print(i)
            x = i % nx
            y = int(np.floor(i/nx))
            z = int(np.floor(y/ny))
            y = y % ny
            
            # slice mask_stack: don't get overlaps unless we are at the end
            if x == 0:
                x_0m = 0
                if nx == 1:
                    x_1m = self.run_width
                else:
                    x_1m = self.run_width-d
            elif x==(nx-1):
                x_0m = dx-2*self.overlap
                x_1m = self.run_width
            else:
                x_0m = d
                x_1m = self.run_width-d

            # slice images
            if x == 0:
                x_0 = 0
            elif x == (nx-1):
                x_0 = im_w - dx + d
            else:
                x_0 = x*self.run_width - (2*x-1)*d
            if y == 0:
                y_0 = 0
            elif y == (nx-1):
                y_0 = im_h - dy + d
            else:
                y_0 = y*self.run_height - (2*y-1)*d

            if y == 0:
                y_0m = 0
                if ny == 1:
                    y_1m = self.run_height
                else:
                    y_1m = self.run_height-d
            elif y==(nx-1):
                y_0m = dy-2*self.overlap
                y_1m = self.run_height
            else:
                y_0m = d
                y_1m = self.run_height-d

            # finish slicing images
            if x == (nx-1):
                x_1 = im_w
                x_0 = x_1-x_1m+x_0m
            else:
                x_1 = x_0+x_1m-x_0m
            if y == (ny-1):
                y_1 = im_h
                y_0 = y_1-y_1m+y_0m
            else:
                y_1 = y_0+y_1m-y_0m
            
            # print(f"output: {output[z, :, x_0:x_1, y_0:y_1].shape}")
            # print(f"stack:  {image_stack[i, :, x_0m:x_1m, y_0m:y_1m].shape}")
            output[z, :, x_0:x_1, y_0:y_1] = image_stack[i, :, x_0m:x_1m, y_0m:y_1m]

        return output

    def train_on_stack(self, train_images, train_targets):
        """train on all the images in train_images
        Parameters
        --------------
        train_images: array [n_images, channel, width, height]
            a batch of input images for training

        train_targets: array [n_images, channel, width, height]
            a batch of labels for training

        Returns
        ------------------
        loss
            augmented images and labels
        """
        
        n_im, ch_im, w_im, h_im = train_images.shape
        n_tg, ch_tg, w_tg, h_tg = train_targets.shape
        assert n_im % self.batch_sz == 0, "number of images must be a multiple of batch size"
        assert (n_im == n_tg) and (w_im == w_tg) and (h_im == h_tg), "images and masks must be the same size"

        subset_ims = np.zeros((self.batch_sz, ch_im, w_im, h_im))
        subset_tgs = np.zeros((self.batch_sz, ch_tg, w_tg, h_tg))
        losses = []
        for i in range(n_im//self.batch_sz):
            # Break the image stack into batch_sz chuncks
            subset_ims = train_images[self.batch_sz*i:self.batch_sz*(i+1), :,:,:]
            subset_tgs = train_targets[self.batch_sz*i:self.batch_sz*(i+1), :,:,:]
            loss = self.train(subset_ims, subset_tgs)
            losses.append(loss)
        # reshape losses
        losses = np.array(losses).reshape(1,-1)

        return losses
    
    def predict(self, X, **kwargs):
        """predict the model for one input image
           note that width and height must be multiples of 16
        Parameters
        --------------
        X: array [batch_size, channel, width, height]
            the input image with n channels

        Returns
        ------------------
        array [batch_size, output_channel, width, height]
            the predicted label image
        """
        assert X.ndim == 4
        X = torch.from_numpy(X).to(device=self.device, dtype=torch.float32)
        outputs = self.model(X, **kwargs)
        return outputs.detach().cpu().numpy()

    def predict_on_slices(self, predict_images):
        """predict masks for all slices in predict_images
        Note that slice width, height must be multiples of target_sz
        Parameters
        --------------
        predict_images: array [n_slices, channel, width, height]
            a batch of input images for training

        Returns
        ------------------
        predict_images: array [n_slices, output_channel, width, height]
            predicted masks
        """
        n_im, ch_im, w_im, h_im = predict_images.shape
        assert (w_im % self.target_sz == 0) and (h_im % self.target_sz == 0), "width, height must be multiples of target_sz"
        
        predictions = []
        for i in range(int(np.ceil(n_im/self.batch_sz))):
            subset_ims = np.zeros((self.batch_sz, ch_im, w_im, h_im))
            # Break the image stack into batch_sz chuncks
            subset_ims = predict_images[self.batch_sz*i:self.batch_sz*(i+1), :,:,:]
            masks_pred = self.predict(subset_ims)
            predictions.append(masks_pred[:,0,:,:])
        # reshape predictions
        predictions = np.array(predictions)

        return predictions
    
    def predict_on_images(self, predict_images):
        initial_dims = predict_images.ndims
        if initial_dims == 3:
            np.expand_dims(predict_images, 0)
        assert predict_images.ndims == 4

        image_stack, *im_params = self.reshape_images_stack(predict_images)
        # Get predictions 
        preds = self.predict_on_slices(image_stack)
        # reshape
        image_preds = self.reshape_stack_images(preds, *im_params)

        if initial_dims == 3:
            image_preds = image_preds[0,:,:,:]

        return image_preds
    
    def validate(self, validate_images, validate_targets):
        ''' run predictions on a set of images and return the losses. 
            automatically save the model if the validation loss is less than previous
        validate_images: array [n_images, channel, width, height]
            a batch of input images for validation

        validate_targets: array [n_images, channel, width, height]
            a batch of labels for validation

        Returns
        ------------------
        loss: array [n_images]
            list of average loss for each image
        '''
        # Get predictions 
        preds = self.predict_on_images(validate_images)
        # get loss
        validate_targets = validate_targets / np.max(validate_targets)
        preds = preds / np.max(preds)
        validate_targets = torch.from_numpy(validate_targets).to(device=self.device, dtype=torch.float32)
        preds = torch.from_numpy(preds).to(device=self.device, dtype=torch.float32)
        loss = np.mean(self.criterion(preds, validate_targets).item())
        print(f"Loss: {loss}")
        
        if self.save_if_lower:
            if loss < self._valid_loss:
                self._valid_loss = loss
                fname = None
                if self.save_intermediate:
                    fname = os.path.join(self.model_dir, f"{self._iterations}_{int(100*loss)}_"+self.model_name)
                
                self.save(file_path=fname)
        return

    def save(self, file_path=None):
        """save the model
        Parameters
        --------------
        file_path: string
            the model file path

        Returns
        ------------------
        None
        """
        assert self.use_trt == False, "can only save if torch"
        if file_path is None:
            file_path = os.path.join(self.model_dir, self.model_name)
        assert isinstance(file_path, str)
        torch.save(self.model.state_dict(), file_path)
        with open(os.path.join(os.path.dirname(file_path), "config.json"), "w") as fil:
            fil.write(json.dumps(self._config))

    def load(self, file_path):
        """load the model
        Parameters
        --------------
        file_path: string
            the model file path

        Returns
        ------------------
        None
        """
        with open(os.path.join(os.path.dirname(file_path), "config.json"), "r") as fil:
            self._config = json.loads(fil.read())
        self.type = self._config["model_config"]["type"]
        self.model_config = self._config["model_config"]
        self.init_model(self.model_config)

        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        return

    def make_trt(self):
        '''
        Convert the given Torch model into TRT
        '''
        assert self.use_trt == False, "can't convert trt to trt"

        data = torch.randn(self.batch_sz, n_channels, self.run_width, self.run_height).to(self.device)
        model_trt = torch2trt.torch2trt(self.model)
        torch.seave(model_trt.state_dict(), savepath)
    
    def parameters(self):
        return self.model.parameters()
    def named_parameters(self):
        return self.model.named_parameters()
    
    def finalize(self):
        self.model = None
