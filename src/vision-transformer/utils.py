import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor


class ImageNetDataset(Dataset):
    
    def __init__(self, image_entry, label_entry, resize_shape = (224,224), patch_size = (16,16), transform = None):
        self.image_entry = image_entry
        self.label_entry = label_entry
        self.resize_shape = resize_shape

        self.x_patch = patch_size[0]
        self.y_patch = patch_size[1]
        self.transform = transform

    def __len__(self):
        return len(self.image_entry)

    def __getitem__(self, idx):

        entry = self.image_entry[idx]
        label = torch.as_tensor(self.label_entry[idx])

        img = Image.open(entry).convert("RGB").resize(self.resize_shape)
        img = np.array(img)

        ## Normalization idea, taken from here -> https://rwightman.github.io/pytorch-image-models/models/vision-transformer/ , mean = [0.5, 0.5, 0.5] and std = [0.5, 0.5, 0.5]
        
        if self.transform is not None:
          img = self.transform(img)
        
        ## Patches taken from here: https://discuss.pytorch.org/t/creating-3d-dataset-dataloader-with-patches/50861/2
        
        if  self.transform is None:
          img = ToTensor()(img)

        img = img.unfold(2,self.x_patch,self.y_patch).unfold(1,self.x_patch,self.y_patch).contiguous().view(-1, 3*self.x_patch*self.y_patch)
        return img, label