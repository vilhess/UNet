import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class CRACK500(Dataset):
    """CRACK500 dataset."""

    def __init__(self, txt_file, root_dir, train=False, data_augmentation=True):
        """
        Args:
            txt_file (string): Path to the txt file list of image paths (eg train.txt, test.txt, val.txt).
            root_dir (string): Parent directory for the directory with all the images.
            train (boolean): True for training dataset (crop images to have size 256x256, otherwise get full image)
            data_augmentation (boolean): True for using data_augmentation during training
        """
        self.imgslist = pd.read_csv(txt_file, sep=' ',header=None)
        self.root_dir = root_dir
        self.train = train
        self.data_augmentation = data_augmentation
    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, idx):
        # get image and mask paths
        img_name = os.path.join(self.root_dir, self.imgslist.iloc[idx, 0])
        mask_name = os.path.join(self.root_dir, self.imgslist.iloc[idx, 1])

        # open images using PIL and apply transform list:
        image = Image.open(img_name)
        image = transforms.ToTensor()(image)
        image = transforms.Grayscale()(image)
        if image.shape[1]>image.shape[2]:
          image = image.transpose(1,2)
        image = transforms.Normalize(0.5, 0.5)(image) # shape 1 x w x h

        # open mask binary image using PIL:
        mask = Image.open(mask_name)
        mask = transforms.ToTensor()(mask)
        if mask.shape[1]>mask.shape[2]:
          mask = mask.transpose(1,2)
        mask = mask.long().squeeze() # shape w x h

        if self.train: #get a subimage of size 256x256:
          if self.data_augmentation:

            # crop center 256x256:
            h,w = mask.shape
            c1 = (h-256)//2 # (c1,c2) is top-left corner
            c2 = (w-256)//2
            image = image[:,c1:(c1+256),c2:(c2+256)]
            mask = mask[c1:(c1+256),c2:(c2+256)]

            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5)
            ])

            both = torch.cat((image, mask.unsqueeze(0)))
            both_tr = transform(both)

            image = both_tr[0].unsqueeze(0)
            mask = both_tr[1].to(int)

          else:
            # crop center 256x256:
            h,w = mask.shape
            c1 = (h-256)//2 # (c1,c2) is top-left corner
            c2 = (w-256)//2
            image = image[:,c1:(c1+256),c2:(c2+256)]
            mask = mask[c1:(c1+256),c2:(c2+256)]
        else:
          # full image cropped to be multiple of 32:
          h,w = mask.shape
          nh = (h//32)*32
          nw = (w//32)*32
          image = image[:,:nh,:nw]
          mask = mask[:nh,:nw]

        sample = [image, mask]
        return sample
