import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob


class MyDataset(Dataset):
    def __init__(self, img_dir, mask_dir, train=True, transform=None):
        super().__init__()
        self.transform = transform
        self.images = sorted(glob.glob(f'{img_dir}/*.jpg'))
        self.masks = sorted(glob.glob(f'{mask_dir}/*.gif'))

        split = int(0.8 * len(self.masks))
        if train:
            self.images = self.images[:split]
            self.masks = self.masks[:split]
        else:
            self.images = self.images[split:]
            self.masks = self.masks[split:]


    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        img_path, mask_path = self.images[index], self.masks[index]
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask = np.where(mask == 255, 1, 0)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        
        return img, mask
