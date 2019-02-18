import os

from PIL import Image
from torch.utils.data import Dataset


# Dataset for test set

class MyDataSet(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.images = os.listdir(root)
        self.images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.images[idx])

        with open(img_name, 'rb') as f:
            img = Image.open(f)
            image = img.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0
