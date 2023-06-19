import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class ADE20KDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.file_list = self._load_file_list()

    def _load_file_list(self):
        file_path = os.path.join(self.root, f'{self.split}.txt')
        with open(file_path, 'r') as file:
            file_list = [line.strip() for line in file]
        return file_list

    def __getitem__(self, index):
        filename = self.file_list[index]
        image_path = os.path.join(self.root, 'imgs', 'ADE_val_' + filename + '.jpg')
        mask_path = os.path.join(self.root, 'masks', 'ADE_val_' + filename + '_seg.png')

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale


        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def __len__(self):
        return len(self.file_list)
