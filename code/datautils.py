import torch 
import numpy as np
import cv2 
import matplotlib.pyplot as plt

YEARBOOK_BASE = '/home/adri/Desktop/cvc/data/yearbook/faces_aligned_small_mirrored_co_aligned_cropped_cleaned'

class Yearbook(torch.utils.data.Dataset):
    def __init__(self, files, img_size = (224, 224), base_path = f"{YEARBOOK_BASE}/") -> None:
        super(Yearbook).__init__()

        self.net_size = img_size
        self.files = [base_path + i.strip().split()[0] for i in open(files, 'r').readlines()]

    def read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
        if type(img) == type(None):
            raise FileNotFoundError(f"{path} not found")
        img = cv2.resize(img, self.net_size)
        return torch.from_numpy(img)

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        path = self.files[index]
        year = path.split('/')[-1].split('_')[0]
        return self.read_image(path), int(year)

if __name__ == '__main__':
    test = Yearbook(YEARBOOK_BASE + '/test_F.txt')
    plt.imshow(test[10][0])
    plt.show()