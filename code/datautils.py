import torch 
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import random

YEARBOOK_BASE = '/home/adri/Desktop/cvc/data/yearbook/faces_aligned_small_mirrored_co_aligned_cropped_cleaned'

class Yearbook(torch.utils.data.Dataset):
    def __init__(self, files, img_size = (224, 224), base_path = f"{YEARBOOK_BASE}/") -> None:
        super(Yearbook).__init__()

        self.net_size = img_size
        self.files = [base_path + i.strip().split()[0] for i in open(files, 'r').readlines()]
        self.years_lookup = {}
        for path in self.files:
            year = int(path.split('/')[-1].split('_')[0])
            if not year in self.years_lookup: self.years_lookup[year] = []
            self.years_lookup[year].append(path)

        self.n_xo = 2 # Number of example images for each target image


    def read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
        if type(img) == type(None):
            raise FileNotFoundError(f"{path} not found")
        img = cv2.resize(img, self.net_size)

        return torch.from_numpy(img)

    def pick_base_images(self, year):

        '''

        Picks a bunch of images so target image gets the style of them.


        '''
        targets = len(self.years_lookup)
        choice = random.randint(0, targets - 1)
        target_year = list(self.years_lookup.keys())[choice] 
        if target_year == year: target_year = list(self.years_lookup.keys())[choice - 1] 
        return random.sample(self.years_lookup[target_year], self.n_xo)

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        path = self.files[index]
        year = path.split('/')[-1].split('_')[0]
        xos = torch.stack([self.read_image(x) for x in self.pick_base_images(year)])
        return self.read_image(path), int(year), xos

if __name__ == '__main__':
    test = Yearbook(YEARBOOK_BASE + '/test_F.txt')
    plt.imshow(test[10][0])
    plt.show()