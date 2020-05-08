import torch
import os
import csv
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


#pytorch Dataset class to load provided data
class FashionDataset(Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        """
        Arguments:
            - data_dir : path to folder containing csv files and "images" folder
            - csv_path : path to csv file (i.e test_set.csv)
            - transform : (optional)
        """
        self.data = pd.read_csv(os.path.join(data_dir, csv_path))
        # generate images names before hand as some images are not there in the folder
        self.generate_image_list(data_dir)
        self.transform = transform
        self.data_dir = data_dir

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_dir, "images", str(self.data.iloc[index]['id'])+".jpg"))
        #uncomment following line to read image in grayscale
        # self.image = Image.open(self.image_names[index]).convert('L')
        
        if self.transform:
            image = self.transform(image)
        return (image, self.label_map[self.data.iloc[index]['articleType']])

    def __len__(self):
        return len(self.data)

    def generate_image_list(self,data_dir):
        self.label_map = {}
        i = 0
        for indx, img_path in self.data.iterrows():
            path = os.path.join(DATA_DIR, "images", str(img_path['id'])+".jpg")
            if not os.path.exists(path):
                self.data.drop(indx, inplace=True)
            if img_path['articleType'] not in self.label_map:
                self.label_map[img_path['articleType']] = i
                i += 1
        print("Found {} images in {} for provided csv file.".format(len(self.data), data_dir))




# for testing purpose
if __name__ == "__main__":
    data_loader = FashionDataset(DATA_DIR, "remainingclasses_set.csv", transform=transforms.Compose([transforms.Resize((280,280)), transforms.ToTensor()]))
    i = 0
    for img, label in data_loader:
        print(label)
        d = transforms.ToPILImage()(img)
        i += 1
        if i == 10:
            d.show()
            break