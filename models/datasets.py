import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import pandas as pd


class BirdsDataset(Dataset):
    def __init__(self, split, transform):
        self.split = split
        assert(split in ["valid", "test", "train"])
        while os.path.split(os.getcwd())[1] != "birds":
            os.chdir("..")
 
        data_dir = os.path.join(os.getcwd(), "data", self.split)
        self.annotation_file = os.path.join(data_dir, "index.csv")
        
        self.img_labels = pd.read_csv(self.annotation_file)
        self.transform = transform  
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = self.img_labels.iloc[index,0]
        label = self.img_labels.iloc[index,1]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        bird = os.path.dirname(img_path)
        return image, label


transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Ensure the image size is consistent
    transforms.RandomCrop((224, 224)),  # Ensure the image size is consistent
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),  # Converts images to PyTorch tensors
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
train_data = BirdsDataset(
    split = "train",
    transform = transform,
)

test_data = BirdsDataset(
    split = "test",
    transform = transform,
)

val_data = BirdsDataset(
    split = "valid",
    transform = transform,
)

train_dataloader = DataLoader(train_data, batch_size = 128, shuffle = True, num_workers = 20, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size = 128, shuffle = True, num_workers = 20, pin_memory=True)
val_dataloader = DataLoader(val_data, batch_size = 128, shuffle = True, num_workers = 20, pin_memory=True)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    images, labels = next(iter(train_dataloader))
    
    for i in range(len(images)):
        img = images[i].squeeze().permute(1,2,0)
        plt.imshow(img)
        plt.show()
        print(f"label: {labels[i]}")
