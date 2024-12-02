import torch
from torch.utils.data import Dataset
from get_mnist import load

class Mnist: 
    def __init__(self, image_list, label_list): 
        self.image_list = image_list
        self.label_list = label_list

    def __getitem__(self, idx): 
        img = self.image_list[idx]
        label = self.label_list[idx]

        img = torch.Tensor(img).unsqueeze(0)
        return img, label

    def __len__(self):
        return len(self.image_list)
    

if __name__ == "__main__":
    (train_imgs, train_labels), (_,_) = load()
    train = Mnist(train_imgs, train_labels)