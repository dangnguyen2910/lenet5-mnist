import torch
import torch.nn as nn
import model
from mnist_dataset import Mnist
from get_mnist import load




def train_one_epoch(train_dataloader, val_dataloader, 
          model, loss_fn, optimizer, device):
    running_loss = 0

    for i, (img, label) in enumerate(train_dataloader):
        img = img.to(device)
        
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss/len(train_dataloader)

    


    

def train(train_dataloader, val_dataloader, 
          model, loss_fn, optimizer, epochs, device):

    for epoch in range(epochs):
        print('-' * 60)
        print(f"Epoch: [{epoch}/{epochs}]")
        
        train_loss = train_one_epoch(train_dataloader, val_dataloader, model, loss_fn, optimizer, device)

        running_vloss = 0
        for i, (img, label) in enumerate(val_dataloader):
            img = img.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            running_vloss += loss.item()

        print(f"   Train loss: {train_loss:.2f}")
        print(f"   Val loss: {running_vloss/len(val_dataloader):.2f}")

    return model





if __name__ == "__main__":
    (train_images, train_labels), (_, _) = load()
    train = Mnist(train_images, train_labels)