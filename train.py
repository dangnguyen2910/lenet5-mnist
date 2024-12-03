import torch
import torch.nn as nn
from mnist_dataset import Mnist
from get_mnist import load
import os 



def train_one_epoch(train_dataloader, val_dataloader, 
          model, loss_fn, optimizer, device):
    running_loss = 0

    for i, (img, label) in enumerate(train_dataloader):
        img = img.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss/len(train_dataloader)

    


    

def train(train_dataloader, val_dataloader, 
          model, loss_fn, optimizer, epochs, device):
    
    if (not os.path.exists('models/')):
        os.makedirs('models')
    
    best_vloss = 100
    for epoch in range(epochs):
        print('-' * 60)
        print(f"Epoch: [{epoch+1}/{epochs}]")
        
        train_loss = train_one_epoch(train_dataloader, val_dataloader, model, loss_fn, optimizer, device)

        running_vloss = 0
        for i, (img, label) in enumerate(val_dataloader):
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            running_vloss += loss.item()

        if (running_vloss/len(val_dataloader) < best_vloss):
            torch.save(model.state_dict(), "models/lenet.pth")

        print(f"   Train loss: {train_loss:.2f}")
        print(f"   Val loss: {running_vloss/len(val_dataloader):.2f}")

    return





if __name__ == "__main__":
    (train_images, train_labels), (_, _) = load()
    train = Mnist(train_images, train_labels)