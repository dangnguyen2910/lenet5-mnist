import torch 
import torch.nn as nn 
from sklearn.metrics import precision_score


def predict(model, image, device):
    image = image.unsqueeze(0).to(device)
    model.eval()
    output = model(image)
    softmax = nn.Softmax(dim=1)
    output = softmax(output)
    output = torch.argmax(output)
    return output


def evaluate(model, test_dataloader, device) -> dict:
    model.eval()
    precision = 0
    y_true = []
    y_pred = []
    for _, (img, label) in enumerate(test_dataloader):
        img = img.to(device)
        label = label.to(device)

        pred = model(img) 
        softmax = nn.Softmax(dim=1)
        output = torch.argmax(softmax(pred))
        y_true.append(label.cpu())
        y_pred.append(output.cpu())
    
    precision = precision_score(y_true, y_pred, average='micro')

    return precision

