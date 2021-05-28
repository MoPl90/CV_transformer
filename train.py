from numpy.core.fromnumeric import mean
import torch
from tqdm import tqdm
import os
import numpy as np
from metrics import dice


def train(model, train_loader, val_loader, device, criterion, optimizer, scheduler, epochs, classes, weights, patience=10, name=''):

    val = []
    for epoch in range(0, epochs):
        epoch_loss     = 0
        epoch_accuracy = 0
        epoch_dice     = 0

        for data, label in tqdm(train_loader):
            data = torch.as_tensor(data, dtype=torch.float).to(device)
            label = torch.as_tensor(label, dtype=torch.long).to(device)

            output = model(data)
            if len(label.shape) > 2:
                criterion.weight = torch.tensor(weights).float().to(device)
                DSC = dice(output, label, classes=classes)
                loss = criterion(output, label.squeeze(1)) + 1 - torch.sum(criterion.weight * DSC) / len(DSC) 
            else:
                loss = criterion(output, label)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.item() / len(train_loader)
            epoch_loss += loss.item() / len(train_loader)

            if len(label.shape) > 2:
                epoch_dice += dice(output.detach(), label.detach(), classes=classes) / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss     = 0
            epoch_val_dice     = 0
            for data, label in val_loader:
                data = torch.as_tensor(data, dtype=torch.float).to(device)
                label = torch.as_tensor(label, dtype=torch.long).to(device)

                val_output = model(data)
                if len(label.shape) > 2:
                    val_loss = criterion(val_output, label.squeeze(1))
                else:
                    val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

                if len(label.shape) > 2:
                    epoch_val_dice += dice(val_output, label, train=False, classes=classes) / len(val_loader)
            
            val_dice_mean = torch.mean(criterion.weight * epoch_val_dice)
            

            #Check checkpoints and early stopping criterion:
            if len(val) > 0:
                if val_dice_mean > val[-1]:
                    torch.save(model.state_dict(), 'data/' + name + '_' + str(epoch + 1) + '.pt')
                    try:
                        os.system("rm data/" + name + '_' + str(epoch) + '.pt')
                    except FileExistsError:
                        pass

                if len(val) > patience and torch.all(val_dice_mean <= val[-patience:]):
                    break

        val.append(val_dice_mean)
        print(val)

        scheduler.step()


        if len(label.shape) > 2:
            print(f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}  - DSC: " + ", ".join([f"{d:.4f}" for d in epoch_dice]) + f" - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} - val_DSC: " + ", ".join([f"{d:.4f}" for d in epoch_val_dice]) + "\n")
        else:
            print(f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")
