import torch
from tqdm import tqdm


def train(model, train_loader, val_loader, device, criterion, optimizer, scheduler, epochs):

    for epoch in range(0, epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = torch.as_tensor(data.to(device), dtype=torch.float)
            label = torch.as_tensor(label.to(device), dtype=torch.int64)

            output = model(data)
            if len(label.shape) > 2:
                loss = criterion(output, torch.argmax(label, dim=1).long())
            else:
                loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in val_loader:
                data = torch.as_tensor(data.to(device), dtype=torch.float)
                label = torch.as_tensor(label.to(device), dtype=torch.int64)

                val_output = model(data)
                if len(label.shape) > 2:
                    val_loss = criterion(val_output, torch.argmax(label, dim=1).long())
                else:
                    val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

        scheduler.step()

        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )