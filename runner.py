import torch

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)


def train_step(model, optimizer, loss_fn, img, label, wbce):
    optimizer.zero_grad(set_to_none=True)
    logits = model(img)
    logits, label = logits.squeeze(1), label.squeeze(1).float()
    loss, _ = loss_fn(logits, label, wbce=wbce)
    loss.backward()
    optimizer.step()
    return loss.item()


def validate_epoch(model, val_loader, loss_fn, wbce, device):
    model.eval()
    epoch_val_loss = 0.
    with torch.no_grad():
        for (img, label) in val_loader:
            img, label = img.to(device), label.to(device)
            logits = model(img)
            logits, label = logits.squeeze(1), label.squeeze(1).float()
            loss, _ = loss_fn(logits, label, wbce=wbce)
            epoch_val_loss += loss.item()
    return epoch_val_loss/len(val_loader)