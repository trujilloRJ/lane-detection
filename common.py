import re

def extract_log_data(log_file_path):
    epochs = []
    train_losses = []
    validation_losses = []
    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(r"Global epoch: (\d+) -> Train loss: ([\d.]+) \| Validation loss: ([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                validation_losses.append(float(match.group(3)))
    return epochs, train_losses, validation_losses