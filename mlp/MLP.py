import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]
import numpy as np
import torch
import tqdm

from datasets import load_dataset
from datetime import datetime

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'MLP'
        self.fc1 = torch.nn.Linear(30, 16)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(16, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def load_datasets():
    class BreastCancerDataset(torch.utils.data.Dataset):
        def __init__(self, hugging_face_dataset, transform=None):
            df = hugging_face_dataset.to_pandas()
            self.y = torch.tensor(df['diagnosis'].values, dtype=torch.float32).unsqueeze(1)
            self.x = torch.tensor(df.drop(columns=['diagnosis']).values, dtype=torch.float32)
            self.transform = transform

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            x = self.x[idx]
            if self.transform:
                x = self.transform(x)
            return x, self.y[idx]

    dataset = load_dataset("scikit-learn/breast-cancer-wisconsin")['train']
    dataset = dataset.remove_columns(['id', 'Unnamed: 32'])
    dataset = dataset.class_encode_column('diagnosis')
    split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='diagnosis')

    train_dataset = BreastCancerDataset(split['train'])
    mean = train_dataset.x.mean(dim=0)
    std = train_dataset.x.std(dim=0)
    normalize = lambda x: (x - mean) / std
    train_dataset.transform = normalize

    val_dataset = BreastCancerDataset(split['test'], transform=normalize)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))

    return train_loader, val_loader

if __name__ == '__main__':
    torch.manual_seed(42)

    learning_rate = 5e-4
    number_of_epochs = 100

    net = MLP()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using {device}')
    net = net.to(device)

    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_function = torch.nn.BCEWithLogitsLoss()

    train_loader, val_loader = load_datasets()

    eta_min = 1e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=number_of_epochs, eta_min=eta_min)

    train_loss_by_epoch = []
    train_acc_by_epoch = []
    val_loss_by_epoch = []
    val_acc_by_epoch = []

    epochs = list(range(number_of_epochs))
    for epoch in epochs:
        net.train()
        train_predictions = []
        train_losses = []
        for x, y in tqdm.tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            outputs = net(x)

            loss = loss_function(outputs, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            predictions = (outputs > 0).float()
            train_predictions.append((predictions == y).float().mean().item())

        net.eval()
        val_results = []
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                outputs = net(x)

                loss = loss_function(outputs, y)

                val_losses.append(loss.item())
                predictions = (outputs > 0).float()
                val_results.append((predictions == y).float().mean().item())

        scheduler.step()

        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_predictions)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_results)

        print(f'Epoch {epoch}')
        print(f'\tTrain:\t\tLoss {train_loss},\tAcc {train_acc}')
        print(f'\tValidation:\tLoss {val_loss},\tAcc {val_acc}')

        train_loss_by_epoch.append(train_loss)
        train_acc_by_epoch.append(train_acc)
        val_loss_by_epoch.append(val_loss)
        val_acc_by_epoch.append(val_acc)

    _, axes = plt.subplots(2, 1, sharex=True)
    axes[0].set_ylabel('Accuracy')
    axes[0].plot(epochs, train_acc_by_epoch, label='Train')
    axes[0].plot(epochs, val_acc_by_epoch, label='Validation')
    axes[0].legend()

    axes[1].set_ylabel('Loss')
    axes[1].plot(epochs, train_loss_by_epoch, label='Train')
    axes[1].plot(epochs, val_loss_by_epoch, label='Validation')

    axes[1].set_xlabel('Epochs')
    axes[1].set_xticks(range(0, number_of_epochs, 8))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{net.name}_{timestamp}.png')

