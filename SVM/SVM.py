import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch

df = pd.read_csv('SVM/data.csv')
df.sample(5)
df.drop(['id'], axis='columns', inplace=True)
df.drop(['Unnamed: 32'], axis='columns', inplace=True)
df.shape
# for col in df.columns:
#       print(f"{col}: {df[col].dtype}")

df['diagnosis_ismalignant'] = df['diagnosis'].map({'M': 1, 'B': 0})
df.drop(['diagnosis'], axis='columns', inplace=True)
df.head(5)
df.isnull().sum()
df.duplicated().sum()
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df.describe().loc[['min', 'max']]

X = df.drop(['diagnosis_ismalignant'], axis='columns')
y = df['diagnosis_ismalignant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

class dataset(Dataset):
    def __init__(self, x, y):
          self.x = torch.tensor(x.values, dtype=torch.float32)
          self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
         return len(self.x)

    def __getitem__(self, index):
         return self.x[index], self.y[index]

training_data = dataset(X_train_scaled, y_train)
test_data = dataset(X_test_scaled, y_test)

train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)



hiddenNeurons= 100
class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.name = 'SVM'
        self.input = nn.Linear(X.shape[1], hiddenNeurons)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hiddenNeurons, 1)

    def forward(self, x):
          x=self.input(x)
          self.relu(x)
          x = self.linear(x)
          return x

model = SVM()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train_scaled.to_numpy(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test_scaled.to_numpy(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)

epochs = 100
for epoch in range(epochs):
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            y_pred_test = model(X_test_tensor)
            y_pred_class = (torch.sigmoid(y_pred_test) > 0.5).float()

            acc = (y_pred_class == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {acc:.4f}")
