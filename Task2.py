
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Reproducibility and device
# ==========================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)


# ==========================================
# 2. Load and split the dataset
# ==========================================

data = load_breast_cancer()

X = data.data.astype(np.float32)
y = data.target.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# Fit scaler ONLY on training features to prevent leakage.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)

# Persist preprocessing parameters for future inference.
np.savez(
    "scaler.npz",
    mean=scaler.mean_,
    scale=scaler.scale_
)


# ==========================================
# 3. Create CSV files
# ==========================================

feature_names = list(data.feature_names)

def save_csv(path, features, labels):
    df = pd.DataFrame(features, columns=feature_names)
    df["target"] = labels.astype(np.int64)
    df.to_csv(path, index=False)

save_csv("train.csv", X_train, y_train)
save_csv("val.csv", X_val, y_val)


# ==========================================
# 4. Custom Dataset
# ==========================================

class CsvDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.features = torch.tensor(
            df.drop(columns=["target"]).values,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            df["target"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


train_dataset = CsvDataset("train.csv")
val_dataset = CsvDataset("val.csv")

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

input_dim = X_train.shape[1]


# ==========================================
# 5. Define the MLP
# ==========================================

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Returns logits, not probabilities.
        return self.network(x).squeeze(-1)


model = MLP(input_dim).to(device)

parameter_count = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)
print("Params:", parameter_count)


# ==========================================
# 6. Loss, optimizer and evaluation
# ==========================================

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            predictions = (logits >= 0).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.numel()
            loss_sum += loss.item() * labels.size(0)

    return loss_sum / total, correct / total


# ==========================================
# 7. Training loop + early stopping
# ==========================================

EPOCHS = 40
PATIENCE = 7

best_val_loss = float("inf")
best_val_accuracy = 0.0
best_epoch = 0
best_state = None
epochs_without_improvement = 0

train_losses = []
val_accuracies = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    samples_seen = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        # 1. Clear old gradients
        optimizer.zero_grad()

        # 2. Forward pass
        logits = model(features)

        # 3. Calculate loss
        loss = criterion(logits, labels)

        # 4. Backpropagation
        loss.backward()

        # 5. Update parameters
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        samples_seen += labels.size(0)

    train_loss = running_loss / samples_seen
    val_loss, val_accuracy = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)

    # Keep the best model according to validation accuracy.
    if (
        best_state is None
        or val_accuracy > best_val_accuracy
    ):
        best_val_accuracy = val_accuracy
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

    # Early stopping monitors validation loss.
    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_accuracy:.4f}"
    )

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break


# Restore the best accuracy checkpoint.
model.load_state_dict(best_state)
model.eval()

# Save only model parameters.
torch.save(model.state_dict(), "best_mlp.pth")

print(f"\nBest val accuracy: {best_val_accuracy:.4f}")
print(f"Best epoch: {best_epoch}")


# ==========================================
# 8. Plot training history
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(train_losses)
axes[0].set_title("Training loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE loss")
axes[0].grid(True)

axes[1].plot(val_accuracies)
axes[1].set_title("Validation accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.show()


# ==========================================
# 9. Reload into a NEW model
# ==========================================

reloaded_model = MLP(input_dim).to(device)

state = torch.load(
    "best_mlp.pth",
    map_location=device,
    weights_only=True
)

reloaded_model.load_state_dict(state)
reloaded_model.eval()

original_loss, original_acc = evaluate(model, val_loader)
reloaded_loss, reloaded_acc = evaluate(
    reloaded_model, val_loader
)

print(f"Original model accuracy: {original_acc:.4f}")
print(f"Reloaded model accuracy: {reloaded_acc:.4f}")

assert original_acc == reloaded_acc
print("Reloaded model accuracy is identical.")


# ==========================================
# 10. Compare with logistic regression
# ==========================================

log_reg = LogisticRegression(
    max_iter=2000,
    random_state=SEED
)

log_reg.fit(X_train, y_train.astype(int))

lr_predictions = log_reg.predict(X_val)
lr_accuracy = accuracy_score(
    y_val.astype(int), lr_predictions
)

print(f"\nMLP validation accuracy: {original_acc:.4f}")
print(f"Logistic regression accuracy: {lr_accuracy:.4f}")

print("\nMLP classification report:")
all_labels = []
all_predictions = []

with torch.no_grad():
    for features, labels in val_loader:
        logits = reloaded_model(features.to(device))
        preds = (logits >= 0).int().cpu().numpy()

        all_predictions.extend(preds.tolist())
        all_labels.extend(labels.int().numpy().tolist())

print(classification_report(
    all_labels,
    all_predictions,
    target_names=data.target_names
))
