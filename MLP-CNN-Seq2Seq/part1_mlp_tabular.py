import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =====================================================================
# 0. DIRECTORY SETUP
# =====================================================================
# Create necessary folders for data and saved models
os.makedirs('data', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

# =====================================================================
# 1. DATA PIPELINE (CLEANING, ENCODING, SPLITTING, NORMALIZING)
# =====================================================================
file_path = 'data/winequality-red.csv'

# Check if the data is already downloaded
if os.path.exists(file_path):
    print("Loading Wine Quality dataset from local file...")
    df = pd.read_csv(file_path) # Read the local copy (notice we don't need sep=';' here because we saved it as a standard comma CSV)
else:
    print("Downloading Wine Quality dataset from the internet...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    df.to_csv(file_path, index=False) 

# Separate features (X) and target (y)
X = df.drop('quality', axis=1).values
y = df['quality'].values

# Encode target labels to start from 0 (required by PyTorch CrossEntropyLoss)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# Splitting: 70% Train / 15% Validation / 15% Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

# Normalization (Fit on train, transform on val/test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val_scaled)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.LongTensor(y_test)

# Create DataLoaders (Batch size 64)
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

print(f"Data Pipeline Complete! Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# =====================================================================
# 2. MODEL DEFINITIONS
# =====================================================================
input_dim = X_train_t.shape[1] # 11 features
hidden_dim1 = 64
hidden_dim2 = 32
output_dim = num_classes       # 6 classes

# Version 1: nn.Sequential
mlp_sequential = nn.Sequential(
    nn.Linear(input_dim, hidden_dim1),
    nn.ReLU(),
    nn.Linear(hidden_dim1, hidden_dim2),
    nn.ReLU(),
    nn.Linear(hidden_dim2, output_dim)
)

# Version 2: Custom nn.Module
class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        out = self.out(x)
        return out

# =====================================================================
# 3. WEIGHT INITIALIZATION STRATEGIES
# =====================================================================
def init_weights_gaussian(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)

def init_weights_constant(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.1) 
        nn.init.zeros_(m.bias)

def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# =====================================================================
# 4. MAIN TRAINING LOOP
# =====================================================================
def train_model(model, train_loader, val_loader, init_function, model_name, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.apply(init_function)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    best_model_path = f'saved_models/best_{model_name}.pth'
    
    print(f"\nTraining {model_name} on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            
    print(f"Training complete. Best model saved to {best_model_path}")
    return best_model_path

# =====================================================================
# 5. EXECUTION EXPERIMENTS
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("STARTING EXPERIMENTS: INITIALIZATION STRATEGIES")
    print("="*50)

    # 1. Test Gaussian Initialization
    print("\n---> Testing Gaussian Initialization")
    model_gaussian = CustomMLP(input_dim, hidden_dim1, hidden_dim2, output_dim)
    path_gaussian = train_model(model_gaussian, train_loader, val_loader, init_weights_gaussian, "MLP_Gaussian")

    # 2. Test Constant Initialization
    print("\n---> Testing Constant Initialization")
    model_constant = CustomMLP(input_dim, hidden_dim1, hidden_dim2, output_dim)
    path_constant = train_model(model_constant, train_loader, val_loader, init_weights_constant, "MLP_Constant")

    # 3. Test Xavier Initialization
    print("\n---> Testing Xavier Initialization")
    model_xavier = CustomMLP(input_dim, hidden_dim1, hidden_dim2, output_dim)
    path_xavier = train_model(model_xavier, train_loader, val_loader, init_weights_xavier, "MLP_Xavier")

    print("\nAll initialization experiments are complete!")

    # =====================================================================
# 6. MODEL INSPECTION & FINAL EVALUATION ON TEST SET
# =====================================================================
print("\n" + "="*50)
print("FINAL EVALUATION: RELOADING BEST MODEL (XAVIER)")
print("="*50)

# 1. Instantiate a fresh model structure
best_model = CustomMLP(input_dim, hidden_dim1, hidden_dim2, output_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = best_model.to(device)

# 2. Reload the saved state_dict (weights)
best_model_path = 'saved_models/best_MLP_Xavier.pth'
best_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
print(f"Successfully reloaded model weights from {best_model_path}\n")

# 3. Inspect parameters using named_parameters() as required
print("Inspecting a sample of model parameters:")
for name, param in best_model.named_parameters():
    if 'weight' in name:
        print(f"Layer: {name} | Shape: {param.shape} | Requires Grad: {param.requires_grad}")

# 4. Final Evaluation on the unseen TEST SET
print("\nEvaluating on the Test Set...")
best_model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = best_model(X_batch)
        _, predicted = torch.max(outputs, 1)
        
        # Move back to CPU for Scikit-Learn metrics
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

# Calculate Metrics
# We use 'weighted' average because wine quality classes are imbalanced
acc = accuracy_score(all_targets, all_preds)
prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(all_targets, all_preds)

print(f"\n--- FINAL TEST METRICS ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\n--- CONFUSION MATRIX ---")
print(conf_matrix)
print("\nPart I Coding is 100% Complete!")