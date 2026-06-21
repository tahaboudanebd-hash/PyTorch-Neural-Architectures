import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# =====================================================================
# 1. DATA PIPELINE: FASHION-MNIST
# =====================================================================
print("Loading Fashion-MNIST Dataset...")

# Images need to be converted to Tensors and normalized. 
# Mean=0.5, Std=0.5 scales the pixels from [0, 1] to [-1, 1].
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load training data
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)

# Download and load testing data
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)

# Create DataLoaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Vision Data Ready! Train images: {len(train_dataset)}, Test images: {len(test_dataset)}\n")

# =====================================================================
# 2. MANUAL IMPLEMENTATIONS (Required by Project Guidelines)
# =====================================================================
print("--- Testing Manual Convolution Operations ---")

def manual_cross_correlation_2d(X, K):
    """
    Computes 2D cross-correlation manually.
    X: Input tensor (Height, Width)
    K: Kernel/Filter (Height, Width)
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

def manual_max_pool_2d(X, pool_size=2, stride=2):
    """
    Computes 2D max pooling manually.
    """
    p_h, p_w = pool_size, pool_size
    Y = torch.zeros(((X.shape[0] - p_h) // stride + 1, (X.shape[1] - p_w) // stride + 1))
    for i in range(0, Y.shape[0]):
        for j in range(0, Y.shape[1]):
            Y[i, j] = X[i*stride : i*stride+p_h, j*stride : j*stride+p_w].max()
    return Y

def manual_avg_pool_2d(X, pool_size=2, stride=2):
    """
    Computes 2D average pooling manually.
    """
    p_h, p_w = pool_size, pool_size
    Y = torch.zeros(((X.shape[0] - p_h) // stride + 1, (X.shape[1] - p_w) // stride + 1))
    for i in range(0, Y.shape[0]):
        for j in range(0, Y.shape[1]):
            Y[i, j] = X[i*stride : i*stride+p_h, j*stride : j*stride+p_w].mean()
    return Y

# =====================================================================
# 3. VERIFICATION EXPERIMENT (Comparing Manual vs PyTorch)
# =====================================================================
# Create a dummy 4x4 input tensor and a 2x2 kernel
X_test = torch.tensor([[0.0, 1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0, 7.0],
                       [8.0, 9.0, 10., 11.],
                       [12., 13., 14., 15.]])

kernel = torch.tensor([[0.0, 1.0], 
                       [2.0, 3.0]])

print("Input X (4x4):\n", X_test)
print("\nManual Cross-Correlation Output:\n", manual_cross_correlation_2d(X_test, kernel))
print("\nManual Max Pooling Output (2x2):\n", manual_max_pool_2d(X_test))
print("\nManual Avg Pooling Output (2x2):\n", manual_avg_pool_2d(X_test))

# =====================================================================
# 4. PYTORCH NATIVE COMPARISON
# =====================================================================
print("\n--- Comparing with PyTorch Native Layers ---")

# PyTorch expects dimensions in (Batch, Channels, Height, Width)
X_torch = X_test.view(1, 1, 4, 4)
kernel_torch = kernel.view(1, 1, 2, 2)

# PyTorch Conv2d (Weight must be explicitly set)
conv_pytorch = nn.Conv2d(1, 1, kernel_size=2, bias=False)
with torch.no_grad():
    conv_pytorch.weight[:] = kernel_torch
    out_conv = conv_pytorch(X_torch)

out_maxpool = F.max_pool2d(X_torch, kernel_size=2, stride=2)
out_avgpool = F.avg_pool2d(X_torch, kernel_size=2, stride=2)

print("PyTorch Conv2d Output:\n", out_conv.squeeze())
print("\nPyTorch Max Pooling Output:\n", out_maxpool.squeeze())
print("\nPyTorch Avg Pooling Output:\n", out_avgpool.squeeze())


import torch.optim as optim

# =====================================================================
# 5. DYNAMIC CNN ARCHITECTURE (For Ablation Studies)
# =====================================================================
class FlexibleLeNet(nn.Module):
    def __init__(self, pool_type='max', use_1x1=False, conv_stride=1, padding=2):
        super(FlexibleLeNet, self).__init__()
        
        self.use_1x1 = use_1x1
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=conv_stride, padding=padding)
        self.pool1 = nn.MaxPool2d(2, 2) if pool_type == 'max' else nn.AvgPool2d(2, 2)
        
        # Optional 1x1 Convolution
        if self.use_1x1:
            self.conv1x1 = nn.Conv2d(6, 6, kernel_size=1)
            
        # Layer 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2) if pool_type == 'max' else nn.AvgPool2d(2, 2)
        
        # --- DYNAMIC FLATTEN CALCULATION ---
        # We do a dummy forward pass with a fake image (1 channel, 28x28) 
        # to calculate exactly how many features will come out of the convolutions.
        dummy_x = torch.zeros(1, 1, 28, 28)
        dummy_x = self.pool1(F.relu(self.conv1(dummy_x)))
        if self.use_1x1: dummy_x = F.relu(self.conv1x1(dummy_x))
        dummy_x = self.pool2(F.relu(self.conv2(dummy_x)))
        flattened_size = dummy_x.numel() 
        
        # Fully Connected Layers using the dynamic size
        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        if self.use_1x1:
            x = F.relu(self.conv1x1(x))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# =====================================================================
# 6. TRAINING LOOP & ABLATION STUDY EXPERIMENTS
# =====================================================================
def train_vision_model(model, train_loader, test_loader, model_name, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n--- Training {model_name} on {device} ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation at the end of the epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Test Accuracy: {val_acc:.2f}%")
        
    return val_acc

if __name__ == "__main__":
    print("\n" + "="*50)
    print("STARTING CNN ABLATION STUDY (Requirement 6)")
    print("="*50)

    # Experiment 1: The Baseline Model (Max Pooling, Standard Padding/Stride)
    model_base = FlexibleLeNet(pool_type='max', use_1x1=False)
    acc_base = train_vision_model(model_base, train_loader, test_loader, "Baseline LeNet (Max Pool)", epochs=5)

    # Experiment 2: Average Pooling instead of Max Pooling
    model_avg = FlexibleLeNet(pool_type='avg', use_1x1=False)
    acc_avg = train_vision_model(model_avg, train_loader, test_loader, "LeNet (Avg Pool)", epochs=5)

    # Experiment 3: Adding the 1x1 Convolution 
    model_1x1 = FlexibleLeNet(pool_type='max', use_1x1=True)
    acc_1x1 = train_vision_model(model_1x1, train_loader, test_loader, "LeNet (With 1x1 Conv)", epochs=5)

    print("\n--- ABLATION STUDY RESULTS ---")
    print(f"Baseline (Max Pool): {acc_base:.2f}%")
    print(f"Average Pooling:     {acc_avg:.2f}%")
    print(f"With 1x1 Conv:       {acc_1x1:.2f}%")

    # =====================================================================
# 7. FEATURE MAP VISUALIZATION
# =====================================================================
print("\n" + "="*50)
print("EXTRACTING FEATURE MAPS")
print("="*50)

# Get a single batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)
sample_image = images[0].unsqueeze(0) # Take the first image and add batch dimension

# Pass the image through the first Convolutional Layer of our best model
model_base.eval()
with torch.no_grad():
    # Move to the correct device
    device = next(model_base.parameters()).device
    sample_image = sample_image.to(device)
    
    # Get the output from conv1
    feature_maps = model_base.conv1(sample_image)
    feature_maps = F.relu(feature_maps)

# Plot the original image and its 6 feature maps
fig, axes = plt.subplots(1, 7, figsize=(15, 3))

# Plot Original Image
axes[0].imshow(sample_image.cpu().squeeze(), cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')

# Plot the 6 feature maps
for i in range(6):
    axes[i+1].imshow(feature_maps[0, i].cpu().squeeze(), cmap='viridis')
    axes[i+1].set_title(f"Filter {i+1}")
    axes[i+1].axis('off')

plt.tight_layout()
plt.savefig("feature_maps_visualization.png")
print("Feature maps saved as 'feature_maps_visualization.png' in your folder.")

# =====================================================================
# 8. COMPARING CNN WITH A SIMPLE MLP ON IMAGES
# =====================================================================
print("\n" + "="*50)
print("COMPARING CNN vs MLP ON FASHION-MNIST")
print("="*50)

# Define a simple MLP for 28x28 images
class SimpleImageMLP(nn.Module):
    def __init__(self):
        super(SimpleImageMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28) # Flatten the image completely
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

mlp_model = SimpleImageMLP()
acc_mlp = train_vision_model(mlp_model, train_loader, test_loader, "Simple MLP", epochs=5)

print("\n--- FINAL COMPARISON ---")
print(f"Best CNN (LeNet): {acc_base:.2f}%")
print(f"Simple MLP:       {acc_mlp:.2f}%")
print("\nDone")
