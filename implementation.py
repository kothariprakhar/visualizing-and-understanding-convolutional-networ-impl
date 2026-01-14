import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. SETUP & DATA STRATEGY
# ==========================================
# We use CIFAR-10 as a proxy for ImageNet.
# It allows for rapid training and demonstrates the feature visualization
# logic without the massive compute overhead of ImageNet.

def load_data(batch_size=64):
    print("Loading CIFAR-10 Dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
        return trainloader, classes
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# ==========================================
# 2. MODEL ARCHITECTURE (Zeiler-Fergus Inspired)
# ==========================================
# To implement the paper's visualization, we need access to pooling indices.
# Standard nn.Sequential hides these. We build a custom module to store
# 'switches' (max pool indices) needed for the unpooling operation.

class VizuNet(nn.Module):
    def __init__(self):
        super(VizuNet, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Classifier
        self.fc1 = nn.Linear(64 * 8 * 8, 10)
        
        # Storage for Visualization (The "Switches")
        self.switches = {}
        self.feature_maps = {}
        
    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        self.feature_maps['layer1'] = x # Store pre-pool activation
        x, idx1 = self.pool1(x)
        self.switches['pool1'] = idx1
        self.switches['pool1_size'] = x.size()
        
        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        self.feature_maps['layer2'] = x
        x, idx2 = self.pool2(x)
        self.switches['pool2'] = idx2
        self.switches['pool2_size'] = x.size()

        # Flatten & Classify
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc1(x)
        return x

# ==========================================
# 3. DECONVNET LOGIC (The Core Implementation)
# ==========================================

class DeconvNetVisualizer:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def reconstruct(self, input_image, layer_name, feature_idx):
        """
        Implements the reverse mapping described in the paper.
        1. Forward pass input image to populate indices (switches).
        2. Zero out all activations in target layer except the chosen feature_idx.
        3. Propagate backwards using Deconv/Unpool/Relu logic.
        """
        # 1. Forward Pass
        with torch.no_grad():
            _ = self.model(input_image)
        
        # 2. Select and Isolate Target Feature
        # Get the activation from the target layer
        if layer_name == 'layer2':
            # We start reconstruction from Layer 2 post-relu (before pool2)
            # In the paper, they project back from the strongest activation.
            activations = self.model.feature_maps['layer2'].clone()
            
            # Set all other feature maps to zero to isolate `feature_idx`
            mask = torch.zeros_like(activations)
            mask[:, feature_idx, :, :] = 1
            isolated_activation = activations * mask
            
            # --- Backward through Layer 2 ---
            # Inverse ReLU (Rectification)
            # Paper: "We pass the reconstructed signal through a relu"
            recon = nn.functional.relu(isolated_activation)
            
            # Inverse Conv (Filtering)
            # Use Transposed Conv with shared weights
            recon = nn.functional.conv_transpose2d(
                recon, 
                self.model.conv2.weight, 
                padding=2, 
                stride=1
            )
            
            # --- Backward through Layer 1 ---
            # Inverse Pool (Unpooling)
            # We need the indices (switches) from the forward pass
            recon = nn.functional.max_unpool2d(
                recon, 
                self.model.switches['pool1'], 
                kernel_size=2, 
                stride=2, 
                output_size=self.model.feature_maps['layer1'].size()
            )
            
            # Inverse ReLU
            recon = nn.functional.relu(recon)
            
            # Inverse Conv
            recon = nn.functional.conv_transpose2d(
                recon, 
                self.model.conv1.weight, 
                padding=2, 
                stride=1
            )
            
            return recon

        elif layer_name == 'layer1':
             # Simplified for Layer 1
            activations = self.model.feature_maps['layer1'].clone()
            mask = torch.zeros_like(activations)
            mask[:, feature_idx, :, :] = 1
            isolated_activation = activations * mask
            
            recon = nn.functional.relu(isolated_activation)
            recon = nn.functional.conv_transpose2d(
                recon, 
                self.model.conv1.weight, 
                padding=2, 
                stride=1
            )
            return recon
            
        return None

# ==========================================
# 4. TRAINING & UTILITIES
# ==========================================

def train_one_epoch(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    print("Training model for one epoch to learn features...")
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i > 100: break # Short training for demonstration speed

def imshow(img, title):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, classes = load_data()
    if trainloader is None: return

    # Init Model
    model = VizuNet().to(device)
    
    # Train briefly so filters are not random noise
    train_one_epoch(model, trainloader, device)

    # Get a single test image
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    test_img = images[0:1].to(device)

    # Initialize DeconvNet
    visualizer = DeconvNetVisualizer(model)

    # Visualize
    plt.figure(figsize=(12, 6))

    # 1. Original Image
    plt.subplot(1, 3, 1)
    imshow(test_img[0], "Original Input")

    # 2. Visualize a Feature from Layer 1 (e.g., Feature Map 5)
    # Layer 1 usually learns edges/colors
    recon_l1 = visualizer.reconstruct(test_img, 'layer1', feature_idx=5)
    plt.subplot(1, 3, 2)
    # Normalizing visualization for contrast
    recon_img_l1 = recon_l1[0]
    recon_img_l1 = (recon_img_l1 - recon_img_l1.min()) / (recon_img_l1.max() - recon_img_l1.min())
    plt.imshow(recon_img_l1.permute(1, 2, 0).cpu().numpy())
    plt.title("Layer 1 Feature #5\n(Low-level Edge/Color)")
    plt.axis('off')

    # 3. Visualize a Feature from Layer 2 (e.g., Feature Map 10)
    # Layer 2 learns textures/corners
    recon_l2 = visualizer.reconstruct(test_img, 'layer2', feature_idx=10)
    plt.subplot(1, 3, 3)
    recon_img_l2 = recon_l2[0]
    recon_img_l2 = (recon_img_l2 - recon_img_l2.min()) / (recon_img_l2.max() - recon_img_l2.min())
    plt.imshow(recon_img_l2.permute(1, 2, 0).cpu().numpy())
    plt.title("Layer 2 Feature #10\n(Complex Texture/Shape)")
    plt.axis('off')

    plt.tight_layout()
    print("Visualization complete. Displaying results...")
    plt.show()

if __name__ == "__main__":
    main()