import torch
import torch.nn as nn
import json
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import TensorDataset, DataLoader, random_split

# Load JSON
with open('dummy_data.json', 'r') as f:
    data = json.load(f)

xyz_list, theta_list, score_list, image_list = [], [], [], []

for entry in data:
    xyz_list.append([entry['x'], entry['y'], entry['z']])
    theta_list.append([entry['theta']])
    score_list.append([entry['grasp_score']])

    img_np = np.array(entry['image'], dtype=np.float32) / 255.0  # shape (300, 300, 3)
    img_np = np.transpose(img_np, (2, 0, 1))  # (3,300,300)
    img_tensor = torch.tensor(img_np, dtype=torch.float32)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    image_list.append(img_tensor)

# Convert to tensors
xyz = torch.tensor(xyz_list, dtype=torch.float32)
theta = torch.tensor(theta_list, dtype=torch.float32)
score = torch.tensor(score_list, dtype=torch.float32)
images = torch.stack(image_list)  # (batch, 3, 300, 300)


#Normalize
# Stack xyz and theta for normalization
pose = torch.cat([xyz, theta], dim=1)  # (N, 4)

# Compute mean and std
pose_mean = pose.mean(dim=0)
pose_std = pose.std(dim=0)

# Avoid division by zero
pose_std[pose_std == 0] = 1.0

# Normalize
xyz = (xyz - pose_mean[:3]) / pose_std[:3]
theta = (theta - pose_mean[3:]) / pose_std[3:]
torch.save({'mean': pose_mean, 'std': pose_std}, 'pose_stats.pt')

# Dataset
class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, images, xyz, theta, score):
        self.images = images
        self.xyz = xyz
        self.theta = theta
        self.score = score

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        input_info = torch.cat([self.xyz[idx], self.theta[idx], self.score[idx]], dim=0)
        target = torch.cat([self.xyz[idx], self.theta[idx]], dim=0)
        return self.images[idx], input_info, target

dataset = GraspDataset(images, xyz, theta, score)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Models
class FeatureExtractor(nn.Module):
    def __init__(self, features_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            img = torch.zeros(1, 3, 300, 300)
            n_flatten = self.cnn(img).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x))

class GraspMLP(nn.Module):
    def __init__(self, input_dim=261, output_dim=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device)
grasp_mlp = GraspMLP().to(device)

optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + list(grasp_mlp.parameters()), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(20):
    feature_extractor.train()
    grasp_mlp.train()
    total_loss = 0.0

    for imgs, in_vec, target in train_loader:
        imgs, in_vec, target = imgs.to(device), in_vec.to(device), target.to(device)
        features = feature_extractor(imgs)  # (B, 256)
        mlp_input = torch.cat([in_vec, features], dim=1)  # (B, 261)
        pred = grasp_mlp(mlp_input)

        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/20] Training Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
feature_extractor.eval()
grasp_mlp.eval()
val_loss = 0.0
with torch.no_grad():
    for imgs, in_vec, target in val_loader:
        imgs, in_vec, target = imgs.to(device), in_vec.to(device), target.to(device)
        features = feature_extractor(imgs)
        mlp_input = torch.cat([in_vec, features], dim=1)
        pred = grasp_mlp(mlp_input)
        val_loss += criterion(pred, target).item()
        print("Predicted grasp pose shape:", pred.shape)
        # print("Predicted grasp poses (normalized):", pred)
        # print("Ground truth poses (normalized):", target)
        stats = torch.load('pose_stats.pt')
        pose_mean = stats['mean'].to(device)
        pose_std = stats['std'].to(device)

        pred_denorm = pred * pose_std + pose_mean
        # target_denorm = target * pose_std + pose_mean

        # print("Predicted grasp poses (denormalized):", pred_denorm)
        # print("Ground truth poses (denormalized):", target_denorm)
        grasp_scores = in_vec[:, 4]  
        best_index = torch.argmax(grasp_scores)
        best_grasp = pred_denorm[best_index]

        print("Best predicted grasp pose (denormalized):", best_grasp)




print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
