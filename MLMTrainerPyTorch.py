from config import *
from torchvision import transforms, models
import torch.nn as nn

# ────────────────────────────────────────────────
#  Key parameters (same as original)
# ────────────────────────────────────────────────
train_dir = 'training-data'
img_size = config["general"]["img_size"]  # original loaded size
processed_img_size = config["general"]["cropped_img_size"]  # after crop / final model input
batch_size = 6
epochs = 100
validation_split = 0.2
learning_rate = 0.0001

# ────────────────────────────────────────────────
#  Model - ConvNeXt
# ────────────────────────────────────────────────
class ModelClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Load pretrained model
        self.base = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)

        # Extract the number of features after global average pooling
        in_features = self.base.classifier[2].in_features

        # Replace only the linear classifier part, keep LayerNorm2d + Flatten
        self.base.classifier = nn.Sequential(
            self.base.classifier[0],  # LayerNorm2d
            self.base.classifier[1],  # Flatten
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base(x)

# Validation / inference transform (deterministic)
val_transform = transforms.Compose([
    transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(processed_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    import os
    import datetime
    import matplotlib.pyplot as plt
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split, Subset
    from torchvision import datasets

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # ────────────────────────────────────────────────
    #  Dataset loading & split (clean pattern – no double transform)
    # ────────────────────────────────────────────────
    base_dataset = datasets.ImageFolder(
        train_dir,
        transform=transforms.ToTensor()
    )

    # ────────────────────────────────────────────────
    #  Data transforms / augmentation
    # ────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=3.6),
        transforms.ColorJitter(contrast=0.5),
        transforms.RandomResizedCrop(
            size=processed_img_size,
            # scale=(0.8, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    n_total = len(base_dataset)
    n_val = int(n_total * validation_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(123)
    indices = list(range(n_total))
    train_idx, val_idx = random_split(
        indices,
        [n_train, n_val],
        generator=generator
    )

    train_subset = Subset(base_dataset, train_idx.indices)
    val_subset = Subset(base_dataset, val_idx.indices)

    # Apply correct transforms
    train_dataset = Subset(train_subset.dataset, train_subset.indices)
    train_dataset.dataset.transform = train_transform

    val_dataset = Subset(val_subset.dataset, val_subset.indices)
    val_dataset.dataset.transform = val_transform

    # ── DataLoaders ────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # ────────────────────────────────────────────────
    #  Classes
    # ────────────────────────────────────────────────
    class_names = base_dataset.classes
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    model = ModelClassifier(num_classes).to(device)

    # ────────────────────────────────────────────────
    #  Optimizer, loss, directories
    # ────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "fit", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = "MLMs"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_acc = 0.0

    # ────────────────────────────────────────────────
    #  Training loop
    # ────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1:3d}/{epochs} | "
              f"train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
              f"val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, "best_model.pt"))
            print("  → Saved new best model")

    # ────────────────────────────────────────────────
    #  Final save & plotting
    # ────────────────────────────────────────────────
    torch.save(model.state_dict(),
               os.path.join(checkpoint_dir, "final_model.pt"))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_accs, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Training completed.")
