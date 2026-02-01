from config import *
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import random

# ────────────────────────────────────────────────
#  Key parameters (same as original)
# ────────────────────────────────────────────────
train_dir = 'training-data'
img_size = config["general"]["img_size"]  # original loaded size
processed_img_size = config["general"]["cropped_img_size"]  # after crop / final model input
batch_size = 3
epochs = 100
validation_split = 0.2
learning_rate = 0.0001


# ────────────────────────────────────────────────
#  Model - ConvNeXt
# ────────────────────────────────────────────────
class ConvnextModelClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        in_features = self.base.classifier[2].in_features
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


# ────────────────────────────────────────────────
#  Data transforms / augmentation
# ────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(contrast=0.5),
    transforms.RandomRotation(degrees=3.6),
    transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),  # First resize to square
    transforms.RandomResizedCrop(size=processed_img_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Then random crop
    transforms.RandomGrayscale(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(processed_img_size),
    transforms.RandomGrayscale(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ────────────────────────────────────────────────
#  Visualize original vs. processed images
# ────────────────────────────────────────────────
def visualize_transform_samples(data_dir, transform, num_samples=6):
    """
    Display side-by-side comparison of original images and their val_transform-processed versions.

    Args:
        data_dir (str): Path to the training data directory (ImageFolder structure)
        transform: The validation transform to apply
        num_samples (int): Number of image pairs to display
    """
    # Collect some image paths from all classes
    image_paths = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
            if len(image_paths) >= num_samples * 2:  # safety margin
                break
        if len(image_paths) >= num_samples * 2:
            break

    if not image_paths:
        print("No images found in the data directory.")
        return

    # Select random samples
    selected_paths = random.sample(image_paths, min(num_samples, len(image_paths)))

    # Create figure
    fig, axes = plt.subplots(nrows=len(selected_paths), ncols=2,
                             figsize=(10, 3 * len(selected_paths)), dpi=300)

    if len(selected_paths) == 1:
        axes = [axes]  # make it iterable

    for i, img_path in enumerate(selected_paths):
        # Load original image
        original_img = Image.open(img_path).convert('RGB')

        # Apply validation transform
        transformed_tensor = transform(original_img)
        # Convert back to display-ready format (undo normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        transformed_img = transformed_tensor * std + mean
        transformed_img = transformed_img.clamp(0, 1)
        transformed_img = transformed_img.permute(1, 2, 0).numpy()

        # Original
        axes[i][0].imshow(original_img)
        axes[i][0].set_title('Original')
        axes[i][0].axis('off')

        # Transformed
        axes[i][1].imshow(transformed_img)
        axes[i][1].set_title('Transformed')
        axes[i][1].axis('off')

        # Show filename/class
        rel_path = os.path.relpath(img_path, data_dir)
        axes[i][0].set_xlabel(rel_path, fontsize=9)

    plt.tight_layout()
    plt.suptitle("Original vs Validation Transform Preview", fontsize=14, y=1.02)
    plt.show()


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
    #  Dataset loading & split
    # ────────────────────────────────────────────────
    base_dataset = datasets.ImageFolder(
        train_dir,
        transform=transforms.ToTensor()
    )

    n_total = len(base_dataset)
    n_val = int(n_total * validation_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(123)
    train_idx, val_idx = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    train_subset = Subset(base_dataset, train_idx.indices)
    val_subset = Subset(base_dataset, val_idx.indices)

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
    class_names = sorted(base_dataset.classes)
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    model = ConvnextModelClassifier(num_classes).to(device)

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
    #  Visualize some validation preprocessing examples
    # ────────────────────────────────────────────────
    if config['general']['debug']:
        print("Showing sample images before/after train_transform...")
        visualize_transform_samples(train_dir, train_transform, num_samples=6)

    # ────────────────────────────────────────────────
    #  Training loop
    # ────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, "best_model.pt"))
            print("  → Saved new best model")

    # Final save & plotting (unchanged)
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