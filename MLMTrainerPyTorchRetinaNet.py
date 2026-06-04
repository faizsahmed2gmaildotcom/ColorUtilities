from config import *
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import random
import os
import datetime
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
import kornia.filters as k_filters

# ────────────────────────────────────────────────
#  Key parameters (same as original)
# ────────────────────────────────────────────────
pattern_training_dir = os.path.join("training-data", "pattern")
weave_training_dir = os.path.join("training-data", "weave")
pattern_full_size = config["general"]["pattern_full_size"]
pattern_crop_size = config["general"]["pattern_crop_size"]
weave_full_size = config["general"]["weave_full_size"]
weave_crop_size = config["general"]["weave_crop_size"]
pattern_batch_size = config["general"]["pattern_batches"]
weave_batch_size = config["general"]["weave_batches"]
pattern_grayscale = False
weave_grayscale = False
epochs = 35
validation_split = 0.2
learning_rate = 1e-5


# ────────────────────────────────────────────────
#  Model - Repurposed RetinaNet (ResNet50-FPN)
# ────────────────────────────────────────────────
class RetinaNetClassifier(nn.Module):
    def __init__(self, _num_classes: int):
        super().__init__()
        # We extract the backbone from a pre-trained RetinaNet.
        # This backbone includes the ResNet-50 + Feature Pyramid Network (FPN).
        retinanet_base = models.detection.retinanet_resnet50_fpn(
            weights="DEFAULT"
        )
        self.backbone = retinanet_base.backbone

        # FPN outputs features at multiple levels (p3 through p7).
        # Each level has 256 channels. We'll pool these for classification.
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, _num_classes)
        )

    def forward(self, x):
        # The FPN backbone returns a dictionary of feature maps at different scales.
        # We use the finest resolution level ('0') for detailed pattern analysis.
        features = self.backbone(x)
        x = features['0']

        x = self.pool(x)
        return self.classifier(x)


# ────────────────────────────────────────────────
#  Kornia / FFT Preprocessing Layers
# ────────────────────────────────────────────────

class BilateralFilterLayer(nn.Module):
    def __init__(self, kernel_size=7, sigma_color=0.1, sigma_space=1.5):
        super().__init__()
        # sigma_color: how much intensity difference is allowed (higher = more smoothing)
        # sigma_space: how much spatial distance is allowed
        self.filter = k_filters.BilateralBlur(
            kernel_size=(kernel_size, kernel_size),
            sigma_color=sigma_color,
            sigma_space=(sigma_space, sigma_space)
        )

    def forward(self, x):
        # Kornia expects (B, C, H, W). If a single image (C, H, W) comes in, unsqueeze it.
        is_single_image = x.ndim == 3
        if is_single_image:
            x = x.unsqueeze(0)

        x = self.filter(x)

        return x.squeeze(0) if is_single_image else x


class FFTLowPassLayer(nn.Module):
    def __init__(self, cutoff_freq=0.15):
        super().__init__()
        self.cutoff = cutoff_freq

    def forward(self, x):
        is_single_image = x.ndim == 3
        if is_single_image:
            x = x.unsqueeze(0)

        # 1. FFT to frequency domain
        f = torch.fft.fftn(x, dim=(-2, -1))
        f_shift = torch.fft.fftshift(f, dim=(-2, -1))

        # 2. Create Mask
        b, c, h, w = x.shape
        center_h, center_w = h // 2, w // 2
        y, x_grid = torch.meshgrid(torch.arange(h, device=x.device),
                                   torch.arange(w, device=x.device), indexing='ij')

        dist = torch.sqrt((y - center_h) ** 2 + (x_grid - center_w) ** 2)
        max_dist = torch.sqrt(torch.tensor(center_h ** 2 + center_w ** 2, device=x.device))
        mask = (dist / max_dist) < self.cutoff

        # 3. Apply and Inverse FFT
        f_shift_filtered = f_shift * mask
        f_filtered = torch.fft.ifftshift(f_shift_filtered, dim=(-2, -1))
        x_filtered = torch.fft.ifftn(f_filtered, dim=(-2, -1))

        output = x_filtered.real
        return output.squeeze(0) if is_single_image else output


# ────────────────────────────────────────────────
#  Data transforms / augmentation
# ────────────────────────────────────────────────
train_transform_pattern = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(contrast=0.5, hue=0.5, saturation=0.25),  # adjust these params?
    transforms.RandomRotation(degrees=3.6),
    transforms.Resize(pattern_full_size, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.RandomCrop(size=pattern_crop_size),
    transforms.RandomGrayscale(int(pattern_grayscale)),
    transforms.ToTensor(),
    BilateralFilterLayer(),
    FFTLowPassLayer(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform_pattern = transforms.Compose([
    transforms.Resize(pattern_full_size, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(pattern_crop_size),
    transforms.RandomGrayscale(int(pattern_grayscale)),
    transforms.ToTensor(),
    BilateralFilterLayer(),
    FFTLowPassLayer(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_transform_weave = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(contrast=0.5),
    transforms.RandomRotation(degrees=3.6),
    transforms.Resize(weave_full_size, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.RandomCrop(size=weave_crop_size),
    transforms.RandomGrayscale(int(weave_grayscale)),
    transforms.ToTensor(),
    BilateralFilterLayer(sigma_color=0.05, sigma_space=1.5),
    FFTLowPassLayer(cutoff_freq=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform_weave = transforms.Compose([
    transforms.Resize(weave_full_size, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(weave_crop_size),
    transforms.RandomGrayscale(int(weave_grayscale)),
    transforms.ToTensor(),
    BilateralFilterLayer(sigma_color=0.05, sigma_space=1.5),
    FFTLowPassLayer(cutoff_freq=0.5),
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
    print(f"Visualizing {data_dir}...")
    # Collect some image paths from all classes
    image_paths = []
    for _ in range(num_samples):
        class_dir = os.path.join(data_dir, random.choice(os.listdir(data_dir)))
        if not os.path.isdir(class_dir):
            continue
        img_path = os.path.join(class_dir, random.choice(os.listdir(class_dir)))
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(img_path)
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


def train(model_name: str, training_dir: str, batch_size: int, plot=False):
    # ── Dataset Loading ────────────────────────────
    base_dataset = datasets.ImageFolder(training_dir)
    n_total = len(base_dataset)
    n_val = int(n_total * validation_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(123)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)

    # To avoid the 'shared transform' bug in Subset, we apply transforms via a wrapper or by class
    class ApplyTransform(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform: x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    train_dataset = ApplyTransform(Subset(base_dataset, train_idx), train_transform_pattern)
    val_dataset = ApplyTransform(Subset(base_dataset, val_idx), val_transform_pattern)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ────────────────────────────────────────────────
    #  Classes
    # ────────────────────────────────────────────────
    class_names = sorted(base_dataset.classes)
    num_classes = len(class_names)
    model_name = f"{model_name}_{num_classes}.pt"
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    model = RetinaNetClassifier(num_classes).to(device)

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
        visualize_transform_samples(training_dir, train_transform_pattern, num_samples=6)

    # ────────────────────────────────────────────────
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
                       os.path.join(checkpoint_dir, model_name))
            print("  → Saved new best model")

    # Final save & plotting (unchanged)
    torch.save(model.state_dict(),
               os.path.join(checkpoint_dir, model_name))

    if plot:
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


if __name__ == '__main__':
    while False:
        visualize_transform_samples(pattern_training_dir, train_transform_pattern)
        # visualize_transform_samples(weave_training_dir, train_transform_weave)
        input()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    train(f"retinanet/weave_best_model{"_grayscale" if pattern_grayscale else ""}", weave_training_dir, weave_batch_size, False)
    train(f"retinanet/pattern_best_model{"_grayscale" if pattern_grayscale else ""}", pattern_training_dir, pattern_batch_size, False)

    # for l2_model_name in os.listdir(l2_training_dir):
    #     l2_training_subdir = os.path.join(l2_training_dir, l2_model_name)
    #     train(l2_training_subdir, l2_model_name)
