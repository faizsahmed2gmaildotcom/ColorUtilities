from config import *
import gc
from torchvision import models
from torchvision.transforms import v2
import torch
import torch.nn as nn
import random, os, datetime
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision import datasets
import kornia.filters as k_filters
from collections import Counter
import pixelLib as pL
from sklearn.model_selection import StratifiedShuffleSplit

# ────────────────────────────────────────────────
#  Key parameters
# ────────────────────────────────────────────────
pattern_full_size = config["general"]["pattern_full_size"]
pattern_crop_size = config["general"]["pattern_crop_size"]
weave_full_size = config["general"]["weave_full_size"]
weave_crop_size = config["general"]["weave_crop_size"]
pattern_batch_size = config["general"]["pattern_batches"]
weave_batch_size = config["general"]["weave_batches"]
pattern_grayscale = False
weave_grayscale = True
TOTAL_EPOCHS = 15
WARMUP_EPOCHS = 3  # For transition from LinearLR to CosineAnnealingLR
train_split = 0.75
validation_split = 0.15
learning_rate = 1e-5


# ────────────────────────────────────────────────
#  ConvNeXt Model
# ────────────────────────────────────────────────
class ConvnextModelClassifier(nn.Module):
    def __init__(self, _num_classes: int, stages_to_freeze: int = 0):
        super().__init__()
        self.base = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        in_features = self.base.classifier[2].in_features

        self.base.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, _num_classes)
        )

        if stages_to_freeze > 0:
            self._freeze_stages(stages_to_freeze)

    def _freeze_stages(self, stages_to_freeze: int):
        """
        Freezes the first N*2 stages
        Torchvision ConvNeXt alternates stem/downsample layers with block sequences,
        so 4 full stages equate to 8 internal feature modules.
        """
        target_idx = stages_to_freeze * 2
        for i, stage in enumerate(self.base.features):
            if i < target_idx:
                for p in stage.parameters():
                    p.requires_grad = False

    def unfreeze_all(self):
        """Unfreezes the entire network for full fine-tuning."""
        for p in self.base.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.base(x)


# ────────────────────────────────────────────────
#  Kornia / FFT Preprocessing Layers
# ────────────────────────────────────────────────

class BilateralFilterLayer(nn.Module):
    def __init__(self, kernel_size=7, sigma_color=0.1, sigma_space=1.5):
        super().__init__()
        # kernel_size: size of blur block
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
        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = self.filter(x)

        return x.squeeze(0) if is_single_image else x


# Makes things blurry. Lower cutoff_freq is blurrier.
class FFTLowPassLayer(nn.Module):
    def __init__(self, cutoff_freq=0.2):
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


class PseudoHeightmapLayer(nn.Module):
    def __init__(self, kernel_size=5, eps=1e-2, edge_weight=0.25):
        """
        Converts textile images into structural heightmaps.

        Args:
            kernel_size (int): Size of the local neighborhood window.
            eps (float): Regularization. Smaller values preserve sharper structural edges.
            edge_weight (float): Intensity of the surface relief contours.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        self.edge_weight = edge_weight

    def forward(self, x):
        # Handle batching dimensions for Kornia (B, C, H, W)
        is_single_image = x.ndim == 3
        if is_single_image:
            x = x.unsqueeze(0)

        # 1. Establish single-channel luminance (Base Height Map)
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x

        # 2. Guided Filter: Suppresses chaotic noise, locks onto coherent structures
        # We use the grayscale profile as its own structural guide map.
        smoothed = k_filters.guided_blur(
            guidance=gray,
            input=gray,
            kernel_size=self.kernel_size,
            eps=self.eps
        )

        # 3. Micro-Surface Relief: Emphasizes organized thread borders
        edges = k_filters.sobel(gray)

        # 4. Synthesize Heightmap
        heightmap = smoothed + self.edge_weight * edges
        heightmap = torch.clamp(heightmap, 0.0, 1.0)

        # 5. Broadcast back to 3 channels to maintain ConvNeXt compatibility
        heightmap = heightmap.repeat(1, 3, 1, 1)

        return heightmap.squeeze(0) if is_single_image else heightmap


# ────────────────────────────────────────────────
#  Data transforms / augmentation
# ────────────────────────────────────────────────
class TransformType:
    def __init__(self, train_transform: v2.Compose, val_transform: v2.Compose):
        self.train = train_transform
        self.validation = val_transform


completed_models = [
    "models/shirting/check/main",
    "models/shirting/stripes/main",
    "models/shirting/main",
    "models/shirting/dots/main"
]


class CustomTrivialAugmentWide(v2.TrivialAugmentWide):
    """
    Subclass of TrivialAugmentWide that allows excluding specific transformations
    such as spatial rotations or translations.
    """

    def __init__(self, exclude_ops=None, *args, **kwargs):
        self.exclude_ops = set(exclude_ops) if exclude_ops else set()
        super().__init__(*args, **kwargs)
        self._AUGMENTATION_SPACE = {k: v for k, v in self._AUGMENTATION_SPACE.items() if k not in exclude_ops}


TRANSFORMS_DEFAULT = TransformType(
    v2.Compose([
        # Image transformations
        v2.RandomResizedCrop(size=pattern_crop_size, scale=(0.5, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),

        # Color processing
        CustomTrivialAugmentWide(exclude_ops={"Rotate", "TranslateX", "TranslateY", "ShearX", "ShearY"}),
        v2.RandomGrayscale(int(pattern_grayscale)),

        # Math filters
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        # BilateralFilterLayer(sigma_color=0.5, sigma_space=1.0),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ]),
    v2.Compose([
        v2.Resize(pattern_crop_size),
        v2.RandomGrayscale(int(pattern_grayscale)),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        # BilateralFilterLayer(sigma_color=0.5, sigma_space=1.0),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
)

TRANSFORMS_DOTS = TransformType(
    v2.Compose([
        # Image transformations
        v2.RandomResizedCrop(size=pattern_crop_size, scale=(0.5, 0.5)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),

        # Color processing
        CustomTrivialAugmentWide(exclude_ops={"Rotate", "TranslateX", "TranslateY", "ShearX", "ShearY"}),
        v2.RandomGrayscale(int(pattern_grayscale)),

        # Math filters
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        # BilateralFilterLayer(sigma_color=0.5, sigma_space=1.0),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ]),
    v2.Compose([
        v2.Resize(pattern_full_size),
        v2.CenterCrop([int(pattern_full_size[0] * 0.5), int(pattern_full_size[1] * 0.5)]),
        v2.Resize(pattern_crop_size),
        v2.RandomGrayscale(int(pattern_grayscale)),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        # BilateralFilterLayer(sigma_color=0.5, sigma_space=1.0),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
)

transforms = {
    "models/shirting/check/main": TRANSFORMS_DEFAULT,
    "models/shirting/stripes/main": TRANSFORMS_DEFAULT,
    "models/shirting/main": TRANSFORMS_DEFAULT,
    "models/shirting/dots/main": TRANSFORMS_DOTS
}

train_transform_weave = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.ColorJitter(contrast=0.5),
    v2.RandomRotation(degrees=45),
    v2.Resize(weave_full_size, interpolation=v2.InterpolationMode.LANCZOS),
    v2.CenterCrop(size=weave_crop_size),
    v2.RandomGrayscale(int(weave_grayscale)),
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    BilateralFilterLayer(sigma_color=0.05, sigma_space=1.5),
    FFTLowPassLayer(cutoff_freq=0.5),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

val_transform_weave = v2.Compose([
    v2.Resize(weave_full_size, interpolation=v2.InterpolationMode.LANCZOS),
    v2.CenterCrop(weave_crop_size),
    v2.RandomGrayscale(int(weave_grayscale)),
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

os.makedirs('processed-images', exist_ok=True)


# ────────────────────────────────────────────────
#  Visualize original vs. processed images
# ────────────────────────────────────────────────
def visualize_transform_samples(data_dir: str, transform, is_train_data_dir: bool, num_samples=6):
    """
    Display side-by-side comparison of original images and their val_transform-processed versions.

    Args:
        data_dir (str): Path to the training data directory
        transform: The validation transform to apply
        is_train_data_dir (bool): Whether data_dir is a training data folder or not
        num_samples (int): Number of image pairs to display
    """
    print(f"Visualizing {data_dir}...")
    # Collect some image paths from all classes
    image_paths = []
    if is_train_data_dir:
        for _ in range(num_samples):
            class_dir = os.path.join(data_dir, random.choice(os.listdir(data_dir)))
            if not os.path.isdir(class_dir):
                continue
            img_path = os.path.join(class_dir, random.choice(os.listdir(class_dir)))
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(img_path)
            if len(image_paths) >= num_samples * 2:
                break
    else:
        image_paths = random.sample(os.listdir(data_dir), num_samples)
        image_paths = list(map(lambda p: os.path.join(data_dir, p), image_paths))

    if not image_paths:
        print(f"No images found in {data_dir}")
        return

    selected_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
    fig, axes = plt.subplots(nrows=len(selected_paths), ncols=2, figsize=(10, 3 * len(selected_paths)), dpi=300)
    if len(selected_paths) == 1:
        axes = [axes]  # make it iterable

    for i, img_path in enumerate(selected_paths):
        # Load original image
        original_img = pL.preprocessImagePIL(img_path)
        if original_img is None:
            print(f"Failed to load {img_path}")

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
    save_path = f'processed-images/{data_dir.replace('/', ' ')}'
    save_num = 0
    while os.path.exists(save_path + '.png'):
        save_path = save_path.split('_')[0]
        save_path += '_' + str(save_num)
        save_num += 1

    save_path += '.png'
    print(f'Saved in: {save_path}')
    plt.savefig(save_path)
    plt.close()


plot_folder_path = f'plots/{len(os.listdir('plots')) + 1}'


def plotResults(
        cur_epoch: int,
        train_accs: list[float],
        val_accs: list[float],
        train_losses: list[float],
        val_losses: list[float],
        save_path: str,
        test_acc: float = None,
        test_loss: float = None,
):
    """Plots training/validation metrics and optional final test benchmark lines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epoch_range = range(1, cur_epoch + 1)

    # Subplot 1: Accuracy
    ax1.plot(
        epoch_range,
        train_accs,
        label="Train Accuracy",
        color="#1f77b4",
        marker="o",
        markersize=3,
    )
    ax1.plot(
        epoch_range,
        val_accs,
        label="Val Accuracy",
        color="#ff7f0e",
        marker="o",
        markersize=3,
    )

    # Render Test Accuracy Benchmark if provided
    if test_acc is not None:
        ax1.axhline(
            y=test_acc,
            color="#2ca02c",
            linestyle="--",
            linewidth=2,
            label=f"Test Accuracy ({test_acc:.4f})"
        )

    ax1.set_title("Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Subplot 2: Loss
    ax2.plot(
        epoch_range,
        train_losses,
        label="Train Loss",
        color="#1f77b4",
        marker="o",
        markersize=3,
    )
    ax2.plot(
        epoch_range,
        val_losses,
        label="Val Loss",
        color="#ff7f0e",
        marker="o",
        markersize=3,
    )

    # Render Test Loss Benchmark if provided
    if test_loss is not None:
        ax2.axhline(
            y=test_loss,
            color="#2ca02c",
            linestyle="--",
            linewidth=2,
            label=f"Test Loss ({test_loss:.4f})"
        )

    ax2.set_title("Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    # Save to disk and immediately close to free memory
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def train(model_path: str, training_dir: str, batch_size: int):
    img_transform = transforms[model_path]
    train_base_dataset = datasets.ImageFolder(training_dir, transform=img_transform.train)
    val_base_dataset = datasets.ImageFolder(training_dir, transform=img_transform.validation)
    test_base_dataset = datasets.ImageFolder(training_dir, transform=img_transform.validation)

    class_names = sorted(train_base_dataset.classes)
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    if model_path in completed_models:
        print("Skipped " + model_path + ".pt\n")
        torch.cuda.empty_cache()
        return
    print("Training " + model_path + ".pt")

    targets = train_base_dataset.targets
    sss_outer = StratifiedShuffleSplit(n_splits=1, train_size=train_split, random_state=123)
    train_idx, temp_idx = next(sss_outer.split(range(len(targets)), targets))
    temp_targets = [targets[i] for i in temp_idx]
    val_ratio_in_temp = validation_split / (1.0 - train_split)
    sss_inner = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio_in_temp, random_state=123)
    val_rel_idx, test_rel_idx = next(sss_inner.split(temp_idx, temp_targets))
    val_idx = [temp_idx[i] for i in val_rel_idx]
    test_idx = [temp_idx[i] for i in test_rel_idx]

    train_dataset = Subset(train_base_dataset, train_idx)
    val_dataset = Subset(val_base_dataset, val_idx)
    test_dataset = Subset(test_base_dataset, test_idx)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    model = ConvnextModelClassifier(num_classes).to(device)
    ema_model = AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(0.99)
    )

    optimizer = optim.AdamW([
        {
            'params': filter(lambda p: p.requires_grad, model.base.features.parameters()),
            'lr': 1e-5,
            'weight_decay': 0.05
        },
        {
            'params': model.base.classifier.parameters(),
            'lr': 5e-4,
            'weight_decay': 0.01
        }
    ])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # warmup_scheduler = optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    # )
    # cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=TOTAL_EPOCHS - WARMUP_EPOCHS, T_mult=1, eta_min=1e-7
    # )
    # main_scheduler = optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_scheduler, cosine_scheduler],
    #     milestones=[WARMUP_EPOCHS]  # Transition epoch
    # )
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "fit", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    best_val_acc = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema_model.update_parameters(model)

            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        ema_model.eval()
        val_loss = 0.0
        correct = total = 0

        # Track misclassifications: (true_label, predicted_label)
        misclassifications = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = ema_model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

                # Identify where mistakes happened and store them
                incorrect_mask = ~pred.eq(labels)
                if incorrect_mask.any():
                    true_inc = labels[incorrect_mask].cpu().numpy()
                    pred_inc = pred[incorrect_mask].cpu().numpy()
                    misclassifications.extend(zip(true_inc, pred_inc))

        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:3d}/{TOTAL_EPOCHS} | "
              f"train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
              f"val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Print top misclassifications if errors exist
        if misclassifications:
            counter = Counter(misclassifications)
            # Display up to the top 3 worst misclassification pairs
            top_errors = counter.most_common(3)
            error_strings = [
                f"'{class_names[true]}' confused for '{class_names[pred]}' ({count}x)"
                for (true, pred), count in top_errors
            ]
            print(f"  → Top Errors: {', '.join(error_strings)}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model': ema_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'class_names': class_names,
                'transform': img_transform.validation
            }, model_path + '.pt')
            print("  → Saved new best model")
        torch.save({
            'epoch': epoch,
            'model': ema_model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'class_names': class_names,
            'transform': img_transform.validation
        }, model_path + config['general']['final_suffix'] + '.pt')
        print("  → Saved final model")

        os.makedirs(plot_folder_path, exist_ok=True)
        if epoch != TOTAL_EPOCHS:
            plotResults(epoch, train_accs, val_accs, train_losses, val_losses, f"{plot_folder_path}/{model_path.replace('/', ' ')}.png")
        # main_scheduler.step()
        plateau_scheduler.step(val_acc)

    # Final Evaluation on Isolated Test Split
    print("\nEvaluating best model on Test dataset...")
    testModel(model, model_path, test_loader, criterion, train_accs, val_accs, train_losses, val_losses)
    print("Evaluating final model on Test dataset...")
    testModel(model, model_path + config['general']['final_suffix'], test_loader, criterion, train_accs, val_accs, train_losses, val_losses)

    # Free model memory for next model training
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("Training completed.\n")


def testModel(model, model_path, test_loader, criterion, train_accs, val_accs, train_losses, val_losses):
    checkpoint = torch.load(model_path + '.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    test_loss = 0.0
    correct = total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct / total
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")
    plotResults(TOTAL_EPOCHS, train_accs, val_accs, train_losses, val_losses, f"{plot_folder_path}/{model_path.replace('/', ' ')}.png", test_acc, test_loss)


def getFirstFP(dirpath: str):
    try:
        first_fp = os.listdir(dirpath)[0]
    except IndexError:
        raise FileNotFoundError(dirpath + " is empty!")
    return os.path.join(dirpath, first_fp)


def containsDir(dirpath: str):
    return os.path.isdir(getFirstFP(dirpath))


def fullTrain(training_dirname="", depth=0):
    training_dirpath = os.path.join('training-data', training_dirname)
    if not containsDir(getFirstFP(training_dirpath)):
        return True

    if not os.path.exists(os.path.join('models', training_dirname)):
        os.mkdir(os.path.join('models', training_dirname))

    if (depth > 0) and not os.path.exists(os.path.join(training_dirpath, 'main')):
        raise FileNotFoundError(f"{training_dirpath}/main does not exist!")

    for dirname in os.listdir(training_dirpath):
        cur_training_dir = os.path.join(training_dirname, dirname)
        if fullTrain(cur_training_dir, depth + 1):
            train(os.path.join('models', cur_training_dir), os.path.join('training-data', cur_training_dir), pattern_batch_size)

    return False


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # for _ in range(2):
    #     visualize_transform_samples('training-data/shirting/stripes/main', TRANSFORMS_STRIPES.train, True, 6)
    fullTrain()
