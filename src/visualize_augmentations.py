import tensorflow as tf
import matplotlib.pyplot as plt
from keras_cv import layers as keras_layers
from pathlib import Path

# ---------------- 1. Define augmentation pipeline --------------------
augment = tf.keras.Sequential([
    keras_layers.RandomFlip("horizontal"),
    keras_layers.RandomRotation(factor=0.10, fill_mode="reflect"),
    keras_layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
    keras_layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
    keras_layers.RandomApply(
        keras_layers.RandomContrast(value_range=(0, 1), factor=0.20), rate=0.5
        ),
    keras_layers.RandomApply(
        keras_layers.RandomBrightness(value_range=(0, 1), factor=0.10), rate=0.5
        ),
    keras_layers.RandomApply(
        keras_layers.RandomColorJitter(
            value_range=(0, 1),
            brightness_factor=0.15,
            contrast_factor=0.20,
            saturation_factor=0.15,
            hue_factor=0.05,
            ),
        rate=0.3,
        ),
    keras_layers.RandomApply(
        keras_layers.RandomGaussianBlur(factor=(0.15, 0.4), kernel_size=3), rate=0.25
        ),
    keras_layers.RandomApply(
        keras_layers.RandomShear(x_factor=0.10, y_factor=0.05), rate=0.3
        ),
    ], name="data_augmentation")

# ---------------- 2. Load sample data --------------------------------
(x_train, _), _ = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(8)

# ---------------- 3. Preview and save --------------------------------
def preview_and_save(batch, save_path="/volumes/data/augmentedimg.png"):
    aug_batch = augment(batch, training=True)

    n_cols = 4
    n_rows = len(batch) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    for ax, img in zip(axes.flatten(), aug_batch):
        ax.imshow(img.numpy().clip(0, 1))
        ax.axis("off")

    fig.suptitle("Augmentation preview", y=0.92, fontsize=14)
    plt.tight_layout()

    # Ensure target directory exists and save the file
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved preview to {save_path}")

    plt.show()

for batch in dataset.take(1):
    preview_and_save(batch)

