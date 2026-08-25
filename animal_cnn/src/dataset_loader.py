"""
=============================================================================
dataset_loader.py — Dataset Loading and Augmentation Pipeline
=============================================================================
"""
import os
import tensorflow as tf
from src import config


def _get_class_names(directory: str) -> list:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    class_names = sorted(e.name for e in os.scandir(directory) if e.is_dir())
    if not class_names:
        raise ValueError(f"No sub-folders found in: {directory}")
    return class_names


def _build_augmentation_layer():
    lo = config.AUG_BRIGHTNESS[0] - 1.0
    hi = config.AUG_BRIGHTNESS[1] - 1.0
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(config.AUG_ROTATION),
        tf.keras.layers.RandomTranslation(
            height_factor=config.AUG_HEIGHT_SHIFT,
            width_factor=config.AUG_WIDTH_SHIFT,
        ),
        tf.keras.layers.RandomZoom(config.AUG_ZOOM),
        tf.keras.layers.RandomBrightness(factor=(lo, hi)),
    ], name="augmentation")


def load_dataset(directory, split_name, class_names=None, shuffle=False, augment=False):
    if class_names is None:
        class_names = _get_class_names(directory)
    num_classes = len(class_names)

    raw_ds = tf.keras.utils.image_dataset_from_directory(
        directory, labels="inferred", label_mode="categorical",
        class_names=class_names, color_mode="rgb",
        batch_size=config.BATCH_SIZE, image_size=config.IMG_SIZE,
        shuffle=shuffle, seed=config.SEED,
    )

    normalise = tf.keras.layers.Rescaling(1.0 / 255.0)

    if augment:
        aug_layer = _build_augmentation_layer()
        def preprocess_train(images, labels):
            return aug_layer(normalise(images), training=True), labels
        dataset = raw_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        def preprocess_eval(images, labels):
            return normalise(images), labels
        dataset = raw_ds.map(preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    num_images = sum(
        len(os.listdir(os.path.join(directory, cls)))
        for cls in class_names if os.path.isdir(os.path.join(directory, cls))
    )
    print(f"[{split_name}] {num_images} images | {num_classes} classes | augment={augment}")
    return dataset, class_names, num_classes


def load_train_dataset():
    return load_dataset(config.TRAIN_DIR, "TRAIN", shuffle=True, augment=True)

def load_validation_dataset(class_names):
    return load_dataset(config.VALIDATION_DIR, "VALIDATION", class_names=class_names, shuffle=False, augment=False)

def load_test_dataset(class_names):
    return load_dataset(config.TEST_DIR, "TEST", class_names=class_names, shuffle=False, augment=False)
