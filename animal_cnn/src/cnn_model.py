"""
=============================================================================
cnn_model.py — Custom CNN Architecture (Designed & Built from Scratch)
=============================================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from src import config


def build_model(num_classes: int) -> tf.keras.Model:
    """
    Build, compile, and return the custom CNN model built from scratch.
    
    Architecture:
      Input (224x224x3)
      -> Block 1: Conv2D(32, 3x3) -> BN -> ReLU -> Conv2D(32, 3x3) -> BN -> ReLU -> MaxPool(2x2)
      -> Block 2: Conv2D(64, 3x3) -> BN -> ReLU -> Conv2D(64, 3x3) -> BN -> ReLU -> MaxPool(2x2)
      -> Block 3: Conv2D(128, 3x3) -> BN -> ReLU -> Conv2D(128, 3x3) -> BN -> ReLU -> MaxPool(2x2)
      -> Block 4: Conv2D(256, 3x3) -> BN -> ReLU -> Conv2D(256, 3x3) -> BN -> ReLU -> MaxPool(2x2)
      -> GlobalAveragePooling2D
      -> Dense(512, ReLU, L2) -> Dropout(0.4)
      -> Dense(num_classes, Softmax)
    """
    kreg = regularizers.L2(config.L2_REG)
    inputs = tf.keras.Input(shape=config.IMG_SHAPE, name="input_image")
    x = inputs

    # --- Conv Block 1 ---
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=kreg, use_bias=False, name="block1_conv1")(x)
    x = layers.BatchNormalization(name="block1_bn1")(x)
    x = layers.Activation("relu", name="block1_relu1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=kreg, use_bias=False, name="block1_conv2")(x)
    x = layers.BatchNormalization(name="block1_bn2")(x)
    x = layers.Activation("relu", name="block1_relu2")(x)
    x = layers.MaxPool2D((2, 2), name="block1_pool")(x)

    # --- Conv Block 2 ---
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=kreg, use_bias=False, name="block2_conv1")(x)
    x = layers.BatchNormalization(name="block2_bn1")(x)
    x = layers.Activation("relu", name="block2_relu1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=kreg, use_bias=False, name="block2_conv2")(x)
    x = layers.BatchNormalization(name="block2_bn2")(x)
    x = layers.Activation("relu", name="block2_relu2")(x)
    x = layers.MaxPool2D((2, 2), name="block2_pool")(x)

    # --- Conv Block 3 ---
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=kreg, use_bias=False, name="block3_conv1")(x)
    x = layers.BatchNormalization(name="block3_bn1")(x)
    x = layers.Activation("relu", name="block3_relu1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=kreg, use_bias=False, name="block3_conv2")(x)
    x = layers.BatchNormalization(name="block3_bn2")(x)
    x = layers.Activation("relu", name="block3_relu2")(x)
    x = layers.MaxPool2D((2, 2), name="block3_pool")(x)

    # --- Conv Block 4 ---
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=kreg, use_bias=False, name="block4_conv1")(x)
    x = layers.BatchNormalization(name="block4_bn1")(x)
    x = layers.Activation("relu", name="block4_relu1")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=kreg, use_bias=False, name="block4_conv2")(x)
    x = layers.BatchNormalization(name="block4_bn2")(x)
    x = layers.Activation("relu", name="block4_relu2")(x)
    x = layers.MaxPool2D((2, 2), name="block4_pool")(x)

    # --- Classification Head ---
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(config.DENSE_UNITS, activation="relu", kernel_initializer="he_normal", kernel_regularizer=kreg, name="dense_head")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_initializer="glorot_uniform", name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="AnimalCNN_Custom")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model
