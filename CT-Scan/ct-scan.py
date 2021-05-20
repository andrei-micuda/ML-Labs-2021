import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
from tensorflow.keras import (
    layers,
    models,
    preprocessing,
    callbacks,
    regularizers,
    constraints,
)
import matplotlib.pyplot as plt


img_height, img_width, img_channels = 50, 50, 1


def process_csv_entry(entry):
    img_path, label = entry.strip().split(",")
    return img_path, int(label)


def decode_img(img):
    img = PIL.Image.open(img)
    return np.asarray(img)


def load_data(dir_name, csv_file):
    features = []
    labels = []
    with open(csv_file, "r") as fin:
        for entry in fin:
            img_path, label = process_csv_entry(entry)
            img = decode_img(dir_name + "/" + img_path)
            features.append(img)
            labels.append(label)
    return np.stack(features), np.stack(labels)


def preprocess_images(imgs):
    processed_images = np.copy(imgs)
    # processed_images = processed_images / 255.0
    print(processed_images.shape)
    # processed_images = 255 - processed_images
    mean = np.mean(processed_images, axis=(1, 2), keepdims=True)
    std = np.std(processed_images, axis=(1, 2), keepdims=True)
    processed_images = (processed_images - mean) / std
    return processed_images
    # ones = np.ones(2500).reshape((50, 50))
    # return np.subtract(ones, processed_images)


def show_data_sample(imgs, labels):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # ones = np.ones(2500).reshape((50, 50))
        # print()
        plt.imshow(imgs[i])
        # plt.imshow(np.subtract(ones, imgs[i]))
        # plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()


class_names = ["native", "arterial", "venous"]

train_ds, train_labels = load_data(r"./dataset/train", r"./dataset/train.txt")
train_labels = tf.keras.utils.to_categorical(train_labels)
train_ds = preprocess_images(train_ds)
# show_data_sample(train_ds, train_labels)

validation_ds, validation_labels = load_data(
    r"./dataset/validation", r"./dataset/validation.txt"
)
validation_ds = preprocess_images(validation_ds)
validation_labels = tf.keras.utils.to_categorical(validation_labels)


# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(50, 50, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation="relu"))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation="relu"))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dense(3))


# model.compile(
#     optimizer="adam",
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )

# train_images = tf.expand_dims(train_ds, axis=-1)
# validation_images = tf.expand_dims(validation_ds, axis=-1)
# print(train_images.shape)
# print(validation_images.shape)

# model.fit(
#     train_images,
#     train_labels,
#     epochs=5,
#     validation_data=(validation_images, validation_labels),
# )

# ===================================================================================


# test = [
#     preprocess_images(train_ds, 11)[0],
#     preprocess_images(train_ds, 10)[0],
#     preprocess_images(train_ds, 9)[0],
#     preprocess_images(train_ds, 8)[0],
#     preprocess_images(train_ds, 7)[0],
#     preprocess_images(train_ds, 6)[0],
#     preprocess_images(train_ds, 5)[0],
#     preprocess_images(train_ds, 4)[0],
#     preprocess_images(train_ds, 3)[0],
# ]

# np_test = np.array(test)
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(np_test[i])
#     plt.title(11 - i)
#     plt.axis("off")
#     plt.colorbar()
# plt.show()

callback = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)


def add_layer(model, layer, is_input_layer=False, is_output_layer=False):
    weight_decay = 1e-4
    parts = [part.strip() for part in layer.split("~")]
    for part in parts:
        if part.startswith("CL"):
            part = part[2:]
            filters, kernel_size = part.split("C")
            filters = int(filters)
            kernel_size = int(kernel_size)
            if is_input_layer:
                model.add(
                    layers.Conv2D(
                        filters,
                        (kernel_size, kernel_size),
                        activation="relu",
                        input_shape=(img_height, img_width, img_channels),
                        kernel_regularizer=regularizers.l2(),
                    )
                )
            else:
                model.add(
                    layers.Conv2D(
                        filters,
                        (kernel_size, kernel_size),
                        activation="relu",
                        kernel_regularizer=regularizers.l2(weight_decay),
                    )
                )
        elif part.startswith("P"):
            pool_size = int(part[1:])
            # model.add(layers.AvgPool2D(pool_size=(pool_size, pool_size)))
            model.add(layers.MaxPool2D(pool_size=(pool_size, pool_size)))
        elif part.startswith("BN"):
            model.add(layers.BatchNormalization())
        elif part.startswith("DO"):
            rate = int(part[2:]) / 100.0
            model.add(layers.Dropout(rate))
        elif part.startswith("F"):
            model.add(layers.Flatten())
        elif part.startswith("D"):
            units = int(part[1:])
            model.add(
                layers.Dense(
                    units,
                    activation="softmax" if is_output_layer else "relu",
                    kernel_constraint=constraints.MaxNorm(3),
                )
            )


def create_model(pattern):
    model = models.Sequential()
    layers = [layer.strip() for layer in pattern.split("->")]
    num_layers = len(layers)
    model.add(
        tf.keras.layers.experimental.preprocessing.Normalization(
            input_shape=(50, 50, 1)
        )
    )
    # model.add(tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1))
    for i, layer in enumerate(layers):
        add_layer(model, layer, is_output_layer=(i == num_layers - 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def model_testing(model_pattern, epochs):
    # model_patterns = ["CL48C5~P2~BN~DO1 -> CL64C5~P2~BN~DO1 -> CL128C5~BN~DO1 -> F -> D512 -> D512 -> D3"]
    model = create_model(model_pattern)

    model.summary()

    train_images = tf.expand_dims(preprocess_images(train_ds), axis=-1)
    datagen = preprocessing.image.ImageDataGenerator(
        # height_shift_range=[-5, 5],
        horizontal_flip=True,
        rotation_range=15,
        # zoom_range=[0.7, 1.0]
    )
    train_images_it = datagen.flow(tf.expand_dims(train_ds, axis=-1), train_labels)
    validation_images = tf.expand_dims(preprocess_images(validation_ds), axis=-1)

    history = model.fit(
        train_images_it,
        epochs=epochs,
        validation_data=(validation_images, validation_labels),
        # callbacks=[callback],
        # steps_per_epoch=468,
        shuffle=True,
    )

    # PLOT ACCURACIES
    fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 12))
    fig.suptitle(model_pattern)
    ax1.plot(history.history["val_accuracy"], linestyle="solid", color="orange")
    ax1.plot(history.history["accuracy"], linestyle="solid", color="blue")
    ax1.set_xticks(range((epochs)))
    ax1.set_title("Accuracies")
    ax1.set_ylabel("accuracy")
    ax1.set_xlabel("epoch")
    ax1.legend(["val_accuracy", "accuracy"], loc="upper left")

    ax2.plot(history.history["val_loss"], linestyle="solid", color="orange")
    ax2.plot(history.history["loss"], linestyle="solid", color="blue")
    ax2.set_xticks(range((epochs)))
    ax2.set_title("Loss")
    ax2.set_ylabel("loss")
    ax2.set_xlabel("epoch")
    ax2.legend(["val_loss", "loss"], loc="upper left")
    # axes = plt.gca()
    # axes.set_ylim([0.98, 1])
    plt.show()


def model_comparison(epochs):
    # model_patterns = ["CL48C5~P2~BN~DO1 -> CL64C5~P2~BN~DO1 -> CL128C5~BN~DO1 -> F -> D512 -> D512 -> D3"]
    model_patterns = [
        # "CL12C3~P2~BN~D40 -> CL24C3~P2~BN~D20 -> F -> D32 -> D16 -> D3",
        # "CL12C3~P2~BN~D40 -> CL24C3~P2~BN~D20 -> F -> D32 -> D32 -> D3",
        # "CL12C3~P2~BN~D40 -> CL24C3~P2~BN~D40 -> F -> D64 -> D16 -> D3",
        # "CL10C5~P2~BN~D20 -> CL10C5~P2~BN~D20 -> F -> D16 -> D3",
        # "CL8C5~P2~BN~D20 -> CL10C5~P2~BN~D20 -> F -> D16 -> D3",
        # "CL8C5~P2~BN~D20 -> CL8C5~P2~BN~D20 -> F -> D16 -> D3",
        # "CL6C3~P2~BN~D40 -> CL6C3~P2~BN~D40 -> CL6C3~P2~BN~D20 -> F -> D24 -> D3",
        # "CL6C5~P2~BN~D20 -> CL6C5~P2~BN~D20 -> F -> D16 -> D3",
        # "CL12C3~BN~P2~DO40 -> CL24C3~BN~P2~DO20 -> F -> D64 -> D32 -> D3",
        # "CL8C5~BN~P2~DO50 -> CL8C5~BN~P2~DO50 -> CL16C3~BN~P2~DO50 -> CL16C3~BN~P2~DO30 -> F -> D64 -> D64 -> D3",
    ]
    model_num = len(model_patterns)
    history = [0] * model_num
    for i, pattern in enumerate(model_patterns):
        model = create_model(pattern)

        # model.summary()

        train_images = tf.expand_dims(preprocess_images(train_ds), axis=-1)
        datagen = preprocessing.image.ImageDataGenerator(
            # height_shift_range=[-5, 5],
            horizontal_flip=True,
            rotation_range=10,
            vertical_flip=True,
            # zoom_range=[0.8, 1.2]
        )
        train_images_it = datagen.flow(tf.expand_dims(train_ds, axis=-1), train_labels)
        validation_images = tf.expand_dims(preprocess_images(validation_ds), axis=-1)

        history[i] = model.fit(
            train_images_it,
            epochs=epochs,
            validation_data=(validation_images, validation_labels),
            # callbacks=[callback],
            steps_per_epoch=468,
        )

    styles = [
        "solid",
        "dotted",
        "dashed",
        "dashdot",
        "solid",
        "dotted",
        "dashed",
        "dashdot",
        "solid",
    ]

    # PLOT ACCURACIES
    plt.figure(figsize=(15, 5))
    for i in range(len(model_patterns)):
        plt.plot(
            history[i].history["val_accuracy"],
            linestyle=styles[i],
            color=np.random.rand(
                3,
            ),
        )
    plt.title("Model Architecture")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(model_patterns, loc="upper left")
    # axes = plt.gca()
    # axes.set_ylim([0.98, 1])
    plt.show()


def data_augmentation_testing(epochs):
    # model_patterns = ["CL48C5~P2~BN~DO1 -> CL64C5~P2~BN~DO1 -> CL128C5~BN~DO1 -> F -> D512 -> D512 -> D3"]
    # model_pattern = "CL48C5~P2~BN~D20 -> F -> D256 -> D3"
    model_pattern = "CL24C5~P2~BN~D30 -> CL48C5~P2~BN~D30 -> F -> D512 -> D3"
    # model_pattern = "F -> D64 -> D3"
    augmentations = 3
    history = [0] * augmentations
    for i in range(augmentations):
        model = create_model(model_pattern)

        # model.summary()

        train_images = tf.expand_dims(preprocess_images(train_ds), axis=-1)
        if i == 0:
            datagen = preprocessing.image.ImageDataGenerator(
                # height_shift_range=[-5, 5],
                # horizontal_flip=True,
                # rotation_range=10
                brightness_range=[0.7, 1.3]
            )
        elif i == 1:
            datagen = preprocessing.image.ImageDataGenerator(
                # height_shift_range=[-5, 5],
                # horizontal_flip=True,
                # rotation_range=15
                zoom_range=[0.7, 1.3]
            )
        elif i == 2:
            datagen = preprocessing.image.ImageDataGenerator(
                # height_shift_range=[-5, 5],
                # horizontal_flip=True,
                # rotation_range=20
                vertical_flip=True
            )
        elif i == 3:
            datagen = preprocessing.image.ImageDataGenerator(
                # height_shift_range=[-5, 5],
                # horizontal_flip=True,
                # rotation_range=25
            )
        train_images_it = datagen.flow(
            tf.expand_dims(preprocess_images(train_ds), axis=-1), train_labels
        )
        validation_images = tf.expand_dims(preprocess_images(validation_ds), axis=-1)

        history[i] = model.fit(
            train_images_it,
            epochs=epochs,
            validation_data=(validation_images, validation_labels),
            steps_per_epoch=468,
        )

    names = ["30% Brightness Shift", "30% Zoom Shift", "Vertical Flip"]

    styles = [
        "solid",
        "dotted",
        "dashed",
        "dashdot",
        "solid",
        "dotted",
        "dashed",
        "dashdot",
        "solid",
    ]

    # PLOT ACCURACIES
    plt.figure(figsize=(15, 5))
    for i in range(augmentations):
        plt.plot(
            history[i].history["val_accuracy"],
            linestyle=styles[i],
            color=np.random.rand(
                3,
            ),
        )
    plt.title("How many filters per layer?")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(names, loc="upper left")
    # axes = plt.gca()
    # axes.set_ylim([0.98, 1])
    plt.show()


# data_augmentation_testing(20)
model_testing(
    "CL12C7~CL12C7~BN~P2~DO50 -> CL24C5~CL24C5~BN~P2~DO50 -> F -> D120~DO50 -> D64~DO50 -> D3",
    50,
)


# history = [
#     {
#         "history": {
#             "accuracy": [0.36266666650772095, 0.39846667647361755, 0.43299999833106995],
#             "val_accuracy": [
#                 0.3333333432674408,
#                 0.3333333432674408,
#                 0.3333333432674408,
#             ],
#         }
#     },
#     {
#         "history": {
#             "accuracy": [0.344733327627182, 0.3545333445072174, 0.38040000200271606],
#             "val_accuracy": [
#                 0.3333333432674408,
#                 0.3333333432674408,
#                 0.3333333432674408,
#             ],
#         }
#     },
# ]


# for i in range(2):
#     print(history[i])
#     print(
#         "CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
#             names[i],
#             5,
#             max(history[i].history["accuracy"]),
#             max(history[i].history["val_accuracy"]),
#         )
#     )
