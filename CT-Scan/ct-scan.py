import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
from tensorflow.keras import layers, models, preprocessing
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
    processed_images = processed_images / 255.0
    mean = np.mean(processed_images, axis=(1, 2), keepdims=True)
    std = np.std(processed_images, axis=(1, 2), keepdims=True)
    processed_images = (processed_images - mean) / std
    return processed_images


def show_data_sample(imgs, labels):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(imgs[i])
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()


data_dir = pathlib.Path("./dataset/train/*")

class_names = ["native", "arterial", "venous"]

train_ds, train_labels = load_data(r"./dataset/train", r"./dataset/train.txt")
train_ds = preprocess_images(train_ds)
# show_data_sample(train_ds, train_labels)

validation_ds, validation_labels = load_data(
    r"./dataset/validation", r"./dataset/validation.txt"
)
validation_ds = preprocess_images(validation_ds)


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
def add_layer(model, layer, is_input_layer, is_output_layer):
    parts = [part.strip() for part in layer.split("~")]
    for part in parts:
        if part.startswith("CL"):
            part = part[2:]
            filters, kernel_size = part.split("C")
            filters = int(filters)
            kernel_size = int(kernel_size)
            if is_input_layer:
                model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation="relu", input_shape=(img_height, img_width, img_channels)))
            else:
                model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation="relu"))
        elif part.startswith("P"):
            pool_size = int(part[1:])
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
            model.add(layers.Dense(units, activation="softmax" if is_output_layer else "relu"))



def create_model(pattern):
    model = models.Sequential()
    layers = [layer.strip() for layer in pattern.split("->")]
    num_layers = len(layers)
    for i, layer in enumerate(layers):
        add_layer(model, layer, is_input_layer=(i == 0), is_output_layer=(i == num_layers - 1))
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model



model_num = 1
history = [0] * model_num
# model_patterns = ["CL48C5~P2~BN~DO1 -> CL64C5~P2~BN~DO1 -> CL128C5~BN~DO1 -> F -> D512 -> D512 -> D3"]
model_patterns = ["CL12C3~P2 -> F -> D64 -> D3"]
for i, pattern in enumerate(model_patterns):
    model = create_model(pattern)

    model.summary

    train_images = tf.expand_dims(preprocess_images(train_ds), axis=-1)
    validation_images = tf.expand_dims(preprocess_images(validation_ds), axis=-1)

    history[i] = model.fit(
        train_images,
        train_labels,
        epochs=3,
        validation_data=(validation_images, validation_labels),
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

# PLOT ACCURACIES
plt.figure(figsize=(15, 5))
for i in range(1):
    plt.plot(history[i].history["val_accuracy"], linestyle=styles[i], color=np.random.rand(3,))
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(model_patterns, loc="upper left")
# axes = plt.gca()
# axes.set_ylim([0.98, 1])
plt.show()
