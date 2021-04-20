import numpy as np
import matplotlib.pyplot as plt

# a
images = np.array([np.load(f"./images/car_{i}.npy") for i in range(9)])
# plt.imshow(images[4])
# plt.show()

# b
print(np.sum(images))

# c
sum_pixels = [np.sum(img) for img in images]
print(sum_pixels)

# d
print(np.argmax(sum_pixels))

# for img in images:
#     plt.figure()
#     plt.imshow(img)
#     plt.show()

# e
mean_img = np.mean(images, axis=0)
plt.imshow(mean_img)
plt.show()

# f
std_devs = images.std(axis=(1, 2))
print(std_devs)

# g
normalized_images = [(images[i] - mean_img) / std_devs[i]
                     for i in range(len(images))]

# for img in normalized_images:
#     plt.figure()
#     plt.imshow(img)
#     plt.show()
print(np.mean(normalized_images, axis=(1, 2)))

# h
cropped_images = np.array([img[200:301, 280:401] for img in images])

for img in cropped_images:
    plt.figure()
    plt.imshow(img)
    plt.show()
