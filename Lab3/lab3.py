import numpy as np
import matplotlib.pyplot as plt


def l1(x, y):
    return np.abs(x - y)


def l2(x, y):
    return np.linalg.norm(x - y)


# class KnnClassifier:


class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric="l2"):
        if metric == "l1":
            distances = np.sum(np.abs((self.train_images - test_image)), axis=1)
        else:
            distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1))
        neighbors = np.argsort(distances)[:num_neighbors]
        return np.bincount(self.train_labels[neighbors]).argmax()

    # def classify_image(self, test_image, num_neighbors=3, metric="l2"):
    #     if metric == "l1":
    #         dist = l1
    #     else:
    #         dist = l2

    #     neighbors = [(i, dist(test_image, n)) for i, n in enumerate(self.train_images)]
    #     first_num_neighbors = sorted(neighbors, key=lambda x: x[1])[:num_neighbors]
    #     labels_in_neighbors = dict()
    #     for i, _ in first_num_neighbors:
    #         labels_in_neighbors[self.train_labels[i]] = (
    #             labels_in_neighbors.get(self.train_labels[i], 0) + 1
    #         )
    #     return max(labels_in_neighbors)

    def classify_images(self, test_images, test_labels, num_neighbors=3, metric="l2"):
        wrongly_classified = 0
        for i in range(len(test_images)):
            is_correct = (
                knn.classify_image(test_images[i, :], num_neighbors, metric)
                == test_labels[i]
            )
            if not is_correct:
                wrongly_classified += 1
        return 1.0 - wrongly_classified / len(test_images)


train_images = np.loadtxt("data/train_images.txt")  # incarcam imaginile
train_labels = np.loadtxt("data/train_labels.txt").astype(int)

test_images = np.loadtxt("data/test_images.txt")  # incarcam imaginile
test_labels = np.loadtxt("data/test_labels.txt").astype(int)
knn = KnnClassifier(train_images, train_labels)
# print(f"Accuracy: {knn.classify_images(test_images, test_labels)}")


accuracies_l1 = [
    knn.classify_images(test_images, test_labels, n, "l1") for n in [1, 3, 5, 7, 9]
]
accuracies_l2 = [
    knn.classify_images(test_images, test_labels, n) for n in [1, 3, 5, 7, 9]
]
plt.plot([1, 3, 5, 7, 9], accuracies_l1, color="blue")
plt.plot([1, 3, 5, 7, 9], accuracies_l2, color="orange")
plt.show()