import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB


def values_to_bins(data, bins):
    return np.digitize(data, bins) - 1


train_images = np.loadtxt('data/train_images.txt')  # incarcam imaginile
# incarcam etichetele avand  							   # tipul de date int
train_labels = np.loadtxt('data/train_labels.txt').astype(int)
# image = train_images[1, :]  # prima imagine

test_images = np.loadtxt('data/test_images.txt')  # incarcam imaginile
# incarcam etichetele avand  							   # tipul de date int
test_labels = np.loadtxt('data/test_labels.txt').astype(int)

# corespunzator
# Atentie! In cazul nostru indexarea elementelor va
# incepe de la 1, intrucat nu avem valori < 0

# returneaza pentru fiecare element intervalul

# for num_bins in range(1, 11):
#     print(f"Num bins {num_bins}")
#     # returneaza intervalele
#     bins = np.linspace(start=0, stop=255, num=num_bins)
#     train_to_bins = values_to_bins(train_images, bins)
#     test_to_bins = values_to_bins(test_images, bins)
#     naive_bayes_model = MultinomialNB()
#     naive_bayes_model.fit(train_to_bins, train_labels)
#     print(naive_bayes_model.score(test_to_bins, test_labels))

bins = np.linspace(start=0, stop=255, num=5)
train_to_bins = values_to_bins(train_images, bins)
test_to_bins = values_to_bins(test_images, bins)

# image = np.reshape(test_to_bins[0], (28, 28))
# plt.imshow(image.astype(np.uint8), cmap='gray')
# plt.show()

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(train_to_bins, train_labels)
confusion_matrix = np.zeros((10, 10))
for index, prediction in enumerate(naive_bayes_model.predict(test_to_bins)):
    # if prediction != test_labels[index]:
    #     plt.title(f"{test_labels[index]} a fost identificat ca {prediction}")
    #     plt.imshow(np.reshape(test_images[index, :], (28, 28)).astype(
    #         np.uint8), cmap='gray')
    #     plt.show()
    confusion_matrix[test_labels[index]][prediction] += 1

print(confusion_matrix)
