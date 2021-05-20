from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def normalize_data(train_data, test_data, type=None):
    if type is None:
        return (train_data, test_data)
    elif type == "standard":
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return (scaled_train_data, scaled_test_data)
    else:
        normalizer = preprocessing.Normalizer(norm=type)
        normalizer.fit(train_data)
        normalized_train_data = normalizer.transform(train_data)
        normalized_test_data = normalizer.transform(test_data)
        return (normalized_train_data, normalized_test_data)


class BagOfWords:
    def __init__(self):
        self.vocab = dict()
        self.add_ord = []

    def build_vocabulary(self, data):
        id = 0
        for sentence in data:
            for word in sentence:
                if word not in self.vocab:
                    self.add_ord.append(word)
                    self.vocab[word] = id
                    id += 1

    def get_features(self, data):
        num_samples = len(data)
        dim_dict = len(self.add_ord)
        res = []
        for i, sentence in enumerate(data):
            features = np.zeros(dim_dict)
            for word in sentence:
                id = self.vocab.get(word, None)
                if id is not None:
                    features[id] += 1
            res.append(features)
        return np.array(res)


train_data = np.load("data/training_sentences.npy", allow_pickle=True)
train_labels = np.load("data/training_labels.npy", allow_pickle=True)
test_data = np.load("data/test_sentences.npy", allow_pickle=True)
test_labels = np.load("data/test_labels.npy", allow_pickle=True)
bow = BagOfWords()
bow.build_vocabulary(train_data)
train_bag = bow.get_features(train_data)
test_bag = bow.get_features(test_data)
train_norm, test_norm = normalize_data(train_bag, test_bag, "l2")

model = svm.SVC(kernel="linear")
model.fit(train_norm, train_labels)
preds = model.predict(test_norm)
print(f"Acc: {accuracy_score(test_labels, preds)}")
print(f"F1: {f1_score(test_labels, preds)}")
w = model.coef_[
    0,
]
w_i = list(zip(w, range(len(w))))
w_i.sort(key=lambda x: x[0])
print("\nTOP 10 POSITIVE WORDS")
for _, i in w_i[:10]:
    print(bow.add_ord[i])

print("\nTOP 10 NEGATIVE WORDS")
for _, i in w_i[-1:-10:-1]:
    print(bow.add_ord[i])