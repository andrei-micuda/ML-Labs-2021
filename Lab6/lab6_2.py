import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


def compute_y(x, W, bias):
    # dreapta de decizie
    # [x, y] * [W[0], W[1]] + b = 0
    return (-x * W[0] - bias) / (W[1] + 1e-10)


def plot_decision_boundary(X, y, W, b, current_x, current_y):
    x1 = -0.5
    y1 = compute_y(x1, W, b)
    x2 = 0.5
    y2 = compute_y(x2, W, b)
    # sterge continutul ferestrei
    plt.clf()
    # ploteaza multimea de antrenare
    color = "r"
    if current_y == -1:
        color = "b"
    plt.ylim((-1, 2))
    plt.xlim((-1, 2))
    plt.plot(X[y == -1, 0], X[y == -1, 1], "b+")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "r+")
    # ploteaza exemplul curent
    plt.plot(current_x[0], current_x[1], color + "s")
    # afisarea dreptei de decizie
    plt.plot([x1, x2], [y1, y2], "black")
    plt.show(block=False)
    plt.pause(0.3)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([-1, 1, 1, -1])
W = np.zeros(2)
b = 0
num_epochs = 70
lr = 0.1
for epoch in range(num_epochs):
    X_shuffled, Y_shufled = shuffle(X, Y, random_state=0)
    predicted = []
    for t in range(len(X)):
        y_hat = np.dot(X_shuffled[t], W) + b
        loss = (y_hat - Y_shufled[t]) ** 2 / 2.0
        W = W - (lr * (y_hat - Y_shufled[t]) * X_shuffled[t])
        b = b - lr * (y_hat - Y_shufled[t])
        predicted.append(np.sign(y_hat))
        plot_decision_boundary(X_shuffled, Y_shufled, W, b, X_shuffled[t], Y_shufled[t])
    # acc = (round(np.array(predicted)) == Y_shuffled).mean()
    # print("Accuracy: ")