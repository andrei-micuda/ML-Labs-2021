import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import math


def compute_y(x, W, bias):
    # dreapta de decizie
    # [x, y] * [W[0], W[1]] + b = 0
    return (-x * W[0] - bias) / (W[1] + 1e-10)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def plot_decision(X_, W_1, W_2, b_1, b_2):
    # sterge continutul ferestrei
    plt.clf()
    # ploteaza multimea de antrenare
    plt.ylim((-0.5, 1.5))
    plt.xlim((-0.5, 1.5))
    xx = np.random.normal(0, 1, (100000))
    yy = np.random.normal(0, 1, (100000))
    X = np.array([xx, yy]).transpose()
    X = np.concatenate((X, X_))
    _, _, _, output = forward(X, W_1, b_1, W_2, b_2)
    y = np.squeeze(np.round(output))
    plt.plot(X[y == 0, 0], X[y == 0, 1], "b+")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "r+")
    plt.show(block=False)
    plt.pause(0.1)


def forward(X, W_1, b_1, W_2, b_2):
    # X - datele de intrare,  W_1, b_1, W_2 si b_2 sunt ponderile retelei
    z_1 = X.dot(W_1) + b_1
    a_1 = np.tanh(z_1)
    z_2 = a_1.dot(W_2) + b_2
    a_2 = sigmoid(z_2)
    return z_1, a_1, z_2, a_2  # vom returna toate elementele calculate


def backward(a_1, a_2, z_1, W_2, X, y, num_samples=1):
    dz_2 = a_2 - y  # derivata functiei de pierdere (logistic loss) in functie de z
    dw_2 = (a_1.T * dz_2) / num_samples
    # der(L/w_2) = der(L/z_2) * der(dz_2/w_2) = dz_2 * der((a_1 * W_2 + b_2)/ W_2)
    db_2 = sum(dz_2) / num_samples
    # der(L/b_2) = der(L/z_2) * der(z_2/b_2) = dz_2 * der((a_1 * W_2 + b_2)/ b_2)
    # primul strat
    da_1 = dz_2 * W_2.T
    # der(L/a_1) = der(L/z_2) * der(z_2/a_1) = dz_2 * der((a_1 * W_2 + b_2)/ a_1)
    dz_1 = da_1 * tanh_derivative(z_1)
    # der(L/z_1) = der(L/a_1) * der(a_1/z1) = da_1 .* der((tanh(z_1))/ z_1)
    dw_1 = X.T * dz_1 / num_samples
    # der(L/w_1) = der(L/z_1) * der(z_1/w_1) = dz_1 * der((X * W_1 + b_1)/ W_1)
    db_1 = sum(dz_1) / num_samples
    # der(L/b_1) = der(L/z_1) * der(z_1/b_1) = dz_1 * der((X * W_1 + b_1)/ b_1)
    return dw_1, db_1, dw_2, db_2


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([-1, 1, 1, -1])


num_epochs = 20
lr = 0.5
miu = 0.0
sigma = 1.0
num_hidden_neurons = 5


# np.random.normal(0, 1, (2, num_hidden_neurons))
W_1 = np.random.normal(miu, sigma, (2, num_hidden_neurons))
# generam aleator matricea ponderilor stratului ascuns (2 - dimensiunea datelor de intrare, num_hidden_neurons - numarul neuronilor de pe stratul ascuns), cu media miu  si deviatia standard sigma.
b_1 = np.zeros(num_hidden_neurons)  # initializam bias-ul cu 0
W_2 = np.random.normal(miu, sigma, (num_hidden_neurons, 1))
# generam aleator matricea ponderilor stratului de iesire (num_hidden_neurons - numarul neuronilor de pe stratul ascuns, 1 - un neuron pe stratul de iesire), cu media miu  si deviatia standard sigma.
b_2 = np.zeros(1)  # initializam bias-ul cu 0

for epoch in range(num_epochs):
    X_shuffled, Y_shufled = shuffle(X, Y, random_state=0)

    z_1, a_1, z_2, a_2 = forward(X[0], W_1, b_1, W_2, b_2)

    loss = (-Y_shufled * np.log(a_2) - (1 - Y_shufled) * np.log(1 - a_2)).mean()
    accuracy = (round(a_2) == Y_shufled).mean()
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    dw_1, db_1, dw_2, db_2 = backward(a_1, a_2, z_1, W_2, X_shuffled[0], Y_shufled[0])

    W_1 -= lr * dw_1  # lr - rata de invatare (learning rate)
    b_1 -= lr * db_1
    W_2 -= lr * dw_2
    b_2 -= lr * db_2

    # print(*forward(X[0], W_1, b_1, W_2, b_2), sep="\n\n")