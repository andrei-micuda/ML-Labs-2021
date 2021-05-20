import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing, svm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error


def normalize_data(train_data, test_data, type="standard"):
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


def ridge_regression_summary():
    for alpha in [1, 10, 100, 1000]:
        mse = []
        mae = []

        kf = KFold(n_splits=3, shuffle=True)
        for tr, te in kf.split(training_data, prices):
            train_data, train_prices = training_data[tr], prices[tr]
            test_data, test_prices = training_data[te], prices[te]
            train_data, test_data = normalize_data(train_data, test_data)
            ridge_regression_model = Ridge(alpha=alpha)
            ridge_regression_model.fit(train_data, train_prices)
            y_pred = ridge_regression_model.predict(test_data)

            mse_value = mean_squared_error(test_prices, y_pred)
            mae_value = mean_absolute_error(test_prices, y_pred)
            mse.append(mse_value)
            mae.append(mae_value)

        print(f"Alpha: {alpha}")
        print(f"MSE: {np.mean(mse)}")
        print(f"MAE: {np.mean(mae)}\n")


def linear_regression_summary():
    mse = []
    mae = []

    kf = KFold(n_splits=3, shuffle=True)
    for tr, te in kf.split(training_data, prices):
        train_data, train_prices = training_data[tr], prices[tr]
        test_data, test_prices = training_data[te], prices[te]
        train_data, test_data = normalize_data(train_data, test_data)
        linear_regression_model = LinearRegression()
        linear_regression_model.fit(train_data, train_prices)
        y_pred = linear_regression_model.predict(test_data)

        mse_value = mean_squared_error(test_prices, y_pred)
        mae_value = mean_absolute_error(test_prices, y_pred)
        mse.append(mse_value)
        mae.append(mae_value)

    print(f"MSE: {np.mean(mse)}")
    print(f"MAE: {np.mean(mae)}")


# load training data
training_data = np.load("data/training_data.npy")
prices = np.load("data/prices.npy")
# print the first 4 samples
# print("The first 4 samples are:\n ", training_data[:4])
# print("The first 4 prices are:\n ", prices[:4])
# shuffle
training_data, prices = shuffle(training_data, prices, random_state=0)
# linear_regression_summary()
# ridge_regression_summary()

train_data, _ = normalize_data(training_data, training_data)
ridge_regression_model = Ridge(alpha=10)
ridge_regression_model.fit(train_data, prices)
weights = ridge_regression_model.coef_
bias = ridge_regression_model.intercept_

print(weights)
print(bias)

w_i = np.argsort(-weights)
print(w_i)
print("Best")
for i in w_i[:2]:
    print(i)

print("Worst")
for i in w_i[-1:-2:-1]:
    print(i)
