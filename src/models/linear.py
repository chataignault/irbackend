import numpy as np
from sklearn.linear_model import LinearRegression


def model_linear_regression(X, Y):
    lin = LinearRegression()
    lin.fit(X, Y)
    Y_pred_lin = lin.predict(X)

    assert Y_pred_lin.shape == Y.shape

    rss = np.sum((Y-Y_pred_lin)**2)
    var = np.sum((Y)**2)
    print(f"Linear R2: {1 - rss / var}")

    return lin
