import numpy as np
from reservoirpy.datasets import mackey_glass, to_forecasting
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import rmse, rsquare


def reservoir_echo_state(X: np.ndarray):
    # # create y by shifting X, and train/test split
    x_train, x_test, y_train, y_test = to_forecasting(X, test_size=0.2)
    reservoir = Reservoir(units=100, sr=1.25, lr=0.3)
    readout = Ridge(ridge=1e-5)
    esn = reservoir >> readout

    esn.fit(x_train, y_train, warmup=100)
    predictions = esn.run(x_test)
    print(
        f"RMSE: {rmse(y_test, predictions)}; R^2 score: {rsquare(y_test, predictions)}"
    )

    return esn
