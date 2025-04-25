import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


def model_decision_tree_regressor(X, Y):
    dtr = DecisionTreeRegressor(criterion="friedman_mse", splitter="best", max_depth=5)
    dtr.fit(X, Y)
    Y_pred_dt = dtr.predict(X)
    rss = np.sum((Y - Y_pred_dt) ** 2)
    var = np.sum((Y) ** 2)
    print(f"Regression Tree R2: {1 - rss / var}")
    return dtr


def model_random_forest(X, Y):
    rf = RandomForestRegressor(
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
    )
    rf.fit(X, Y)
    Y_pred_rf = rf.predict(X)
    rss = np.sum((Y - Y_pred_rf) ** 2)
    var = np.sum((Y) ** 2)
    print(f"RandomForest R2: {1 - rss / var}")
    return rf


def model_light_gbm(X, Y):
    train_data = lgb.Dataset(X, label=Y.flatten())

    params = {
        # "learning_rate": 0.1,
        # "max_depth": 2,
        # "num_iterations": 5,
        # "lambda_l1": 1.,
        # "num_leaves": 100,
        # "min_data_in_leaf": 1,
        # "feature_fraction": .1,
        # "bagging_fraction": .1,
        "objective": "regression",
        "verbose": 1,
        # "boosting": "dart",
        # "num_iterations": 10,
        # "tree_learner": "feature",
        # "force_col_wise": True,
        # "min_gain_to_split": 0.
    }
    bst = lgb.train(train_set=train_data, params=params, num_boost_round=10)
    Y_pred = bst.predict(X, num_iteration=bst.best_iteration)
    rss = np.sum((Y - Y_pred) ** 2)
    var = np.sum((Y) ** 2)
    print(f"LightGBM R2: {1 - rss / var}")
    return bst
