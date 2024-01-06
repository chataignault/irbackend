import numpy as np
import pandas as pd


def generate_data():
    data = pd.DataFrame({"Open": [1.0, 3.0, 4.0], "Date": [1, 2, 3]})
    return data.set_index("Date")


def get_principal_components(M: np.array):
    U, D, V = np.linalg.svd(M)
    return U, D, V
