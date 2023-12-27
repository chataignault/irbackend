import os
import numpy as np
import pandas as pd


def load_file(data_dir, file_name):
    path = r"C:\Users\leonc\leon\Cours\Cours3A\FI\PCA on Treasury Bonds"
    data = pd.read_csv(os.path.join(path, data_dir, file_name))[["Open", "Date"]]
    return data


def inner_join_df(df_list):
    if len(df_list) == 1:
        return df_list[0]
    df = df_list[0]
    for other in df_list[1:]:
        df = pd.merge(df, other, how="inner", on="Date")
    return df


def unwrap_data(data_dir, name_file_d: dict):
    data_ = []
    for name in name_file_d:
        quotes = pd.DataFrame({"Open": [1.0, 3.0, 4.0], "Date": [1, 2, 3]})
        # load_file(data_dir, name_file_d[name])
        quotes.columns = [name, "Date"]
        data_.append(quotes)
    # concatenate the columns
    data = inner_join_df(data_)
    return data.set_index("Date")


def process_rates(rates: pd.DataFrame, center=True):
    rates = rates.dropna(axis=0)
    if center:
        rates = rates - rates.mean()
    return rates


def get_principal_components(M: np.array):
    U, D, V = np.linalg.svd(M)
    return U, D, V


def load_rates_from_dict(data_dir, d: dict, center=False):
    rates = unwrap_data(data_dir, d)
    clean_rates = process_rates(rates, center=center)
    return clean_rates


treasury_bonds = {
    "13W": "^IRX.csv",
    "5Y": "^FVX.csv",
    "10Y": "^TNX.csv",
    "30Y": "^TYX.csv",
}
data_dir = "./data"
