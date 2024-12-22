import pandas as pd


def load_birds_csv(path = 'data/raw/birds_dataset.csv'):
    return pd.read_csv(path)