"""
Module for functions to process the dataset.

Original implementation for reference ->
    def build_dataset():
        data = [json.loads(x) for x in open('MLA_100k_checked_v3.jsonlines')]
        target = lambda x: x.get('condition')
        N = -10000
        X_train = data[:N]
        X_test = data[N:]
        y_train = [target(x) for x in X_train]
        y_test = [target(x) for x in X_test]
        for x in X_test:
            del x['condition']
        return X_train, y_train, X_test, y_test
"""

import json

from src.utils.config import get_dataset_path
from src.utils.logging import bprint


def split_dataset(idx_split: int = -10000):
    """
    Split the dataset into train and test sets.

    Parameters
    ----------
    idx_split : int, optional
        Index to split the dataset into train and test sets, by default -10000.
    """
    bprint('Train-Test Splitting', level=1)

    raw_path = get_dataset_path('raw_items')
    bprint(f'Reading data from {raw_path}', level=2)
    with open(raw_path, 'r', encoding='utf-8') as f:
        data = [json.loads(x) for x in f]

    train = data[:idx_split]
    test = data[idx_split:]
    bprint(f'Dataset contains {len(data):,} items', level=3)
    bprint(f'Train/test sets contain {len(train):,}/{len(test):,}', level=3)

    bprint('Writing train and test data', level=2)
    with open(get_dataset_path('raw_items_train'), 'w', encoding='utf-8') as f:
        for item in train:
            f.write(json.dumps(item) + '\n')

    with open(get_dataset_path('raw_items_test'), 'w', encoding='utf-8') as f:
        for item in test:
            f.write(json.dumps(item) + '\n')

    bprint('Dataset split successfully', level=2)


if __name__ == '__main__':
    bprint('DATA PREPROCESSING 💽', level=0)
    split_dataset()
