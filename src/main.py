"""Main script for running the project."""

from src.data import preprocess_dataset, split_dataset
from src.eval import evaluate_test
from src.model import train_model
from src.utils.logging import bprint

if __name__ == '__main__':
    bprint('DATA PREPROCESSING 💽', level=0)
    split_dataset(-10000)
    preprocess_dataset()

    bprint('ML MODELING 🤖', level=0)
    train_model(cv=5)
    train_model()

    bprint('EVALUATION 📊', level=0)
    evaluate_test()

    bprint('DONE 🎉', level=0)
