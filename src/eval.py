"""Script to make the machine learn."""

import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.config import get_dataset_path
from src.utils.logging import bprint
from src.utils.ml import CustomTFIDF  # pylint: disable=unused-import # noqa

TARGET_COLUMN = 'condition'


def evaluate_test() -> None:
    """Evaluate the model on the test set."""
    # pylint: disable=too-many-locals
    # Load the data
    bprint('Loading data', level=2)
    df = pd.read_parquet(get_dataset_path('clean_items_test'))
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Load the model
    bprint('Loading model', level=2)
    with open(get_dataset_path('model'), 'rb') as f:
        model = load(f)

    # Making predictions
    bprint('Making predictions', level=2)
    y_pred = model.predict(x)

    # Define the scoring metrics
    scoring = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score,
    }

    bprint('Metrics:', level=2)
    for metric, scorer in scoring.items():
        score = scorer(y, y_pred)
        bprint(f'{metric:<12} {score:.4f}', level=3)

    bprint('Classification report:', level=2)
    bprint(classification_report(y, y_pred), level=3)

    bprint('Confusion matrix:', level=2)
    bprint(
        confusion_matrix(y, y_pred, labels=[0, 1], normalize='true').round(3), level=3
    )

    # Save the predictions
    pd.DataFrame({'y': y, 'y_pred': y_pred}).to_parquet(get_dataset_path('pred_test'))


if __name__ == '__main__':
    bprint('EVALUATION 📊', level=0)
    evaluate_test()
