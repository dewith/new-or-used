"""Script to make the machine learn."""
# pylint: disable=duplicate-code

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from xgboost import XGBClassifier

from src.utils.config import get_dataset_path
from src.utils.logging import bprint
from src.utils.ml import CustomTFIDF

TARGET_COLUMN = 'condition'


def build_preprocessor() -> tuple[Pipeline, Pipeline, Pipeline]:
    """
    Build the preprocessors for the model.

    Returns
    -------
    tuple[Pipeline, Pipeline, Pipeline]
        Three pipelines for the vectorizer, the categorical preprocessor, and
        the numerical preprocessor.
    """
    # Create the preprocessors
    vectorizer = Pipeline(
        steps=[
            (
                'tfidf',
                CustomTFIDF(TfidfVectorizer(ngram_range=(1, 2))),
            ),
        ]
    )

    cat_preprocessor = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('target', TargetEncoder()),
        ]
    )

    num_preprocessor = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )
    return vectorizer, cat_preprocessor, num_preprocessor


def train_model(cv: int = 0) -> None:
    """
    Train the model.

    Parameters
    ----------
    cv : int, optional
        The number of folds for cross-validation, by default 0 (no cross-validation).
    """
    # pylint: disable=too-many-locals
    if cv:
        bprint(f'Training model with cv = {cv}', level=1)
    else:
        bprint('Training model with all the data', level=1)

    # Load the data
    bprint('Loading data', level=2)
    df = pd.read_parquet(get_dataset_path('clean_items_train'))
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Separate features by type
    text_features = ['title_norm']
    categorical_features = [
        col
        for col in df.columns
        if df[col].dtype == 'object' and col not in text_features + [TARGET_COLUMN]
    ]
    numerical_features = [
        col
        for col in df.columns
        if df[col].dtype in ['int64', 'float64'] and col != TARGET_COLUMN
    ]

    # Create the preprocessors
    bprint('Building preprocessors', level=2)
    vectorizer, cat_preprocessor, num_preprocessor = build_preprocessor()

    # Create the pipeline
    bprint('Building pipeline', level=2)
    transformers = ColumnTransformer(
        transformers=[
            ('text', vectorizer, text_features),
            ('cat', cat_preprocessor, categorical_features),
            ('num', num_preprocessor, numerical_features),
        ],
        remainder='drop',
        n_jobs=-1,
    )
    classifier = XGBClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[('features', transformers), ('classifier', classifier)])

    if cv:
        # Define the scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1_score': make_scorer(f1_score),
        }

        # Perform cross-validation
        bprint('Performing cross-validation score', level=2)
        cv_scores = cross_validate(pipeline, x, y, cv=cv, scoring=scoring, n_jobs=-1)
        bprint('Performing cross-validation predict:', level=2)
        y_pred = cross_val_predict(pipeline, x, y, cv=cv, n_jobs=-1)

        bprint('Cross-validation metrics:', level=3)
        for metric, scores in cv_scores.items():
            if metric in ('fit_time', 'score_time'):
                continue
            bprint(f'{metric:<16} {scores.mean():.4f} ± {scores.std():.4f}', level=4)

        bprint('Classification report:', level=3)
        bprint(classification_report(y, y_pred), level=4)

        bprint('Confusion matrix:', level=3)
        bprint(
            confusion_matrix(y, y_pred, labels=[0, 1], normalize='true').round(3),
            level=4,
        )
    else:
        # Define the scoring metrics
        scoring = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1_score': f1_score,
        }

        bprint('Training the model', level=2)
        pipeline.fit(x, y)

        bprint('Metrics on train:', level=2)
        y_pred = pipeline.predict(x)
        for metric, scorer in scoring.items():
            score = scorer(y, y_pred)
            bprint(f'{metric:<12} {score:.4f}', level=3)

        bprint('Saving the model', level=2)
        with open(get_dataset_path('model'), 'wb', encoding=None) as f:
            dump(pipeline, f, protocol=5)


if __name__ == '__main__':
    bprint('ML MODELING 🤖', level=0)
    train_model(cv=5)
    train_model()
