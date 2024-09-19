"""Utilities to make the machine learn."""

from sklearn.base import BaseEstimator, TransformerMixin


class CustomTFIDF(BaseEstimator, TransformerMixin):  # numpydoc ignore=PR02
    """
    A custom wrapper to support column transformer with TF-IDF.

    Parameters
    ----------
    BaseEstimator : class
        Sklearn base estimator.
    TransformerMixin : class
        Sklearn base transformer.
    tfidf : TfidfVectorizer
        The TF-IDF vectorizer.
    """

    # pylint: disable=unused-argument
    def __init__(self, tfidf):  # numpydoc ignore=PR01,RT01
        """Initialize the TF-IDF transformer."""
        self.tfidf = tfidf

    def fit(self, x, y=None):  # numpydoc ignore=PR01,RT01
        """Fit TF-IDF on the dataset."""
        joined_x = x.apply(' '.join, axis=1)
        self.tfidf.fit(joined_x)
        return self

    def transform(self, x):  # numpydoc ignore=PR01,RT01
        """Transform the new dataset."""
        joined_x = x.apply(' '.join, axis=1)
        return self.tfidf.transform(joined_x)

    def get_feature_names_out(self, input_features=None):  # numpydoc ignore=PR01,RT01
        """Get the feature names out."""
        return self.tfidf.get_feature_names_out(input_features)
