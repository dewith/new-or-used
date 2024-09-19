"""Text processing utilities."""

import re
import unicodedata

import nltk
import pandas as pd
from nltk.corpus import stopwords


def get_stopwords(set_name='spanish') -> set:
    """
    Get stopwords from NLTK corpus.

    Parameters
    ----------
    set_name : str, optional
        The name of the stopwords language set, by default 'spanish'.

    Returns
    -------
    set
        The set of stopwords.
    """
    try:
        return set(stopwords.words(set_name))
    except Exception:  # pylint: disable=broad-except
        nltk.download('stopwords')
        return set(stopwords.words(set_name))


def normalize_text(text: str, stops: set = None, spaces_unders: bool = False) -> str:
    """
    Normalize text by converting to lowercase, removing accents, punctuation,
    special characters, extra whitespace, and stopwords.

    Parameters
    ----------
    text : str
        The text to normalize.
    stops : set, optional
        The set of stopwords to remove, by default None.
    spaces_unders : bool, optional
        Whether to replace spaces with underscores, by default False.

    Returns
    -------
    str
        The normalized text.
    """

    if pd.isna(text):
        return None

    # Convert to lowercase
    text = text.lower()
    # Remove accents
    text = unicodedata.normalize('NFKD', text)
    # Remove punctuation and special characters
    # text = re.sub(r'[^\w\s]|\d', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    if stops:
        text = ' '.join([word for word in text.split() if word not in stops])
    if spaces_unders:
        text = text.replace(' ', '_')
    return text
