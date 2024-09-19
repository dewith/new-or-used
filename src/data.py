"""
Module for functions to preprocess the dataset.

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
import pickle

import numpy as np
import pandas as pd

from src.utils.config import get_dataset_path
from src.utils.logging import bprint
from src.utils.text import get_stopwords, normalize_text

STOPWORDS = get_stopwords('spanish')
TARGET_COLUMN = 'condition'


class Preprocessor:
    """
    Class to preprocess the dataset. It needs to be fitted with the dataset in order
    to compute a list of some fields to keep dynamically.
    """

    def __init__(self):
        self.attributes_to_keep = None
        self.categories_to_keep = None
        self.not_useful_columns = [
            # These columnes have many nulls and/or unique values
            'sub_status',
            'site_id',
            'listing_source',
            'parent_item_id',
            'last_updated',
            'international_delivery_mode',
            'id',
            'differential_pricing',
            'thumbnail',
            'date_created',
            'secure_thumbnail',
            'video_id',
            'subtitle',
            'permalink',
            'dimensions',
            'status',
        ]
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Fit the preprocessor.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to fit the preprocessor.
        """
        self.attributes_to_keep = self._get_valid_attributes(df)
        self.categories_to_keep = self._get_valid_categories(df)
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to transform.

        Returns
        -------
        pd.DataFrame
            The transformed dataset.
        """
        # pylint: disable=too-many-statements

        if not self._fitted:
            raise ValueError('Preprocessor not fitted')
        df = df.copy()

        bprint('Processing dict columns', level=4)
        # Drop location information, it won't be used because high cardinality
        df.drop(columns=['seller_address'], inplace=True)

        # Extract shipping information from the dictionary
        shipping_df = pd.json_normalize(df['shipping'])
        shipping_df.columns = [
            'local_pick_up',
            'shipping_methods',
            'shipping_tags',
            'free_shipping',
            'shipping_mode',
            'dimensions',
            'shipping_free_methods',
        ]
        df = pd.concat([df.drop('shipping', axis=1), shipping_df], axis=1)

        bprint('Processing list columns', level=4)
        # Make it a string instead of a list
        df['sub_status'] = df['sub_status'].apply(lambda x: x[0] if x else np.nan)

        # Drop these column since most of the values are empty
        df.drop(columns=['deal_ids'], inplace=True)
        df.drop(columns=['shipping_tags'], inplace=True)

        # Extract payment methods
        payment_df = df['non_mercado_pago_payment_methods'].apply(
            self.extract_pay_methods
        )
        pay_methods_dumms = payment_df.str.get_dummies(sep=' / ')
        pay_methods_dumms['n_payment_methods'] = pay_methods_dumms.sum(axis=1)
        df = pd.concat([df, pay_methods_dumms], axis=1)
        df.drop(columns=['non_mercado_pago_payment_methods'], inplace=True)

        # Make a binary column if the item has variations or not
        df['has_variations'] = df['variations'].apply(lambda x: 1 if x else 0)
        df.drop(columns=['variations'], inplace=True)

        # Extract attributes
        attributes_series = df['attributes'].apply(self.extract_attributes)
        attributes_norm_df = pd.json_normalize(attributes_series)
        attributes_norm_df['n_attributes'] = attributes_norm_df.notna().sum(axis=1)
        attributes_norm_df = attributes_norm_df.notna().astype(int)
        df = pd.concat([df, attributes_norm_df], axis=1)
        df.drop(columns=['attributes'], inplace=True)

        # Turn the tags into flag columns
        df['tags_str'] = df['tags'].apply(' / '.join)
        tags_dummies = df['tags_str'].str.get_dummies(sep=' / ')
        df = pd.concat([df, tags_dummies], axis=1)
        df.drop(columns=['tags', 'tags_str'], inplace=True)

        # Get number of pictures and average resolution
        df['n_pictures'] = df['pictures'].apply(len)
        df['pixels_per_picture'] = df['pictures'].apply(
            lambda x: np.median(
                [self.get_pixels(picture['size']) for picture in x if picture['size']]
            )
            if x
            else np.nan
        )
        df.drop(columns=['pictures'], inplace=True)

        # Let's turn shipping_free_methods into a flag
        df.shipping_free_methods.value_counts(dropna=False)
        df['has_free_shipping'] = df.shipping_free_methods.apply(
            lambda x: 1 if isinstance(x, list) else 0
        )
        df.drop(['shipping_free_methods'], axis=1, inplace=True)

        # Drop coverage_areas since it is empty
        df.drop(columns=['coverage_areas'], inplace=True)

        # Drop descriptions since they are all unique
        df.drop(columns=['descriptions'], inplace=True)

        # Drop shipping_methods since it is empty
        df.drop(columns=['shipping_methods'], inplace=True)

        bprint('Processing categorical columns', level=4)
        # Not useful columns from the beggining
        df.drop(self.not_useful_columns, axis=1, inplace=True)

        # The users just input natural text into warranty
        df.drop(['warranty'], axis=1, inplace=True)

        # Drop categories that are not in the top more common
        df.category_id = df.category_id.where(
            df.category_id.isin(self.categories_to_keep), 'Other'
        )

        # Normalize the title column and get the length of the title
        df['title_norm'] = df.title.apply(normalize_text, stops=STOPWORDS)
        df['title_len'] = df.title_norm.str.len()
        df.drop('title', axis=1, inplace=True)

        bprint('Processing numerical columns', level=4)
        # Drop the seller ID since it is not relevant for the model
        df.drop('seller_id', axis=1, inplace=True)

        # Drop catalog_product_id since it is almost empty
        df.drop('catalog_product_id', axis=1, inplace=True)

        # Drop both because no correlation with the target variable
        df.drop(['start_time', 'stop_time'], axis=1, inplace=True)

        # Just initial_quantity since this is what we will have at the beggining.
        df.drop(['available_quantity', 'sold_quantity'], axis=1, inplace=True)

        # Let's turn this official_store_id into a flag column.
        df['is_official_store'] = df.official_store_id.apply(
            lambda x: 1 if pd.notna(x) else 0
        )
        df.is_official_store.value_counts()
        df.drop('official_store_id', axis=1, inplace=True)

        # Drop the base price since it is very similar to the price column.
        # And let's drop original price too, because is null most of the time.
        df.drop(['base_price', 'original_price'], axis=1, inplace=True)

        bprint('Processing boolean columns', level=4)
        # Not accepting Mercado Pago is a little bit more common in used items.
        df.accepts_mercadopago = df.accepts_mercadopago.astype(int)

        # Automatic relist is not very common when the item is used
        df.automatic_relist = df.automatic_relist.astype(int)

        # Local pick up is not very common when the item is used
        df.local_pick_up = df.local_pick_up.astype(int)

        #  Free shipping is not very common when the item is used
        df.free_shipping = df.free_shipping.astype(int)

        return df

    def extract_pay_methods(self, payment_methods: list[dict]) -> str:
        """
        Extract payment methods from a list of dictionaries.

        Parameters
        ----------
        payment_methods : list[dict]
            The list of payment methods to extract.

        Returns
        -------
        str
            Payment methods concatenated.
        """
        if not payment_methods:
            return None

        card_prefixes = {'visa', 'mastercard', 'american'}

        def categorize_method(method):  # numpydoc ignore=PR01,RT01
            """Categorize payment method."""
            description = method['description'].lower()
            return (
                'pay_tarjeta'
                if any(description.startswith(prefix) for prefix in card_prefixes)
                else f'pay_{description.split()[0]}'
            )

        unique_methods = {categorize_method(method) for method in payment_methods}
        return ' / '.join(unique_methods)

    def extract_attributes(self, attributes: list[dict]) -> dict:
        """
        Extract attributes from a list of dictionaries.

        Parameters
        ----------
        attributes : list[dict]
            The list of attributes to extract.

        Returns
        -------
        dict
            The extracted attributes in a dictionary.
        """
        if not attributes:
            return None
        new_attributes = {}
        for attribute in attributes:
            attribute_name = 'attr_' + normalize_text(
                attribute['name'], stops=STOPWORDS, spaces_unders=True
            )
            value = normalize_text(
                attribute['value_name'], stops=STOPWORDS, spaces_unders=True
            )
            if attribute_name in self.attributes_to_keep:
                new_attributes[f'{attribute_name}'] = value
        return new_attributes

    def get_pixels(self, size: str) -> int:
        """
        Get the number of pixels of a picture.

        Parameters
        ----------
        size : str
            The size of the picture in the format "width x height".

        Returns
        -------
        int
            The number of pixels.
        """
        w, h = map(int, size.split('x'))
        return w * h

    def _get_valid_attributes(self, df: pd.DataFrame) -> list[str]:
        """
        Use gini impurity to get the most informative attributes.

        The hypothesis is that what is informative is not the value of the
        attribute but if the item has that attribute or not.
        For example, the value of the attribute "antigüedad" is not informative per se,
        but if the item has the attribute at all.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to get the valid attributes.

        Returns
        -------
        list[str]
            The list of valid attributes to extract.
        """
        # pylint: disable=too-many-locals
        # Flatten and consolidate attributes
        len_mask = df['attributes'].apply(len) > 0
        attributes_df = pd.json_normalize(df.loc[len_mask, 'attributes'])
        all_attributes = attributes_df.values.tolist()

        # Get most common ones
        attributes = []
        for row in all_attributes:
            for attribute in row:
                if attribute:
                    attribute_name = normalize_text(
                        attribute['name'], stops=STOPWORDS, spaces_unders=True
                    )
                    value = normalize_text(
                        attribute['value_name'], stops=STOPWORDS, spaces_unders=True
                    )
                    attributes.append({'attribute': attribute_name, 'value': value})

        attributes_df = pd.DataFrame(attributes)
        common_attributes = (
            attributes_df['attribute'].value_counts() >= 100
        ).index.tolist()

        # Consolidate attributes
        def clean_attrs(row_attributes):  # numpydoc ignore=PR01,RT01
            """Clean and consolidate attributes."""
            new_attributes = {}
            for attribute in row_attributes:
                attribute_name = normalize_text(
                    attribute['name'], stops=STOPWORDS, spaces_unders=True
                )
                value = normalize_text(
                    attribute['value_name'], stops=STOPWORDS, spaces_unders=True
                )
                if attribute_name in common_attributes:
                    new_attributes[f'attr_{attribute_name}'] = value
            return new_attributes

        attributes_dict_df = df['attributes'].apply(clean_attrs).to_frame()
        attributes_norm_df = pd.json_normalize(attributes_dict_df['attributes'])
        attributes_df = pd.concat([df[['condition']], attributes_norm_df], axis=1)
        attributes_df.fillna('missing', inplace=True)

        # Further filtering of attributes
        attr_count = attributes_norm_df.map(lambda x: 1 if pd.notna(x) else 0).sum(
            axis=0
        )
        usable_attr = (
            attr_count.where(attr_count >= 200)
            .dropna()
            .sort_values(ascending=False)
            .index.tolist()
        )
        attributes_df = attributes_df[['condition'] + usable_attr]

        def gini_impurity(value_counts):  # numpydoc ignore=PR01,RT01
            """Calculate gini impurity."""
            n = value_counts.sum()
            p_sum = 0
            for key in value_counts.keys():
                p_sum = p_sum + (value_counts[key] / n) * (value_counts[key] / n)
            gini = 1 - p_sum
            return gini

        # Calculate Gini Impurity for each attribute
        gini_attiribute = {}
        for attribute_name in attributes_df.columns[1:]:
            # We remove the missing values to compute the gini of the attribute alone.
            # Since most items have no attributes, the nulls bias the gini a lot.
            attributes_df_all = attributes_df[['condition', attribute_name]].query(
                f'{attribute_name} != "missing"'
            )
            attribute_values = attributes_df_all[attribute_name].value_counts()
            gini_all = 0
            for key in attribute_values.keys():
                key_mask = attributes_df_all[attribute_name] == key
                df_k = attributes_df_all[TARGET_COLUMN][key_mask].value_counts()
                n_k = attribute_values[key]
                n = attributes_df_all.shape[0]
                gini_k = gini_impurity(df_k)
                gini_all += (n_k / n) * gini_k

            gini_attiribute[attribute_name] = gini_all

        # Select attributes that have a gini impurity lower than 0.2
        return [k for k, v in gini_attiribute.items() if v < 0.2]

    def _get_valid_categories(self, df: pd.DataFrame) -> list[str]:
        """
        Select categories that are present in more than 0.2% items.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to get the valid categories.

        Returns
        -------
        list[str]
            The list of valid categories.
        """
        threshold = 0.002
        category_counts = df.category_id.value_counts(normalize=True)
        valid_categories = (
            category_counts.where(category_counts >= threshold).dropna().index
        )
        return valid_categories


def preprocess_dataset():
    """Clean the dataset."""
    bprint('Preprocessing data sets', level=1)
    train_path = get_dataset_path('raw_items_train')
    test_path = get_dataset_path('raw_items_test')
    clean_train_path = get_dataset_path('clean_items_train')
    clean_test_path = get_dataset_path('clean_items_test')

    # Load the data into a pandas dataframe
    bprint('Loading data', level=2)
    with open(train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        train_df = pd.DataFrame.from_records([json.loads(line) for line in lines])
        bprint(f'Train set shape: {train_df.shape}', level=3)

    with open(test_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        test_df = pd.DataFrame.from_records([json.loads(line) for line in lines])
        bprint(f'Test set shape: {test_df.shape}', level=3)

    # Label preprocessing
    label_mapping = {'used': 0, 'new': 1}
    bprint('Mapping target label to', label_mapping, level=2)
    train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].map(label_mapping)
    test_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].map(label_mapping)
    label_proportion = train_df[TARGET_COLUMN].value_counts(normalize=True).round(3)
    bprint('Proportion of each class:', label_proportion.to_dict(), level=3)

    # Fitting preprocessor
    bprint('Fitting preprocessor', level=2)
    preprocessor = Preprocessor()
    preprocessor.fit(train_df)
    bprint(f'Keeping {len(preprocessor.attributes_to_keep)} attributes', level=3)
    bprint(f'Keeping {len(preprocessor.categories_to_keep)} categories', level=3)

    # Transform the data
    bprint('Transforming data', level=2)
    bprint('Train set', level=3)
    train_df = preprocessor.transform(train_df)
    bprint('Test set', level=3)
    test_df = preprocessor.transform(test_df)

    # Save the preprocessed data
    bprint('Saving preprocessed data', level=2)
    train_df.to_parquet(clean_train_path, index=False)
    test_df.to_parquet(clean_test_path, index=False)

    # Save the preprocessor
    bprint('Writing preprocessor class with pickle', level=2)
    with open(get_dataset_path('preprocessor'), 'wb') as f:
        pickle.dump(preprocessor, f)

    bprint('Done', level=2)


def split_dataset(idx_split: int = -10000):
    """
    Split the dataset into train and test sets.

    Parameters
    ----------
    idx_split : int, optional
        Index to split the dataset into train and test sets, by default -10000.
    """
    bprint('Train-test splitting', level=1)

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

    bprint('Done', level=2)


if __name__ == '__main__':
    bprint('DATA PREPROCESSING 💽', level=0)
    split_dataset(-10000)
    preprocess_dataset()
