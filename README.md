<h1 align="center">Items Condition Prediction</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow" alt="Status"/>
  <img src="https://img.shields.io/badge/Version-0.1-yellow" alt="Version"/>
  <img src="https://img.shields.io/badge/Python-3.12-yellow" alt="Python"/>
  <img src="https://img.shields.io/badge/License-Apache-yellow" alt="License"/>
</p>

## Why

In the context of Mercado Libre's marketplace, an algorithm is needed to predict if an item listed in the marketplace is new or used.

This algorithm will help improve the user experience while searching and shopping, because they will know with confidence the condition of their items. And besides that, this could help reduce the cost of returns and chargebacks.

### Why now

The algorithm is needed now because the marketplace is growing and the number of items listed is increasing. This makes it difficult for users to find what they are looking for, and it is important to provide them with the best experience possible.

## Running instructions

### Environment

First, clone the repository and install all dependencies.

```bash
$ git clone https://github.com/dewith/new-or-used.git
$ make install
$ make pre-commit
$ source .env/bin/activate
```

### Dataset

Download the raw data from [here](https://drive.google.com/file/d/1Iphj_MD5LJP7pkxYs14wQ3xW38T5DOy0/).

You should save the raw file at `data/01_raw/MLA_100k_checked_v3.jsonlines`.

### Execution

Finally, we can run the training-evaluation pipeline located on `src/main.py`.

```bash
$ python -m src.main
```

You should see the following output:

```bash
| DATA PREPROCESSING 💽
| Train-test splitting
|     Reading data from data/01_raw/MLA_100k_checked_v3.jsonlines
|         Dataset contains 100,000 items
|         Train/test sets contain 90,000/10,000
|     Writing train and test data
|     Done
| Preprocessing data sets
|     Loading data
|         Train set shape: (90000, 45)
|         Test set shape: (10000, 45)
|     Mapping target label to {'used': 0, 'new': 1}
|         Proportion of each class: {1: 0.537, 0: 0.463}
|     Fitting preprocessor
|         Keeping 62 attributes
|         Keeping 42 categories
|     Transforming data
|         Train set
|             Processing dict columns
|             Processing list columns
|             Processing categorical columns
|             Processing numerical columns
|             Processing boolean columns
|         Test set
|             Processing dict columns
|             Processing list columns
|             Processing categorical columns
|             Processing numerical columns
|             Processing boolean columns
|     Saving preprocessed data
|     Writing preprocessor class with pickle
|     Done
| ML MODELING 🤖
| Training model with cv = 5
|     Loading data
|     Building preprocessors
|     Building pipeline
|     Performing cross-validation score
|     Performing cross-validation predict:
|         Cross-validation metrics:
|             test_accuracy    0.8803 ± 0.0010
|             test_precision   0.8992 ± 0.0030
|             test_recall      0.8752 ± 0.0034
|             test_f1_score    0.8871 ± 0.0010
|         Classification report:
|                           precision    recall  f1-score   support
|
|                        0       0.86      0.89      0.87     41648
|                        1       0.90      0.87      0.89     48352
|
|                 accuracy                           0.88     90000
|                macro avg       0.88      0.88      0.88     90000
|             weighted avg       0.88      0.88      0.88     90000
|         Confusion matrix:
|             [[0.887 0.113]
|              [0.126 0.874]]
| Training model with all the data
|     Loading data
|     Building preprocessors
|     Building pipeline
|     Training the model
|     Saving the model
| EVALUATION ON TEST 📊
|     Loading data
|     Loading model
|     Making predictions
|     Metrics:
|         accuracy     0.8801
|         precision    0.9007
|         recall       0.8746
|         f1_score     0.8875
|     Classification report:
|                       precision    recall  f1-score   support
|
|                    0       0.86      0.89      0.87      4594
|                    1       0.90      0.87      0.89      5406
|
|             accuracy                           0.88     10000
|            macro avg       0.88      0.88      0.88     10000
|         weighted avg       0.88      0.88      0.88     10000
|     Confusion matrix:
|         [[0.887 0.113]
|          [0.125 0.875]]
| DONE 🎉
```
