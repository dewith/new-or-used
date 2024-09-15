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

Finally, we can run the flows in the Metaflow project.

```bash
$ python src/flows/intermediate/flow.py run --subset 1
$ python src/flows/primary/flow.py run --dev 1 --ratio 0.1
$ python src/flows/modeling/flow.py run --knn_n 100
$ python src/flows/deployment/flow.py run --sagemaker_deploy 1
```

## Project overview

### Data

#### Analysis

### Model

### Pipeline

## Conclusion
