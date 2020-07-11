# Deep Learning with Python and Keras

## Setup
You can use the Dockerfile to create an Ubuntu image with all packages needed to run the scripts.

Run the following commands in your shell:
```
docker-compose build

docker-compose run --rm deep_learning /bin/bash
```

## Binary Classifier - IMDB
`python3 binary_classifier_imdb.py`

## Multiple Classifier - Reuters
`python3 multiple_classifier_reuters.py`

## Regression - Boston Housing
`python3 regression_boston_housing.py`
