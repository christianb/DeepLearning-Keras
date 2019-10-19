# Deep Learning with Python and Keras

## Setup
You can use the Dockerfile to create an Ubuntu image with all packages needed to run the scripts.

Run the following commands in your shell:
```
docker-compose build

docker-compose run --rm deep_learning /bin/bash
```

## Binary IMDB Classifier
`python3 binary-classifier-imdb.py`

## Multiple Reuters Classifier
`python3 multiple-classifier-reuters.py`
