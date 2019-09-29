# Deep Learning with Python and Keras

## Setup
You can use the Dockerfile to create an Ubuntu image with all packages needed to run the scripts.
If you use `docker-compose` you need to define `PATH_DEEP_LEARNING` as environment variable pointing to this repository.
It will be mounted to the docker container.

Run the following commands in your shell:
```
docker-compose build
docker-compose up

docker exec -ti docker_deep_learning_1 /bin/bash
```

## Binary IMDB Classifier
`python3 binary-classifier-imdb.py`

## Multiple Reuters Classifier
`python3 multiple-classifier-reuters.py`
