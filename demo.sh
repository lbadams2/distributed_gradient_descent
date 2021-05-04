#!/bin/bash

build_images() {
    docker build -t grad_calc:1 -f grad_calc.dockerfile .
    docker run --rm -e port=8080 -e seed=10 worker:1

    docker build -t optimizer:1 -f optimizer.dockerfile .
    docker run --rm optimizer:1
}

docker-compose up -d
docker-compose logs -t -f grad_calc_1
docker-compose logs -t -f grad_calc_2
docker-compose logs -t -f grad_calc_3
docker-compose logs -t -f grad_calc_4
docker-compose logs -t -f optimizer
docker-compose down
docker-compose rm

docker cp 99309c8ce5aa:/home/dockerc/ /host/path/target