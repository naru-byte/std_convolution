#!/bin/bash
image_name="sssd2019test"
docker build -t  sssd2019 .
docker run -it --gpus all --name $image_name -v ./data/datasets:/project/data/datasets sssd2019