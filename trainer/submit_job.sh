#!/bin/bash

set -ex

DATASET_NAME=$1
JOB_SUFFIX=$2
CL=$3
TRAIN_PATH=$4

NUM_GPUS=${5:-1}

# submit from one directory up from trainer/
cd "$(dirname "$0")/.."

JOB_DIR="gs://hx-lawrence/geneml-jobs"
DATE=$(TZ=America/Los_Angeles date +%Y%m%d)
JOB_NAME="geneml_${DATE}_${DATASET_NAME}_${JOB_SUFFIX}"

# https://cloud.google.com/vertex-ai/docs/training/pre-built-containers#tensorflow
export DOCKER_DEFAULT_PLATFORM=linux/amd64
gcloud ai custom-jobs create \
    --region us-central1 \
    --display-name=geneML \
    --args=--context-length,$CL,--dataset-name,$DATASET_NAME,--train-path,$TRAIN_PATH,--job-name,$JOB_NAME,--job-dir,$JOB_DIR,--model-type,gene_ml \
    --worker-pool-spec=machine-type=n1-highmem-16,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,\
executor-image-uri='us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-17.py310:latest',\
script=train_model.py,\
local-package-path=trainer/
