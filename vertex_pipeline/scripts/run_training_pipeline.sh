#!/bin/bash

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cd "$( dirname "${BASH_SOURCE[0]}" )" || exit
DIR="$( pwd )"
SRC_DIR=${DIR}"/../"
export PYTHONPATH=${PYTHONPATH}:${SRC_DIR}
echo "PYTHONPATH=""${PYTHONPATH}"

PROJECT_ID=$(gcloud config get-value project)

# Please modify the following accordingly
AF_REGISTRY_LOCATION=us-central1
AF_REGISTRY_NAME=mlops-vertex-kit
PIPELINE_REGION=us-central1

PIPELINE_ROOT=gs://vertex_pipeline_demo_root_hy/pipeline_root # The GCS path for storing artifacts of pipeline runs
DATA_PIPELINE_ROOT=gs://vertex_pipeline_demo_root_hy/compute_root # The GCS staging location for custom job
GCS_OUTPUT_PATH=gs://vertex_pipeline_demo_root_hy/datasets/training # The GCS path for storing processed data
PIPELINE_REGISTRY=gs://vertex_pipeline_demo_root_hy/pipeline_spec

# Setup for dataset
DATA_REGION=us-central1
# The dataset used throughout the demonstration is
# Banknote Authentication Data Set, you may change according to your needs.
# The schema should be in the format of 'field_name:filed_type;...'
DATA_SCHEMA='reviewtext:string;Class:int'
# Instance used to test deployed model
TEST_INSTANCE='[{"reviewtext": "pet circle is not recommended","Class":"0"}, 
		{"reviewtext": "pet circle is highly recommended","Class":"1"}, 
		{"reviewtext": "think twice before you buy","Class":"0"},
		{"reviewtext": "great product. will buy again.","Class":"1"}]'

# Setup for training
TRAIN_IMAGE_URI=${AF_REGISTRY_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${AF_REGISTRY_NAME}/training:latest
SERVING_IMAGE_URI=${AF_REGISTRY_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${AF_REGISTRY_NAME}/serving:latest
# Additional arguments passed to training step
VPC_NETWORK=""
METRIC_NAME=au_prc
METRIC_THRESHOLD=0.4

TASK_TYPE=training

# Service account to run the job
CUSTOM_JOB_SA=734227425472-compute@developer.gserviceaccount.com

PIPELINE_SPEC_PATH=$PIPELINE_REGISTRY/latest/training_pipeline_job.json

python -m pipelines.training_pipeline_runner \
  --project_id "$PROJECT_ID" \
  --pipeline_region $PIPELINE_REGION \
  --pipeline_root $PIPELINE_ROOT \
  --pipeline_job_spec_path $PIPELINE_SPEC_PATH \
  --data_pipeline_root $DATA_PIPELINE_ROOT \
  --training_data_schema ${DATA_SCHEMA} \
  --data_region $DATA_REGION \
  --gcs_data_output_folder $GCS_OUTPUT_PATH \
  --training_container_image_uri "$TRAIN_IMAGE_URI" \
  --output_model_file_name model.h5 \
  --machine_type n1-standard-4 \
  --serving_container_image_uri "$SERVING_IMAGE_URI" \
  --custom_job_service_account $CUSTOM_JOB_SA \
  --vpc_network "$VPC_NETWORK" \
  --metrics_name $METRIC_NAME \
  --metrics_threshold $METRIC_THRESHOLD \
  --task_type $TASK_TYPE \
  --hptune_region $PIPELINE_REGION \
  --hp_config_max_trials 30 \
  --hp_config_suggestions_per_request 5
