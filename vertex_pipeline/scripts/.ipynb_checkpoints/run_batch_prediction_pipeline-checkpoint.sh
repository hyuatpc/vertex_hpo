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
PIPELINE_REGION=us-central1

PIPELINE_ROOT=gs://vertex_pipeline_demo_root_hy/pipeline_root # The GCS path for storing artifacts of pipeline runs
DATA_PIPELINE_ROOT=gs://vertex_pipeline_demo_root_hy/compute_root # The GCS staging location for custom job
GCS_OUTPUT_PATH=gs://vertex_pipeline_demo_root_hy/datasets/prediction # The GCS path for storing processed data
GCS_PREDICTION_PATH=gs://vertex_pipeline_demo_root_hy/prediction # The GCS path for storing prediction results
PIPELINE_REGISTRY=gs://vertex_pipeline_demo_root_hy/pipeline_spec

# Setup for dataset
DATA_REGION=us-central1

# The endpoint resource name that hosting the target model
# You may also use target model resource name directly, in the case please use
# --model_resource_name $MODEL_RN
MODEL_RN=projects/734227425472/locations/us-central1/models/6878091744576012288
TASK_TYPE=batch_prediction

#PIPELINE_SPEC_PATH=./pipeline_spec/batch_prediction_pipeline_job.json
PIPELINE_SPEC_PATH=$PIPELINE_REGISTRY/latest/batch_prediction_pipeline_job.json

python -m pipelines.batch_prediction_pipeline_runner \
  --project_id "$PROJECT_ID" \
  --pipeline_region $PIPELINE_REGION \
  --pipeline_root $PIPELINE_ROOT \
  --pipeline_job_spec_path $PIPELINE_SPEC_PATH \
  --data_pipeline_root $DATA_PIPELINE_ROOT \
  --data_region $DATA_REGION \
  --gcs_data_output_folder $GCS_OUTPUT_PATH \
  --gcs_result_folder $GCS_PREDICTION_PATH \
  --model_resource_name $MODEL_RN \
  --task_type $TASK_TYPE \
  --machine_type n1-standard-8 \
  --accelerator_count 0 \
  --accelerator_type ACCELERATOR_TYPE_UNSPECIFIED \
  --starting_replica_count 1 \
  --max_replica_count 2
