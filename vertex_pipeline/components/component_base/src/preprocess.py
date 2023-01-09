# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom component for performing data processing with BigQuery."""

from typing import Tuple
import os
import json
import jsonlines
import logging
import argparse
from datetime import datetime

from google.cloud import bigquery, storage
from kfp.v2.components import executor
from kfp.v2.dsl import Dataset, Input, Output

import texthero as hero
from texthero import preprocessing

# pylint: disable=logging-fstring-interpolation


def _bq_uri_to_fields(uri: str) -> Tuple[str, str, str]:
  uri = uri[5:]
  project, dataset, table = uri.split('.')
  return project, dataset, table


def _gs_uri_to_fields(uri: str) -> Tuple[str, str]:
  bucket = uri.replace('gs://', '').split('/')[0]
  directory = uri.replace('gs://'+bucket+'/', '')
  return bucket, directory


def upload_blob(project_id: str, bucket_name: str, source_file_name: str, destination_blob_name: str):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    
    
def collect_data(query: str, project_id: str):
  
  client = bigquery.Client(project=project_id)
  query_job = client.query(query)  # Make an API request.

  query_result = (
      query_job.result().to_dataframe(
          # Optionally, explicitly request to use the BigQuery Storage API. As of
          # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
          # API is used by default.
          create_bqstorage_client=True,
      )
  )
  
  return query_result


def preprocess_data(
    project_id: str,
    data_region: str,
    gcs_output_folder: str,
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    gcs_output_format: str = bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON,
    task_type: str = 'training'
):
  """Extract a BQ table to an output Dataset artifact.

  Args:
    project_id: The project ID.
    data_region: The region for the BQ extraction job.
    gcs_output_folder: The GCS location to store the resulting CSV file.
    input_dataset: The output artifact of the resulting dataset.
    output_dataset: The output artifact of the resulting dataset.
    gcs_output_format: The output format.

  Raises:
    RuntimeError: If the BigQuery job fails.
  """
  
  if task_type not in ('training', 'batch_prediction'):
    raise ValueError(f"task_type {task_type} is not one of 'training' and 'batch_prediction'.")
  
  # Query the data
  logging.info(f'Collecting data for task type {task_type}.')
  
  with open(os.path.join('pipelines/component/src/query', f'{task_type}_query.txt'), 'r') as f:
    query = f.read()
    
  data = collect_data(query, project_id)
  logging.info(f'Collected {len(data)} rows.')
  
  timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
  if task_type == 'training':
      # Cleaning
    logging.info('Cleaning text data.')
    custom_pipeline = [preprocessing.lowercase,
                  preprocessing.remove_punctuation,
                  preprocessing.remove_diacritics,
                  preprocessing.remove_whitespace]  
    data['reviewtext'] = hero.clean(data['reviewtext'], custom_pipeline)
    
    logging.info('Dropping dirty data.')
    data = data[data['rating'] != -1]
    data['reviewtext_word_count'] = data['reviewtext'].apply(lambda x: len(x.split()))
    data = data[(data['reviewtext_word_count'] > 0) & (data['reviewtext_word_count'] < 1000)]
    data['rating'] = (data['rating'] > 3).astype(int)
    data = data[['reviewtext', 'rating']]
    logging.info(f'Retained {len(data)} rows after cleaning.')
    source_file_name = f'processed_data-{timestamp}.csv'
    data.to_csv(source_file_name, index=False)
  else:
    source_file_name = f'processed_data-{timestamp}.jsonl'
    with jsonlines.open(source_file_name, 'w') as writer:
      writer.write_all(data.to_dict(orient='records'))
  
  bucket, directory = _gs_uri_to_fields(gcs_output_folder)
  destination_blob_name = os.path.join(directory, source_file_name)
  upload_blob(project_id, bucket, source_file_name, destination_blob_name)
  
  output_dataset.uri = os.path.join(gcs_output_folder, source_file_name)
  
  logging.info(f'Dumped {destination_blob_name} to bucket {bucket}.')


def executor_main():
  """Main executor."""

  parser = argparse.ArgumentParser()
  parser.add_argument('--executor_input', type=str)
  parser.add_argument('--function_to_execute', type=str)

  args, _ = parser.parse_known_args()
  executor_input = json.loads(args.executor_input)
  function_to_execute = globals()[args.function_to_execute]

  executor.Executor(
      executor_input=executor_input,
      function_to_execute=function_to_execute).execute()


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  executor_main()
