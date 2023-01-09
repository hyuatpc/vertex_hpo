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

"""Custom serving program."""

import os
import logging

import flask
import numpy as np
import pandas as pd
from google.cloud import storage

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras.backend as K 


import pickle

import texthero as hero
from texthero import preprocessing

# pylint: disable=logging-fstring-interpolation


################################################################################
# model load code
################################################################################

# Todo: delete
#os.environ['TRAINING_DATA_SCHEMA'] = 'reviewtext:string;Class:int'
#os.environ['MODEL_FILENAME'] = 'model.h5'
#os.environ["AIP_STORAGE_URI"] = 'gs://vertex_pipeline_demo_root_hy/compute_root/aiplatform-custom-training-2022-03-01-20:41:57.648/model'


DATA_SCHEMA = os.environ['TRAINING_DATA_SCHEMA']
features = [field.split(':') for field in DATA_SCHEMA.split(';')][0:-1]
feature_names = [item[0] for item in features]
logging.info(f'feature schema: {features}')

custom_pipeline = [preprocessing.lowercase,
                preprocessing.remove_punctuation,
                preprocessing.remove_diacritics,
                preprocessing.remove_whitespace]

MODEL_FILENAME = os.environ['MODEL_FILENAME']
# Todo: this shall be an env var
TKNZ_FILENAME = 'tknz.pkl'
logging.info(f'model file name: {MODEL_FILENAME}; tokenizer file name: {TKNZ_FILENAME}')


################################################################################
# customized TF objects
################################################################################
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(y_pred) * y_true)
    possible_positives = K.sum(y_true)
    return true_positives / (possible_positives + K.epsilon())

def balanced_acc(y_true, y_pred):
    return (recall(y_true,y_pred) + recall(1-y_true,1-y_pred))/2.

def balanced_acc_diff(y_true, y_pred):
    return K.abs(recall(y_true,y_pred) - recall(1-y_true,1-y_pred))
  
################################################################################

def download_from_gcs(gcs_file_path, local_file_path):
  client = storage.Client()
  with open(local_file_path, 'wb') as f:
    client.download_blob_to_file(gcs_file_path, f)


def load_model(model_store):
  """Load TF model."""
  logging.info(f'Loading TF model {MODEL_FILENAME} from {model_store}.')
  gcs_file_path = os.path.join(model_store, MODEL_FILENAME)
  local_file_path = os.path.join('tmp', MODEL_FILENAME)
  download_from_gcs(gcs_file_path, local_file_path)
  
  model = keras.models.load_model(local_file_path, 
                                  custom_objects={'balanced_acc': balanced_acc, 
                                                  'balanced_acc_diff':balanced_acc_diff})
  
  return model


def load_tknz(model_store):
  """Load tokenizer."""
  logging.info(f'Loading TF model {TKNZ_FILENAME} from {model_store}.')
  gcs_file_path = os.path.join(model_store, TKNZ_FILENAME)
  local_file_path = os.path.join('tmp', TKNZ_FILENAME)
  download_from_gcs(gcs_file_path, local_file_path)
  
  with open(local_file_path, 'rb') as f:
    tknz = pickle.load(f)
    
  return tknz


if 'AIP_STORAGE_URI' not in os.environ:
  raise KeyError(
      'The `AIP_STORAGE_URI` environment variable has not been set. '
      'See https://cloud.google.com/ai-platform-unified/docs/predictions'
      '/custom-container-requirements#artifacts')

logging.info(f'AIP_STORAGE_URI: {os.environ["AIP_STORAGE_URI"]}')
model = load_model(os.environ['AIP_STORAGE_URI'])
tknz = load_tknz(os.environ['AIP_STORAGE_URI'])

################################################################################
# Run the inference server
################################################################################

app = flask.Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
  """For direct API calls through request."""
  maxlen = 100
  data = flask.request.get_json(force=True)
  logging.info(f'prediction: received requests containing '
               f'{len(data["instances"])} records')

  df = pd.json_normalize(data['instances'])[feature_names].squeeze(axis=1)
  logging.info(f'df is {df}')
  df = hero.clean(df, custom_pipeline)
  df = pad_sequences(tknz.texts_to_sequences(df), maxlen=maxlen)
  
  predictions = model.predict(np.expand_dims(df, -1)).flatten()

  output = [{
    'confidences': [y, 1 - y],
    'displayNames': ['1', '0']
  } for y in predictions.tolist()]

  response_dict = {
    'predictions': output
  }

  return flask.make_response(flask.jsonify(response_dict), 200)


@app.route('/health', methods=['GET', 'POST'])
def health():
  """For direct API calls through request."""
  status_code = flask.Response(status=200)
  return status_code


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  logging.info('prediction container starting up')

  port = int(os.getenv('AIP_HTTP_PORT', '8080'))
  logging.info(f'http port: {port}')

  app.run(host='0.0.0.0', port=port)
