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

"""Model training program."""

from typing import Dict, Tuple, Optional, List, Iterable

import argparse
import json
import logging
import os

import yaml

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics as sk_metrics

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, BatchNormalization, Dropout, Bidirectional
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.backend import clear_session

import pickle
# pylint: disable=logging-fstring-interpolation


################################################################################
# Model serialization code
################################################################################

MODEL_FILENAME = 'model.h5'
INSTANCE_SCHEMA_FILENAME = 'instance_schema.yaml'
PROB_THRESHOLD = 0.5


def _save_tf_model(model: Model, model_store: str):
    """Export trained lgb model."""
    file_path = os.path.join(model_store, MODEL_FILENAME)
    model.save(MODEL_FILENAME)
    tf.io.gfile.copy(MODEL_FILENAME, file_path, overwrite=True)
    
def _save_training_tknz(tknz_name: str, model_store: str):
    file_path = os.path.join(model_store, tknz_name)
    tf.io.gfile.copy(tknz_name, file_path, overwrite=True)

def _save_metrics(metrics: dict, output_path: str):
    """Export the metrics of trained lgb model."""
    with tf.io.gfile.GFile(output_path, 'w') as eval_file:
        eval_file.write(json.dumps(metrics))




################################################################################
# Data loading
################################################################################

def load_csv_dataset(data_uri_pattern: str
                     ) -> pd.DataFrame:
    """Load CSV data into features and label DataFrame."""
    all_files = tf.io.gfile.glob(data_uri_pattern)

    df = pd.concat((pd.read_csv('gs://' + f) for f in all_files), ignore_index=True)

    label_distribution = df['rating'].value_counts().to_dict()
    logging.info(f'Collected {len(df)} rows from {data_uri_pattern}. Label distribution: {label_distribution}')
    logging.info(df.head(2))

    return df


################################################################################
# Model training
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
# Main Logic.
################################################################################

def batch_predict(data: pd.Series, model: Model):
  
    """The main training logic."""

    if 'AIP_MODEL_DIR' not in os.environ:
        raise KeyError(
            'The `AIP_MODEL_DIR` environment variable has not been set. '
            'See https://cloud.google.com/ai-platform-unified/docs/tutorials/'
            'image-recognition-custom/training'
        )
    output_model_directory = os.environ['AIP_MODEL_DIR']

    logging.info(f'AIP_MODEL_DIR: {output_model_directory}')
    logging.info(f'batch_data_uri: {args.training_data_uri}')
    logging.info(f'metrics_output_uri: {args.metrics_output_uri}')

    # prepare the data
    df = load_csv_dataset(data_uri_pattern=args.training_data_uri)

    # train, val, test split
    X_train, X_vt, y_train, y_vt, = train_test_split(df['reviewtext'], df['rating'], test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test, = train_test_split(X_vt, y_vt, test_size=0.5, random_state=42)
    len(y_train), y_train.mean(), len(y_val), y_val.mean(), len(y_test), y_test.mean()

    logging.info(f'Training {len(y_train)} samples with {y_train.mean()} positive ratio, validation \
  {len(y_val)} samples with {y_val.mean()} positive ratio, test {len(y_test)} samples with {y_test.mean()} positive ratio.')

    logging.info('Tokenizing words..')
    tknz = Tokenizer()
    tknz.fit_on_texts(X_train)
    # Todo: this shall be an env var
    tknz_name = 'tknz.pkl'
    with open(tknz_name, 'wb') as f:
        pickle.dump(tknz,f)
    _save_training_tknz(tknz_name, output_model_directory)

    logging.info('Padding words..')
    maxlen=100
    text_train_padded = pad_sequences(tknz.texts_to_sequences(X_train), maxlen=maxlen)
    text_val_padded = pad_sequences(tknz.texts_to_sequences(X_val), maxlen=maxlen)
    text_test_padded = pad_sequences(tknz.texts_to_sequences(X_test), maxlen=maxlen)

    logging.info('Training model..')
    model = tf_training(text_train_padded=text_train_padded,
                        text_val_padded=text_val_padded,
                        y_train=y_train,
                        y_val=y_val,
                        vocab_size=len(tknz.word_index)+1,
                        seq_len=maxlen)

    # save the generated model
    logging.info('Saving training artifacts..')
    _save_tf_model(model, output_model_directory)
    _save_analysis_schema(X_train, output_model_directory)

    # save eval metrics
    metrics = _evaluate_binary_classification(model, text_test_padded, y_test)
    if args.metrics_output_uri:
        _save_metrics(metrics, args.metrics_output_uri)
        
    logging.info('Training is successful! Congratulations!')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    # For training data
    parser.add_argument('--training_data_uri', type=str,
                        help='The training dataset location in GCS.')
    parser.add_argument('--training_data_schema', type=str, default='',
                        help='The schema of the training dataset. The'
                             'example schema: name:type;')
    parser.add_argument('--features', type=str, default='',
                        help='The column names of features to be used.')
    parser.add_argument('--label', type=str, default='',
                        help='The column name of label in the dataset.')

    parser.add_argument('--metrics_output_uri', type=str,
                        help='The GCS artifact URI to write model metrics.')
    # For model hyperparameter
    parser.add_argument('--min_data_in_leaf', default=5, type=float,
                        help='Minimum number of observations that must '
                             'fall into a tree node for it to be added.')
    parser.add_argument('--num_boost_round', default=300, type=float,
                        help='Number of boosting iterations.')
    parser.add_argument('--max_depth_hp_param_min', default=-1, type=float,
                        help='Max tree depth for base learners, <=0 means no '
                             'limit. Min value for hyperparam param')
    parser.add_argument('--max_depth_hp_param_max', default=4, type=float,
                        help='Max tree depth for base learners, <=0 means no '
                             'limit.  Max value for hyperparam param')
    parser.add_argument('--num_leaves_hp_param_min', default=6, type=float,
                        help='Maximum tree leaves for base learners. '
                             'Min value for hyperparam param.')
    parser.add_argument('--num_leaves_hp_param_max', default=10, type=float,
                        help='Maximum tree leaves for base learners. '
                             'Max value for hyperparam param.')
    # For hyperparameter tuning with Vizer
    parser.add_argument('--perform_hp', action='store_true', default=False,
                        help='Specify whether to perform hyperparameter tuning.')
    parser.add_argument('--hp_config_max_trials', default=20, type=float,
                        help='Maximum number of hyperparam tuning trials.')
    parser.add_argument('--hp_config_suggestions_per_request',
                        default=5, type=float,
                        help='Suggestions per vizier request')
    parser.add_argument('--hp_config_gcp_region', default='asia-east1', type=str,
                        help='Vizier GCP Region. Data or model no passed to '
                             'vizier. Simply tuning config.')
    parser.add_argument('--hp_config_gcp_project_id',
                        default='woven-rush-197905', type=str,
                        help='GCP project id.') 

    logging.info(parser.parse_args())
    known_args, _ = parser.parse_known_args()
    train(known_args)
