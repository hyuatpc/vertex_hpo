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

"""Define the training pipeline on Vertex AI Pipeline."""

import os
import argparse
from functools import partial

import yaml
import jinja2
import kfp
from kfp.v2 import dsl
from kfp.v2.compiler import compiler
from kfp.v2.dsl import Dataset


def _load_custom_component(project_id: str,
                           af_registry_location: str,
                           af_registry_name: str,
                           components_dir: str,
                           component_name: str):
  """Load custom Vertex AI Pipeline component."""
  component_path = os.path.join(components_dir,
                                component_name,
                                'component.yaml.jinja')
  with open(component_path, 'r', encoding='utf-8') as f:
    component_text = jinja2.Template(f.read()).render(
      project_id=project_id,
      af_registry_location=af_registry_location,
      af_registry_name=af_registry_name)

  return kfp.components.load_component_from_text(component_text)


def create_training_pipeline(project_id: str,
                             pipeline_name: str,
                             af_registry_location: str,
                             af_registry_name: str,
                             components_dir: str,
                             pipeline_job_spec_path: str):
  """Creat training pipeline."""
  load_custom_component = partial(_load_custom_component,
                                  project_id=project_id,
                                  af_registry_location=af_registry_location,
                                  af_registry_name=af_registry_name,
                                  components_dir=components_dir)

  preprocess_op = load_custom_component(component_name='data_preprocess')
  train_op = load_custom_component(component_name='train_model')
  check_metrics_op = load_custom_component(component_name='check_model_metrics')

  @dsl.pipeline(name=pipeline_name)
  def pipeline(project_id: str,
               data_region: str,
               gcs_data_output_folder: str,
               training_data_schema: str,
               data_pipeline_root: str,
               training_container_image_uri: str,
               serving_container_image_uri: str,
               custom_job_service_account: str,
               metrics_name: str,
               metrics_threshold: float,
               hptune_region: str,
               output_model_file_name: str = 'model.h5',
               machine_type: str = 'n1-standard-8',
               accelerator_count: int = 0,
               accelerator_type: str = 'ACCELERATOR_TYPE_UNSPECIFIED',
               vpc_network: str = '', 
               task_type: str = 'training',
               hp_config_max_trials: int = 30,
               hp_config_suggestions_per_request: int = 5):

    # pylint: disable=not-callable
    preprocess_task = preprocess_op(
      project_id=project_id,
      data_region=data_region,
      gcs_output_folder=gcs_data_output_folder,
      gcs_output_format='CSV',
      task_type=task_type)

    train_task = train_op(
      project_id=project_id,
      data_region=data_region,
      data_pipeline_root=data_pipeline_root,
      input_data_schema=training_data_schema,
      training_container_image_uri=training_container_image_uri,
      serving_container_image_uri=serving_container_image_uri,
      custom_job_service_account=custom_job_service_account,
      input_dataset=preprocess_task.outputs['output_dataset'],
      output_model_file_name=output_model_file_name,
      machine_type=machine_type,
      accelerator_count=accelerator_count,
      accelerator_type=accelerator_type,
      hptune_region=hptune_region,
      hp_config_max_trials=hp_config_max_trials,
      hp_config_suggestions_per_request=hp_config_suggestions_per_request,
      vpc_network=vpc_network)

    check_metrics_task = check_metrics_op(
      metrics_name=metrics_name,
      metrics_threshold=metrics_threshold,
      basic_metrics=train_task.outputs['basic_metrics'])
      

  compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=pipeline_job_spec_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str,
                      help='The config file for setting default values.')

  args = parser.parse_args()

  with open(args.config, 'r', encoding='utf-8') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    create_training_pipeline(
      project_id=config['gcp']['project_id'],
      af_registry_location=config['gcp']['af_registry_location'],
      af_registry_name=config['gcp']['af_registry_name'],
      pipeline_name=config['train']['name'],
      components_dir=config['pipelines']['pipeline_component_directory'],
      pipeline_job_spec_path=config['train']['pipeline_job_spec_path'])
