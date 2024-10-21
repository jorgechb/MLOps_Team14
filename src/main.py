from Dataset import Dataset
from Model import Model
from DataTransformer import DataTransformer
from utilities import create_logger, get_config
import argparse

from urllib.parse import urlparse

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import json
import os
import pandas as pd 

mlflow.set_tracking_uri("http://127.0.0.1:5000/") 
mlflow.set_experiment("Equipo14_MLOps")

def load_metrics(file_path):
    """Leer los archivos o resultados del JSON del test."""
    with open(file_path, 'r') as file:
        metrics = json.load(file)
    return metrics

class Pipeline: 
    def __init__(self, logger, phase: str):
        self.logger = logger
        self.config = get_config()
        self.phase = phase
        self.dataset = Dataset(logger)
        self.model = Model(logger)

    def run(self):
        # Select which phase to execute
        if self.phase == 'explore':
            self.logger.info("Executing explore data phase...") 
            self.dataset.explore()
        elif self.phase == 'split_data': 
            self.logger.info("Executing split data phase...") 
            self.dataset.split_data()
        elif self.phase == 'transform': 
            self.logger.info("Executing transform phase...") 
            self.dataset.transform()
        elif self.phase == 'train': 
            self.logger.info("Executing train phase...") 
            self.model.train()
        elif self.phase == 'evaluate': 
            self.logger.info("Executing evaluate phase...") 
            self.model.evaluate()

            current_dir = os.getcwd()

            metrics_path = os.path.join(current_dir, "models", "metrics.json")
            model_path = os.path.join(current_dir, "models", "model.joblib")

            example_input = pd.read_csv(os.path.join(self.config['file_paths']['transfomed_path'], 'xtrainT.csv')).tail(1)
            example_input.fillna(0, inplace=True)  # Fill missing values

            if os.path.exists(metrics_path) and os.path.exists(model_path):

                metrics = load_metrics(metrics_path)
                tracking_url = mlflow.get_tracking_uri()

                with mlflow.start_run() as run:

                    mlflow.log_params(self.config['hyperparameters'])

                    # Registra las métricas de Validación
                    mlflow.log_metric('val_accuracy', metrics['Validation']['accuracy'])
                    mlflow.log_metric('val_precision', metrics['Validation']['precision'])
                    mlflow.log_metric('val_recall', metrics['Validation']['recall'])
                    mlflow.log_metric('val_f1_score', metrics['Validation']['f1_score'])

                    # Registra las métricas de Test
                    mlflow.log_metric('test_accuracy', metrics['Test']['accuracy'])
                    mlflow.log_metric('test_precision', metrics['Test']['precision'])
                    mlflow.log_metric('test_recall', metrics['Test']['recall'])
                    mlflow.log_metric('test_f1_score', metrics['Test']['f1_score'])

                    model = joblib.load(model_path)
                    signature = infer_signature(example_input, model.predict(example_input))

                    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=example_input)
            
            else:
                if not os.path.exists(metrics_path):
                    self.logger.error(f"Metrics file not found at {metrics_path}")
                
                if not os.path.exists(model_path):
                    self.logger.error(f"Model file not found at {model_path}")    

        else: 
            self.logger.info("Executing complete pipeline...") 
            self.dataset.explore()
            self.dataset.split_data()
            self.dataset.transform()
            self.model.train()
            self.model.evaluate()

            current_dir = os.getcwd()

            metrics_path = os.path.join(current_dir, "models", "metrics.json")
            model_path = os.path.join(current_dir, "models", "model.joblib")

            example_input = pd.read_csv(os.path.join(self.config['file_paths']['transfomed_path'], 'xtrainT.csv')).tail(1)
            example_input.fillna(0, inplace=True)  # Fill missing values

            if os.path.exists(metrics_path) and os.path.exists(model_path):

                metrics = load_metrics(metrics_path)
                tracking_url = mlflow.get_tracking_uri()

                with mlflow.start_run() as run:

                    mlflow.log_params(self.config['hyperparameters'])

                    # Registra las métricas de Validación
                    mlflow.log_metric('val_accuracy', metrics['Validation']['accuracy'])
                    mlflow.log_metric('val_precision', metrics['Validation']['precision'])
                    mlflow.log_metric('val_recall', metrics['Validation']['recall'])
                    mlflow.log_metric('val_f1_score', metrics['Validation']['f1_score'])

                    # Registra las métricas de Test
                    mlflow.log_metric('test_accuracy', metrics['Test']['accuracy'])
                    mlflow.log_metric('test_precision', metrics['Test']['precision'])
                    mlflow.log_metric('test_recall', metrics['Test']['recall'])
                    mlflow.log_metric('test_f1_score', metrics['Test']['f1_score'])

                    model = joblib.load(model_path)
                    signature = infer_signature(example_input, model.predict(example_input))

                    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=example_input)
            
            else:
                if not os.path.exists(metrics_path):
                    self.logger.error(f"Metrics file not found at {metrics_path}")
                
                if not os.path.exists(model_path):
                    self.logger.error(f"Model file not found at {model_path}")

        self.teardown()
        

    def teardown(self): 
        
        if self.phase == 'all': 
            self.logger.info("Pipeline executed successfully")
        else: 
            self.logger.info("Phase completed")
    

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phase', type=str, help='Phase of the pipeline to be implemented', 
                        default='all', choices=['explore', 'split_data', 'transform', 'train', 'evaluate'])
    args = parser.parse_args()
    phase = args.phase
    
    logger =  create_logger()
    pipeline = Pipeline(logger, phase)
    pipeline.run()
