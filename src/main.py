from Dataset import Dataset
from Model import Model
from DataTransformer import DataTransformer
from utilities import create_logger, get_config
import argparse

from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import json
import os

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

            metrics_path = "models/metrics.json"

            if os.path.exists(metrics_path):
                metrics = load_metrics(metrics_path)['Test']
                #tracking_url = mlflow.get_tracking_uri()
                print('En el if de metrics')
                print(metrics['accuracy'])
                with mlflow.start_run():
                     self.logger.info("ML_Flow...") 
                     mlflow.log_params(self.config['hyperparameters'])

                     mlflow.log_metric('accuracy', metrics['accuracy'])
                     mlflow.log_metric('precision', metrics['precision'])
                     mlflow.log_metric('recall', metrics['recall'])
                     mlflow.log_metric('f1_score', metrics['f1_score'])
            else:
                self.logger.error(f"Metrics file not found at {metrics_path}") 
        else: 
            self.logger.info("Executing complete pipeline...") 
            self.dataset.explore()
            self.dataset.split_data()
            self.dataset.transform()
            self.model.train()
            self.model.evaluate()

            metrics_path = "models/metrics.json"

            if os.path.exists(metrics_path):
                metrics = load_metrics(metrics_path)
                #tracking_url = mlflow.get_tracking_uri()

                with mlflow.start_run():
                     self.logger.info("ML_Flow...") 
                     mlflow.log_params(self.config['hyperparameters'])

                     mlflow.log_metric('accuracy', metrics['accuracy'])
                     mlflow.log_metric('precision', metrics['precision'])
                     mlflow.log_metric('recall', metrics['recall'])
                     mlflow.log_metric('f1_score', metrics['fi_score'])
            else:
                self.logger.error(f"Metrics file not found at {metrics_path}") 
            
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
