import os
import logging
import pandas as pd
import joblib
import json  
import pickle
from utilities import create_logger, get_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


class Model: 
    def __init__(self, logger):
        self.logger = logger
        self.config = get_config()
        self.hyperparameters = self.config['hyperparameters']
        self.model = RandomForestClassifier(**self.hyperparameters)
        self.logger.info("Model successfully created")

    def train(self):
        # Load training data 
        self.xtrain = pd.read_csv(os.path.join(self.config['file_paths']['transfomed_path'], 'xtrainT.csv'))
        self.ytrain = pd.read_csv(os.path.join(self.config['file_paths']['transfomed_path'], 'ytrainT.csv'))

        self.model.fit(self.xtrain, self.ytrain)
        self.logger.info("Training complete")

        # Save Model
        self.save_model()
        return self.model

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self):
        '''
        Loads the model created, computes accuracy, precision, recall and score
        and saves them in a json file 
        '''
        self.metrics = dict() 
        pred_model = joblib.load(self.config['file_paths']['model_path'])

        # Load test and validation data 
        self.xtest = pd.read_csv(os.path.join(self.config['file_paths']['transfomed_path'], 'xtestT.csv'))
        self.ytest = pd.read_csv(os.path.join(self.config['file_paths']['transfomed_path'], 'ytestT.csv'))
        self.xval = pd.read_csv(os.path.join(self.config['file_paths']['transfomed_path'], 'xvalT.csv'))
        self.yval = pd.read_csv(os.path.join(self.config['file_paths']['transfomed_path'], 'yvalT.csv'))

        # Get predictions 
        val_pred = pred_model.predict(self.xtest)
        test_pred = pred_model.predict(self.xval)

        # Compute metrics
        self.get_metrics(self.yval, val_pred, 'Validation')
        self.get_metrics(self.ytest, test_pred, 'Test')

        # Save metrics 
        metrics_path = os.path.join(self.config['file_paths']['metrics_path'])
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as outfile: 
            json.dump(self.metrics, outfile, indent=4)

        self.logger.info(f"Metrics can be found in {self.config['file_paths']['metrics_path']}")

    def get_metrics(self, ytrue, ypred, phase: str): 
  
        self.metrics[phase] = {'accuracy': accuracy_score(ytrue, ypred),
                               'precision': precision_score(ytrue, ypred),
                               'recall': recall_score(ytrue, ypred),
                               'f1_score': f1_score(ytrue, ypred)}
        
        # Log metrics
        self.logger.info(f'{phase} metrics: ')
        for key, value in self.metrics[phase].items(): 
            self.logger.info(f"{key}: {value}")

    def save_model(self):
        try: 
            joblib.dump(self.model, self.config['file_paths']['model_path'])
            self.logger.info("Model saved successfully")

        except: 
            self.logger.error("Unable to save the model")

    def save_model_pickle(self):
        try:
            # Asegurarse de que el archivo tenga la extensión '.pkl'
            model_path = self.config['file_paths']['model_path']
        
            # Si la ruta no termina en '.pkl', la modificamos
            if not model_path.endswith('.pkl'):
                model_path = os.path.splitext(model_path)[0] + '.pkl'

            # Guardamos el modelo con pickle
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)  # Guardamos el modelo entrenado en formato pickle

            # Imprimir y loggear la dirección donde se guardó el modelo
            print(f"Pickle Model saved successfully at: {model_path}")
            self.logger.info(f"Pickle Model saved successfully at {model_path}") 

        except Exception as e: 
            self.logger.error(f"Pickle Unable to save the model: {e}")


if __name__ == '__main__':

    logger = create_logger()
    model = Model(logger=logger)
    model.save_model()
    model.save_model_pickle()


