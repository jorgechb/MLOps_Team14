import yaml 
import os
import logging
import joblib
import json  
import numpy as np
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

    def train(self, xtrain, ytrain): 
        self.model.fit(xtrain, ytrain)
        self.logger.info("Training complete")

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, ytrue, ypred):
        self.metrics = dict() 

        self.metrics['accuracy'] = accuracy_score(ytrue, ypred)
        self.metrics['precision'] = precision_score(ytrue, ypred)
        self.metrics['recall'] = recall_score(ytrue, ypred)
        self.metrics['fi_score'] = f1_score(ytrue, ypred)

        for key, value in self.metrics.items(): 
            self.logger.info(f"{key}: {value}")

        with open("metrics.json", 'w') as outfile: 
            json.dump(self.metrics, outfile)

        self.logger.info("Metrics can be found in ...")

    def save_model(self):
        try: 
            joblib.dump(self.model, self.config['file_paths']['model_path'])
            self.logger.info("Model saved successfully") 
        except: 
            self.logger.error("Unable to save the model")


if __name__ == '__main__':

    logger = create_logger()
    model = Model(logger=logger)
    model.save_model()


