from Dataset import Dataset
from Model import Model
from DataTransformer import DataTransformer
from utilities import create_logger, get_config
import argparse

class Pipeline: 
    def __init__(self, logger, phase: str):
        self.logger = logger
        self.config = get_config()
        self.phase = phase
        self.dataset = Dataset(logger)
        

    def run(self):

        # Select which phase to execute
        if self.phase == 'explore':
            self.dataset.explore()
        elif self.phase == 'split_data': 
            self.logger.info("Executing split data phase...") 
            self.dataset.split_data()
        elif self.phase == 'transform': 
            # self.dataset.transform()
            self.logger.info("Executing transform phase...") 
        elif self.phase == 'train': 
            self.logger.info("Executing train phase...") 
            # model = Model(logger)
            # model.train(self.dataset.xtrainT, self.dataset.ytrainT)
        elif self.phase == 'evaluate': 
            self.logger.info("Executing evaluate phase...") 
            # predict_val = model.predict(self.dataset.xvalT)
            # predict_test = model.predict(self.dataset.xtestT)

            # self.logger.info("Metrics on validation set: ") 
            # model.evaluate(self.dataset.yvalT, predict_val)
            # self.logger.info("Metrics on test set: ") 
            # model.evaluate(self.dataset.ytestT, predict_test)
        else: 
            self.logger.info("Executing complete pipeline...") 
            # model = Model(logger)
            # model.train(self.dataset.xtrainT, self.dataset.ytrainT)
            # predict_val = model.predict(self.dataset.xvalT)
            # predict_test = model.predict(self.dataset.xtestT)

            # self.logger.info("Metrics on validation set: ") 
            # model.evaluate(self.dataset.yvalT, predict_val)
            # self.logger.info("Metrics on test set: ") 
            # model.evaluate(self.dataset.ytestT, predict_test)
            
            # model.save_model()
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
