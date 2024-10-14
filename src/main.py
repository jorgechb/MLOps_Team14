from Dataset import Dataset
from Model import Model
from DataTransformer import DataTransformer
from utilities import create_logger, get_config

class Pipeline: 
    def __init__(self, logger):
        self.logger = logger
        self.config = get_config()
        self.dataset = Dataset(logger)
        self.dataset.explore()
        self.dataset.split_data()
        self.dataset.transform()

    def run(self):
        self.logger.info("Executing pipeline...") 
        model = Model(logger)
        model.train(self.dataset.xtrainT, self.dataset.ytrainT)
        predict_val = model.predict(self.dataset.xvalT)
        predict_test = model.predict(self.dataset.xtestT)

        self.logger.info("Metrics on validation set: ") 
        model.evaluate(self.dataset.yvalT, predict_val)
        self.logger.info("Metrics on test set: ") 
        model.evaluate(self.dataset.ytestT, predict_test)
        
        model.save_model()
        self.teardown() 

    def teardown(self): 

        self.logger.info("Pipeline executed successfully")
        pass 


if __name__ == "__main__": 
    
    logger =  create_logger()
    pipeline = Pipeline(logger)
    pipeline.run()
