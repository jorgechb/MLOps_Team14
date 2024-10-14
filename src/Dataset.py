import pandas as pd
from utilities import get_config, create_logger
from DataAnalysis import DataAnalysis
from DataTransformer import DataTransformer
from sklearn.model_selection import train_test_split


class Dataset: 
    def __init__(self, logger):
        self.config = get_config()
        self.logger = logger
        self.raw_df = pd.read_csv(self.config['file_paths']['raw_dataset'])
        self.xtrain = []
        self.ytrain = []
        self.xval = [] 
        self.yval = []
        self.xtest = []
        self.ytest = []

    def explore(self): 
        analyzer = DataAnalysis(logger=self.logger)
        self.raw_df = analyzer.EDA(self.raw_df)

    def split_data(self):
        self.logger.info("Splitting data...")

        X = self.raw_df.drop('class', axis=1)
        y = self.raw_df['class']
        
        xtrain, xtv, ytrain, ytv = train_test_split(X, y, train_size=0.6, shuffle=True, random_state=5, stratify=y)
        xval, xtest, yval, ytest = train_test_split(xtv, ytv, test_size=0.5, shuffle=True, random_state=7, stratify=ytv)

        self.xtrain = xtrain 
        self.ytrain = ytrain 
        self.xval = xval 
        self.yval = yval
        self.xtest = xtest
        self.ytest = ytest 

        self.logger.info(f'Data partition:')
        self.logger.info(f'Train -> {self.xtrain.shape}')
        self.logger.info(f'Test -> {self.xtest.shape}')
        self.logger.info(f'Validation -> {self.xval.shape}')

    
    def transform(self):
        transformer = DataTransformer(logger=self.logger)
        self.xtrainT, self.ytrainT, self.xvalT, self.yvalT, self.xtestT, self.ytestT = transformer.transform_data(self.xtrain, self.ytrain, self.xval, self.yval, self.xtest, self.ytest) 
        


     
if __name__ == "__main__": 
    logger = create_logger()
    dataset = Dataset(logger=logger)
    dataset.explore()