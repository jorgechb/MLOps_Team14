import pytest
import pandas as pd
import os
from Dataset import Dataset
from Model import Model
from DataAnalysis import DataAnalysis
from utilities import create_logger

logger =  create_logger()

@pytest.fixture
def pipeline():
    pl = Dataset(logger) 
    return pl

@pytest.fixture
def pipelineEDA():
    pEDA = DataAnalysis(logger)
    return pEDA

@pytest.fixture
def pipelineModel():
    pModel = Model(logger)
    return pModel

def test_read_raw_data(pipeline):
    expected_shape = (61069, 21)
    result = pipeline.raw_df
    assert result.shape[0] == expected_shape[0]

def test_EDA(pipelineEDA):
    expected_shape = (55729, 14)
    data_path = os.path.join('data', 'raw', 'raw_dataset.csv')
    result = pipelineEDA.EDA(pd.read_csv(data_path))
    assert result.shape[0] == expected_shape[0]

def test_split(pipeline):
    expected_length_train_rows = 33437
    expected_length_valtest_rows = 11146
    expected_length_columns = 13
    result1, result2 = pipeline.split_data()
    assert result1.shape[0] == expected_length_train_rows
    assert result1.shape[1] == expected_length_columns
    assert result2.shape[0] == expected_length_valtest_rows

def test_transform(pipeline):
    expected_length_trainT_rows = 33437 
    expected_length_valtestT_rows = 11146
    expected_lengthT_columns = 15
    result1, result2 = pipeline.transform()
    assert result1.shape[0] == expected_length_trainT_rows
    assert result1.shape[1] == expected_lengthT_columns
    assert result2.shape[0] == expected_length_valtestT_rows

def test_model(pipelineModel):
    expected_max_depth = 5
    expected_n_estimators = 10
    expected_random_state = 42
    result1 = pipelineModel.train()
    assert result1.max_depth == expected_max_depth
    assert result1.n_estimators == expected_n_estimators
    assert result1.random_state == expected_random_state

if __name__ == "__main__":
    pytest.main()