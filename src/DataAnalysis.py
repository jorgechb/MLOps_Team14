import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
class DataAnalysis: 
    def __init__(self, logger):
        self.logger = logger
        pass
    def EDA(self, data):
        self.logger.info("Iniciando el análisis exploratorio de datos (EDA)...")
        
        # 1 Describir el dataset
        self.describe_data(data)
        
        # 2 Verificar valores nulos y eliminar columnas con más del 20% de valores nulos
        data_cleaned = self.handle_missing_data(data)

        # 3 Verificar valores únicos en columnas categóricas
        self.unique_values(data_cleaned)

        # 4 Graficar la distribución de la variable objetivo 'class'
        self.plot_class_distribution(data_cleaned, 'class')

        # 5 Graficar boxplots y manejar valores atípicos
        data_cleaned = self.handle_outliers(data_cleaned)

        # 6 Graficar distribuciones y resumen final
        self.plot_numeric_distributions(data_cleaned)

        return data_cleaned

    def describe_data(self, data):
        #Describe la data general 
        self.logger.info("Descripción general del dataset...")
        print(data.describe())
        print(data.describe(include='object'))
        print(f"Dimensiones del dataset: {data.shape}")

        # Verifica nulos
        self.logger.info("Verificando valores nulos por columna...")
        print(data.isnull().sum() / len(data) * 100)

    def EDA(self, dataset): 
        self.logger.info("Performing Exploratory Data Analysis...")
        pass 