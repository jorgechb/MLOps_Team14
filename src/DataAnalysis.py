import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

class DataAnalysis: 
    def __init__(self, logger):
        self.logger = logger
        self.plot_dir = 'reports/figures'
        self.explore_csv = 'data/processed/explored'
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.explore_csv, exist_ok=True)

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

        # Guardar el archivo csv para la sig fase
        explore_path = os.path.join(self.explore_csv, 'explored_dataset.csv')
        data.to_csv(explore_path, index=False)
        print(f"Dataset limpio y explorado guardado en {explore_path}")

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

    def handle_missing_data(self, data):
        # Elimina valores con mas de 20% null NAN
        self.logger.info("Eliminando columnas con más del 20 por ciento de valores nulos...")
        null_percentage = data.isnull().sum() / len(data) * 100
        columns_to_drop = null_percentage[null_percentage > 20].index.tolist()
        data_cleaned = data.drop(columns=columns_to_drop)

        print("Columnas eliminadas:")
        for column in columns_to_drop:
            print(f"- {column}: Más del 20% de valores nulos.")
        
        # Rellena valores nulos en columnas específicas
        data_cleaned[['gill-attachment', 'ring-type']] = data_cleaned[['gill-attachment', 'ring-type']].fillna('no_data')

        return data_cleaned
    
    # analisis de features categoricos 
    def unique_values(self, data):
        #Mostrar valores únicos en X categóricas
        self.logger.info("Verificando valores únicos en columnas categóricas...")
        for col in data.select_dtypes(include='object').columns:
            print(f'Valores únicos en {col}: {data[col].unique()}')
    
    # variable y
    def plot_class_distribution(self, data, target_col):
        #Graficos la distribución de la variable objetivo (clase)
        self.logger.info(f"Graficando la distribución de {target_col}...")
        class_counts = data[target_col].value_counts()
        plt.figure(figsize=(5, 3))
        class_counts.plot(kind='bar', color=['green', 'blue'])
        plt.title('Distribución de Clases')
        plt.xlabel('Clase')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=0)
        plot_path = os.path.join(self.plot_dir, f'{target_col}_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Gráfico guardado en {plot_path}")

    # manejo outliers 

    def handle_outliers(self, data):
        #Eliminar valores atípicos y graficar boxplots
        self.logger.info("Eliminando valores atípicos usando el rango intercuartílico (IQR)...")
        numeric_columns = data.select_dtypes(include=np.number).columns
        data_v2 = data.copy()

        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_range = 1.5 * IQR

            # Identificar y eliminar valores atípicos
            outliers = (data[col] < Q1 - outlier_range) | (data[col] > Q3 + outlier_range)
            outlier_percentage = (outliers.sum() / len(data)) * 100
            data_v2 = data_v2[~outliers]

            print(f"{col}: {outlier_percentage:.2f}% de valores atípicos eliminados")

        # Graficar boxplots
        self.plot_boxplots(data_v2, numeric_columns)

        return data_v2
    
    # Analisis varaibles numericas despues de outliers
    def plot_boxplots(self, data, columns):
        # Graficar boxplots para las columnas numéricas
        self.logger.info("Graficando boxplots para las columnas numéricas...")
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(columns, 1):
            plt.subplot(1, len(columns), i)
            plt.boxplot(data[col])
            plt.title(f'Box Plot de {col}')
            plt.ylabel('Valores')
        plot_path = os.path.join(self.plot_dir, f'boxplot_col_numericas.png')
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Boxplot guardado en {plot_path}")
    

    def plot_numeric_distributions(self, data):
        #Graficar distribuciones de las columnas numéricas
        self.logger.info("Graficando distribuciones de columnas numéricas...")
        numeric_columns = data.select_dtypes(include=np.number).columns
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(numeric_columns, 1):
            plt.subplot(1, len(numeric_columns), i)
            plt.hist(data[col], bins=30, edgecolor='k', alpha=0.7)
            plt.title(f'Distribución de {col}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
        plot_path = os.path.join(self.plot_dir, f'distribucion_col_numericas.png')
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Gráfico guardado en {plot_path}")
