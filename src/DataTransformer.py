import yaml
import logging
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utilities import create_logger, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class DataTransformer: 
    def __init__(self, logger):
        self.config = get_config()
        self.logger = logger
        self.transformed_csv = 'data/processed/transformed'
        os.makedirs(self.transformed_csv, exist_ok=True)
        self.plot_dir = 'reports/figures'

    def transform_data(self, xtrain, ytrain, xval, yval, xtest, ytest):
        self.logger.info("Iniciando las transformaciones de los datos...")
        
        # Leer los data frames separados después de la limpieza
        xtrain = pd.read_csv(os.path.join(self.config['file_paths']['split_dataset'], 'xtrain.csv'))
        xval = pd.read_csv(os.path.join(self.config['file_paths']['split_dataset'], 'xval.csv'))
        xtest = pd.read_csv(os.path.join(self.config['file_paths']['split_dataset'], 'xtest.csv'))
        ytrain = pd.read_csv(os.path.join(self.config['file_paths']['split_dataset'], 'ytrain.csv'))
        yval = pd.read_csv(os.path.join(self.config['file_paths']['split_dataset'], 'yval.csv'))
        ytest = pd.read_csv(os.path.join(self.config['file_paths']['split_dataset'], 'ytest.csv'))

        # Separación de columnas por tipos de datos
        self.num_columns = xtrain.select_dtypes(include=np.number).columns
        self.cat_bin_columns = ['does-bruise-or-bleed','has-ring']
        self.cat_nom_columns = xtrain.select_dtypes(include='object').columns
        self.cat_nom_columns = self.cat_nom_columns.difference(self.cat_bin_columns)

        # 1 transformación de variables numéricas
        XtrainNum, XvalNum, XtestNum = self.num_transform(xtrain, xval, xtest)
        # 2 visualización de variables numéricas
        self.num_plot(XtrainNum, ytrain)
        # 3 limpieza de variables numéricas con alta correlación
        XtrainNum, XvalNum, XtestNum = self.num_clean(XtrainNum, XvalNum, XtestNum)
        # 4 transformación de variables binarias
        XtrainBin, XvalBin, XtestBin = self.bin_transform(xtrain, xval, xtest)
        # 5 tranformación de variable de salida 
        ytrainT, yvalT, ytestT = self.out_transform(ytrain, yval, ytest)
        # 6 transformación de variables categóricas nominales
        XtrainCat, XvalCat, XtestCat = self.cat_transform(xtrain, xval, xtest)
        # 7 Concatenación de variables transformadas
        XtrainT, XvalT, XtestT = self.concat(XtrainNum, XvalNum, XtestNum, XtrainBin, XvalBin, XtestBin, XtrainCat, XvalCat, XtestCat)

        # Guardar archivos en la carpeta de Data para versionar
        XtrainT.to_csv(os.path.join(self.transformed_csv, 'xtrainT.csv'), index=False)
        XvalT.to_csv(os.path.join(self.transformed_csv, 'xvalT.csv'), index=False)
        XtestT.to_csv(os.path.join(self.transformed_csv, 'xtestT.csv'), index=False)

        ytrainT.to_csv(os.path.join(self.transformed_csv, 'ytrainT.csv'), index=False)
        yvalT.to_csv(os.path.join(self.transformed_csv, 'yvalT.csv'), index=False)
        ytestT.to_csv(os.path.join(self.transformed_csv, 'ytestT.csv'), index=False)

        return XtrainT, ytrainT, XvalT, yvalT, XtestT, ytestT

    def num_transform(self, xtrain, xval, xtest):
        self.logger.info("Iniciando transformación numérica...")
        # Copiado de datasets originales para transformar
        XtrainT = xtrain[self.num_columns].copy()
        XvalT = xval[self.num_columns].copy()
        XtestT = xtest[self.num_columns].copy()
        # Inicialización de transformadores numéricos
        scaler = StandardScaler()
        yeo = PowerTransformer('yeo-johnson', standardize=False)
        # Transformación Standard
        XtrainT[self.num_columns] = yeo.fit_transform(XtrainT[self.num_columns])
        XvalT[self.num_columns] = yeo.transform(XvalT[self.num_columns])
        XtestT[self.num_columns] = yeo.transform(XtestT[self.num_columns])
        # Transformación Yeo johnson
        XtrainT[self.num_columns] = scaler.fit_transform(XtrainT[self.num_columns])
        XvalT[self.num_columns] = scaler.transform(XvalT[self.num_columns])
        XtestT[self.num_columns] = scaler.transform(XtestT[self.num_columns])
        # Visualización de Media y Desviación Estándar de los 3 conjuntos
        # self.logger.info(f'XtrainT: \nMedia: \n', XtrainT[self.num_columns].mean(),f'\nDesviación Estándar: \n', XtrainT[self.num_columns].std())
        # self.logger.info(f'\nXvalT: \nMedia: \n', XvalT[self.num_columns].mean(),f'\nDesviación Estándar: \n', XvalT[self.num_columns].std())
        # self.logger.info(f'\nXtestT: \nMedia: \n', XtestT[self.num_columns].mean(),f'\nDesviación Estándar: \n', XtestT[self.num_columns].std())
        # Regresa los 3 conjuntos ya transformados por la función 
        return XtrainT, XvalT, XtestT
    
    def num_plot(self, xtrain, ytrain):
        self.logger.info("Visualización de variables numéricas...")
        # Inicialización
        fila = 0
        columna = 0
        columnas = self.num_columns
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        plt.subplots_adjust(right=0.9, top=0.8, hspace=0.5)

        for i, columna_actual in enumerate(columnas[:3]):
            # Configuración de gráfica
            ax = axs[fila, columna]
            sns.histplot(xtrain[columna_actual], ax=ax, kde=True)
            ax.set_title(columna_actual)
            # Cálculo de media y mediana
            media = xtrain[columna_actual].mean()
            mediana = xtrain[columna_actual].median()
            # Configuración de gráfica para media y mediana
            ax.axvline(media, color='red', linestyle='--', label=f'Media: {media:.2f}')
            ax.axvline(mediana, color='green', linestyle='--', label=f'Mediana: {mediana}')
            ax.text(0.5, 0.85, f'Media: {media:.2f}', transform=ax.transAxes, color='red')
            ax.text(0.5, 0.75, f'Mediana: {mediana}', transform=ax.transAxes, color='green')
            columna += 1
            if columna == 2:
                fila += 1
                columna = 0
        axs[fila, columna].axis('off')
        plot_path = os.path.join(self.plot_dir, f'distribucion_numericas_t.png')
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Gráfico guardado en {plot_path}")
        # Matriz de correlación
        correlacion_num = pd.concat([ytrain, xtrain],axis=1).corr(numeric_only = True)
        # Crear un mapa de calor
        plt.figure(figsize=(10, 10))
        sns.heatmap(correlacion_num, annot=True, fmt=".2f")
        plt.title("Mapa de calor de correlación")
        plot_path = os.path.join(self.plot_dir, f'mapa_de_calor.png')
        plt.savefig(plot_path)
        plt.close()

    def num_clean(self, xtrain, xval, xtest):
        self.logger.info("Limpieza de variables numéricas...")
        # Copiado de datasets para transformar
        XtrainT = xtrain.copy()
        XvalT = xval.copy()
        XtestT = xtest.copy()
        # Eliminación de columnas 
        # Hay una alta correlacion entre cap diameter y stem width
        # Eliminaremos stem width para evitar multicolinealidad dentro del modelo
        XtrainT.drop(columns=['stem-width'], inplace=True)
        XvalT.drop(columns=['stem-width'], inplace=True)
        XtestT.drop(columns=['stem-width'], inplace=True)
        # Regresa los 3 conjuntos ya transformados por la función 
        return XtrainT, XvalT, XtestT

    def bin_transform(self, xtrain, xval, xtest):
        self.logger.info("Iniciando transformación binaria...")
        # Copiado de datasets originales para transformar
        XtrainT = xtrain[self.cat_bin_columns].copy()
        XvalT = xval[self.cat_bin_columns].copy()
        XtestT = xtest[self.cat_bin_columns].copy()
        # Inicialización de transformador binario
        label_encoder = LabelEncoder()
        for i in self.cat_bin_columns:
            XtrainT[i] = label_encoder.fit_transform(XtrainT[i])
            XvalT[i]   = label_encoder.transform(XvalT[i])
            XtestT[i]  = label_encoder.transform(XtestT[i])
        # Regresa los 3 conjuntos ya transformados por la función 
        return XtrainT, XvalT, XtestT
    
    def out_transform(self, ytrain, yval, ytest):
        self.logger.info("Iniciando transformación binaria para variable de salida...")
        # Copiado de datasets originales para transformar
        ytrainT = ytrain.copy()
        yvalT = yval.copy()
        ytestT = ytest.copy()
        # Formato de DataFrame
        ytrainT = pd.DataFrame(ytrainT, columns=['class'])
        yvalT   = pd.DataFrame(yvalT, columns=['class'])
        ytestT  = pd.DataFrame(ytestT, columns=['class'])
        # inicialización de transformador binario
        label_encoder = LabelEncoder()
        ytrainT['class'] = label_encoder.fit_transform(ytrainT['class'])
        yvalT['class'] = label_encoder.transform(yvalT['class'])
        ytestT['class'] = label_encoder.transform(ytestT['class'])
        # Regresa los 3 conjuntos ya transformados por la función 
        return ytrainT, yvalT, ytestT

    def cat_transform(self, xtrain, xval, xtest):
        self.logger.info("Iniciando transformación categórica nominal...")
        # Copiado de datasets para transformar
        XtrainT = xtrain[self.cat_nom_columns].copy()
        XvalT = xval[self.cat_nom_columns].copy()
        XtestT = xtest[self.cat_nom_columns].copy()
        # Inicialización de transformador categórico
        nominal_encoder =  OneHotEncoder(drop='first', sparse_output=False)
        # Seleccionar columnas categóricas sin las binarias que ya se transformaron
        nominales = self.cat_nom_columns
        # Inicialización de DataFrames
        train_NominalDf = pd.DataFrame()
        test_NominalDf  = pd.DataFrame()
        val_NominalDf   = pd.DataFrame()
        # Transforma los 3 conjuntos de datos con OneHotEncoder
        for i in nominales:
            # Train
            NominalTrain = nominal_encoder.fit_transform(XtrainT[[i]])
            NominalTrain_df = pd.DataFrame(NominalTrain)
            NominalTrain_df.columns = nominal_encoder.get_feature_names_out()
            train_NominalDf = pd.concat([train_NominalDf , NominalTrain_df],axis=1)
            # Test
            NominalTTest = nominal_encoder.transform(XtestT[[i]])
            NominalTTest_df = pd.DataFrame(NominalTTest)
            NominalTTest_df.columns = nominal_encoder.get_feature_names_out()
            test_NominalDf = pd.concat([test_NominalDf , NominalTTest_df],axis=1)
            # Validation
            NominalVal = nominal_encoder.transform(XvalT[[i]])
            NominalVal_df = pd.DataFrame(NominalVal)
            NominalVal_df.columns = nominal_encoder.get_feature_names_out()
            val_NominalDf = pd.concat([val_NominalDf , NominalVal_df],axis=1)
            # Regresa los 3 conjuntos ya transformados por la función 
            return train_NominalDf, val_NominalDf, test_NominalDf

    def concat(self, XtrainNum, XvalNum, XtestNum, XtrainBin, XvalBin, XtestBin, XtrainCat, XvalCat, XtestCat):
        self.logger.info("Concatenando conjuntos de datos transformados...")
        XtrainT = pd.concat([XtrainNum, XtrainBin, XtrainCat.set_index(XtrainNum.index)], axis=1)
        XtestT = pd.concat([XtestNum, XtestBin, XtestCat.set_index(XtestNum.index)], axis=1)
        XvalT = pd.concat([XvalNum, XvalBin, XvalCat.set_index(XvalNum.index)], axis=1)
        # Regresa los 3 conjuntos ya concatenados por la función 
        return XtrainT, XvalT, XtestT

if __name__ == '__main__':
    logger = create_logger()
    transformer = DataTransformer(logger=logger)
