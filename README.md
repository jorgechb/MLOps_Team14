# MLOps_Team14

## Integrantes del Equipo 14:

- Rubén Díaz García A01371849
- Jorge Chávez Badillo A01749448
- José Manuel García Ogarrio A01795147
- Paúl Andrés Yungán Pinduisaca A01795702
- Ana Gabriela Fuentes Hernández A01383717
- David Emmanuel Villanueva Martínez A01638389

## Estructura del proyecto

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops and configuration for tools like black
│
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
│
└── mlops   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```pip install ucimlrepo
from ucimlrepo import fetch_ucirepo

# fetch dataset
secondary_mushroom = fetch_ucirepo(id=848)

# data (as pandas dataframes)
X = secondary_mushroom.data.features
y = secondary_mushroom.data.targets

# metadata
print(secondary_mushroom.metadata)

# variable information
print(secondary_mushroom.variables)
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch dataset
secondary_mushroom = fetch_ucirepo(id=848)

# Data as pandas dataframes
X = secondary_mushroom.data.features
y = secondary_mushroom.data.targets

# Convertir X y y en un DataFrame completo
data = pd.concat([X, y], axis=1)

# Ver los primeros registros del conjunto de datos
print(data.head())

# Verificar los tipos de datos y valores nulos
print(data.info())

# Eliminar filas con valores nulos
data_cleaned = data.dropna()

# Opcional: Eliminar filas donde la columna "class" tenga valores inconsistentes (por ejemplo, valores no válidos)
# Suponiendo que 'class' tiene valores válidos específicos. Reemplaza 'valores_validos' con los valores permitidos.
valores_validos = data['class'].unique()  # Obtener los valores únicos de la columna 'class'

# Filtrar filas con valores no válidos en la columna 'class'
data_cleaned = data_cleaned[data_cleaned['class'].isin(valores_validos)]

# Verificar el resultado después de la limpieza
print(data_cleaned.info())
print(data_cleaned.head())
print(data_cleaned.shape)  # Verifica las dimensiones
print(data_cleaned.head())  # Muestra las primeras filas para verificar datos
print(data.isnull().sum())  # Contar valores nulos en cada columna del DataFrame original
for col in data.columns:
    print(f'Valores únicos en {col}: {data[col].unique()}')
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Fetch dataset
secondary_mushroom = fetch_ucirepo(id=848)

# Data as pandas dataframes
X = secondary_mushroom.data.features
y = secondary_mushroom.data.targets

# Convertir X y y en un DataFrame completo
data = pd.concat([X, y], axis=1)

# Verificar valores nulos en el DataFrame original
print("Valores nulos en el DataFrame original:")
print(data.isnull().sum())

# Verificar valores únicos en cada columna
for col in data.columns:
    print(f'Valores únicos en {col}: {data[col].unique()}')

# Estrategia de limpieza alternativa: Rellenar valores nulos o eliminar filas con valores nulos solo en columnas específicas
data_cleaned = data.dropna(subset=['cap-diameter', 'stem-height', 'stem-width', 'class'])

# Comprobar dimensiones después de la limpieza
print("Dimensiones después de la limpieza:", data_cleaned.shape)

# Transformaciones necesarias para preparar los datos

# 1. Codificación de variables categóricas
# Identificar columnas categóricas
categorical_cols = data_cleaned.select_dtypes(include=['object']).columns

# Usar Label Encoding para la columna objetivo (class)
label_encoder = LabelEncoder()
data_cleaned['class'] = label_encoder.fit_transform(data_cleaned['class'])

# Aplicar One-Hot Encoding solo si hay columnas categóricas
if not categorical_cols.empty:
    data_cleaned = pd.get_dummies(data_cleaned, columns=categorical_cols.drop('class'), drop_first=True)

# 2. Normalización de características
X = data_cleaned.drop('class', axis=1)
y = data_cleaned['class']

# Verificar dimensiones de X
print("Dimensiones de X:", X.shape)

# Normalizar características usando StandardScaler solo si hay filas
if X.shape[0] > 0:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    print("No hay datos suficientes para normalizar.")

# 3. Dividir el conjunto de datos en entrenamiento y prueba
if X.shape[0] > 0:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f'Tamaño del conjunto de entrenamiento: {X_train.shape[0]}')
    print(f'Tamaño del conjunto de prueba: {X_test.shape[0]}')
else:
    print("No se pudo realizar la división en entrenamiento y prueba debido a datos insuficientes.")
