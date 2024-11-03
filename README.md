# MLOps_Team14

## Integrantes del Equipo 14:

- Rubén Díaz García A01371849
- Jorge Chávez Badillo A01749448
- José Manuel García Ogarrio A01795147
- Paúl Andrés Yungán Pinduisaca A01795702
- Ana Gabriela Fuentes Hernández A01383717
- David Emmanuel Villanueva Martínez A01638389

### Links Importantes:

- [Repositorio Github](https://github.com/jorgechb/MLOps_Team14)
- [Presentación Ejecutiva](https://www.canva.com/design/DAGTaL2Y4dc/GJ4gBQqx5Rr6QvJZ_YBj2Q/edit?utm_content=DAGTaL2Y4dc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- [Video]()

## Abstract del proyecto:

Este proyecto de clasificación de hongos como venenosos o no venenosos utiliza técnicas de Machine Learning en un entorno de MLOps. Incluye un pipeline completo de limpieza de datos, gestión de outliers y análisis exploratorio de datos (EDA), además de transformaciones para mejorar la precisión del modelo. Con roles claramente definidos, como SME, Data Scientist, Software Engineers, DevOps, Model Risk Managers y ML Architects, se ha implementado una metodología MLOps que asegura un flujo de trabajo automatizado, eficiente y escalable. Este enfoque colaborativo garantiza la calidad, el despliegue continuo y la alineación con los objetivos de negocio, promoviendo la reproducibilidad y la gestión de riesgos en el modelado de datos.

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
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── DataAnalysis.py         <- Analysis module
    │
    ├── Dataset.py              <- Creates dataset and gets it ready for ML
    │
    ├── DataTransformer.py      <- Data Transformations module
    │
    ├── Model.py                <- ML algorithm wrapper
    |
    ├── DataLoader.py           <- Fetch raw data
    │
    └── utilities.py            <- Common utilities between scripts
```
# # Gobernanza del Proyecto
Este proyecto implementa técnicas de gobernanza para asegurar calidad, seguridad, y cumplimiento con estándares éticos y legales par el problema principal.

## Políticas de Gobernanza
1. Model Governance
Documentación precisa del proyecto y modelo: 
Establecemos directrices para la creación, entrenamiento y despliegue de modelos, asegurando precisión y transparencia en los resultados.
Este proceso incluye validaciones y revisiones que mitigan riesgos de sesgo y optimizan el rendimiento del modelo.

2. Estándares de Código

- Para mejorar la mantenibilidad y colaboración, el código sigue estándares de estilo, convenciones de nombres y patrones de diseño que facilitan su comprensión y prevención de errores.
Verificaciones Éticas y de Riesgo

- El proyecto incluye evaluaciones de impacto ético y de riesgo, implementando medidas para reducir riesgos asociados, tanto en la privacidad de datos como en el potencial de sesgos en los modelos.
Cumplimiento Legal

- Se documentan procedimientos para asegurar el cumplimiento de normativas locales e internacionales aplicables, con especial énfasis en la privacidad de datos.
Documentación de Gobernanza

## Las políticas y procedimientos de gobernanza implementados ayudan a asegurar:

Calidad: Con procedimientos de revisión y pruebas que mantienen altos estándares en el proyecto.
Seguridad: Con controles de protección de datos y control de accesos que protegen la integridad del proyecto.
Cumplimiento: Garantizando conformidad con regulaciones legales y normativas éticas.
Estas políticas consolidan un marco de gobernanza robusto, contribuyendo al desarrollo responsable del proyecto.
