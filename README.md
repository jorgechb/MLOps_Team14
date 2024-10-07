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
- [Presentación Ejecutiva](https://www.canva.com/design/DAGS07C2Vyc/Q0pUguD4AfYOGY2HarbBwQ/edit?utm_content=DAGS07C2Vyc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- [Video]()

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
```
