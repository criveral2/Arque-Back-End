# Arqué Backend

Backend de **Arqué**, una aplicación web orientada a la estimación de costos de construcción mediante modelos de aprendizaje automático.

El backend está desarrollado con **FastAPI** y será responsable de gestionar el procesamiento de datos, entrenamiento de modelos, evaluación, persistencia e inferencia de predicciones.

## Objetivo

Arqué busca proporcionar una estimación del costo por metro cuadrado de construcción a partir de características de un proyecto, utilizando modelos de aprendizaje automático entrenados con datos históricos de edificaciones del Ecuador.

La fuente principal de información utilizada para el desarrollo del modelo corresponde a las **Estadísticas de Edificaciones (ESED)** publicadas por el Instituto Nacional de Estadística y Censos del Ecuador (INEC).

## Tecnologías

- Python
- FastAPI
- Uvicorn
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Pydantic

## Modelos de aprendizaje automático

Inicialmente se evaluarán los siguientes algoritmos orientados a problemas de regresión:

- Regresión Lineal
- Random Forest Regressor
- XGBoost Regressor

Los modelos serán comparados mediante las siguientes métricas:

- MAE - Mean Absolute Error
- RMSE - Root Mean Squared Error
- R² - Coeficiente de determinación

El modelo con mejor desempeño será seleccionado para realizar las predicciones de la aplicación.

## Variable objetivo

La variable objetivo utilizada será:

`COSM2`

Correspondiente al costo estimado por metro cuadrado de construcción registrado en la base de datos ESED.

Las predicciones generadas representan el comportamiento de los valores disponibles en esta fuente estadística y no necesariamente corresponden al costo final ejecutado de una obra ni al valor vigente de mercado.

## Arquitectura

El proyecto sigue una **arquitectura hexagonal ligera**, aplicando principios de Clean Architecture y separación de responsabilidades.

La aplicación se divide principalmente en:

- API
- Casos de uso
- Dominio
- Infraestructura
- Procesamiento y entrenamiento de modelos
- Persistencia de modelos
- Resultados experimentales

## Estructura del proyecto

```text
arque-backend/
│
├── app/
│   ├── main.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── exceptions.py
│   │
│   ├── api/
│   │   ├── router.py
│   │   └── v1/
│   │       ├── routes/
│   │       │   ├── health.py
│   │       │   ├── prediction.py
│   │       │   └── model.py
│   │       │
│   │       └── schemas/
│   │           ├── prediction_request.py
│   │           ├── prediction_response.py
│   │           └── model_response.py
│   │
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── construction_project.py
│   │   │   └── prediction.py
│   │   │
│   │   └── ports/
│   │       ├── predictor.py
│   │       ├── model_repository.py
│   │       └── dataset_repository.py
│   │
│   ├── application/
│   │   └── use_cases/
│   │       ├── predict_cost.py
│   │       ├── initialize_model.py
│   │       └── train_model.py
│   │
│   └── infrastructure/
│       ├── ml/
│       │   ├── preprocessing.py
│       │   ├── algorithms.py
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   └── pipeline.py
│       │
│       ├── repositories/
│       │   ├── file_model_repository.py
│       │   └── esed_repository.py
│       │
│       └── prediction/
│           └── sklearn_predictor.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── model.joblib
│   └── metadata.json
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── experiments/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── .env
├── .gitignore
├── pyproject.toml
└── README.md
