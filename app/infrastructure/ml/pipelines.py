from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

from app.infrastructure.ml.preprocessing import build_preprocessor


def build_linear_regression_pipeline():

    # ---------------------------------------------------------
    # PREPROCESADOR
    # ---------------------------------------------------------

    # Creamos el ColumnTransformer definido anteriormente.
    #
    # Este se encargará de:
    # - aplicar OneHotEncoder a las variables categóricas
    # - aplicar StandardScaler a las variables numéricas
    preprocessor = build_preprocessor()

    # ---------------------------------------------------------
    # PIPELINE
    # ---------------------------------------------------------

    # Pipeline ejecutará los pasos en el orden indicado:
    #
    # 1. Preprocesar X
    # 2. Entrenar el modelo de Regresión Lineal
    #
    # Cuando posteriormente llamemos predict(),
    # también aplicará automáticamente el mismo
    # preprocesamiento antes de realizar la predicción.
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )

    return pipeline

def build_random_forest_pipeline():

    # ---------------------------------------------------------
    # PREPROCESADOR
    # ---------------------------------------------------------

    # Utilizamos exactamente el mismo preprocesamiento
    # definido para los demás modelos.
    #
    # Esto garantiza que todos los algoritmos reciban
    # las mismas variables bajo las mismas condiciones.
    preprocessor = build_preprocessor()

    # ---------------------------------------------------------
    # RANDOM FOREST
    # ---------------------------------------------------------

    # Random Forest crea varios árboles de decisión
    # y combina sus resultados para realizar la predicción.
    #
    # n_estimators=200:
    # se construirán 200 árboles.
    #
    # random_state=42:
    # permite reproducir exactamente el experimento.
    #
    # n_jobs=-1:
    # utiliza todos los núcleos disponibles del procesador
    # para acelerar el entrenamiento.
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # ---------------------------------------------------------
    # PIPELINE
    # ---------------------------------------------------------

    # Primero se procesan las variables.
    # Después se entrena Random Forest.
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline

def build_xgboost_pipeline():

    # ---------------------------------------------------------
    # PREPROCESADOR
    # ---------------------------------------------------------

    # Reutilizamos exactamente el mismo preprocesamiento
    # utilizado por Regresión Lineal y Random Forest.
    #
    # De esta manera todos los modelos trabajan con
    # las mismas variables y transformaciones.
    preprocessor = build_preprocessor()

    # ---------------------------------------------------------
    # XGBOOST
    # ---------------------------------------------------------

    # XGBoost construye árboles de forma secuencial.
    #
    # Cada nuevo árbol intenta corregir los errores
    # cometidos por los árboles anteriores.
    #
    # Esta es una configuración inicial.
    # Todavía NO estamos optimizando hiperparámetros.
    model = XGBRegressor(

        # Cantidad de árboles que se construirán.
        n_estimators=200,

        # Controla cuánto aporta cada árbol nuevo.
        # Un valor relativamente pequeño permite
        # un aprendizaje más gradual.
        learning_rate=0.05,

        # Profundidad máxima de cada árbol.
        max_depth=6,

        # Utilizamos el 80 % de los registros en cada árbol.
        # Puede ayudar a reducir sobreajuste.
        subsample=0.8,

        # Cada árbol utiliza el 80 % de las características.
        colsample_bytree=0.8,

        # Función objetivo para problemas de regresión.
        objective="reg:squarederror",

        # Algoritmo eficiente para construcción de árboles.
        tree_method="hist",

        # Garantiza reproducibilidad.
        random_state=42,

        # Utiliza todos los núcleos disponibles.
        n_jobs=-1
    )

    # ---------------------------------------------------------
    # PIPELINE
    # ---------------------------------------------------------

    # Primero se transforman los datos y posteriormente
    # XGBoost aprende a predecir COSM2.
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline