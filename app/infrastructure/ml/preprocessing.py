from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Variables categóricas del modelo.
#
# Aunque muchas están almacenadas como números en ESED,
# representan categorías y no cantidades.
CATEGORICAL_VARIABLES = [
    "codcantf",
    "careaur",
    "CTIPOBR",
    "cimi",
    "piso",
    "estru",
    "pared",
    "cubi",
    "CASAEDIF",
    "CDISPUSO"
]


# Variables numéricas reales.
NUMERIC_VARIABLES = [
    "NUPICAL",
    "CSUTE",
    "CARCO"
]


def build_preprocessor():

    # ---------------------------------------------------------
    # TRANSFORMACIÓN DE VARIABLES CATEGÓRICAS
    # ---------------------------------------------------------

    # OneHotEncoder convierte cada categoría en columnas binarias.
    #
    # Ejemplo:
    #
    # careaur
    # 1 = urbano
    # 2 = rural
    #
    # se transforma aproximadamente en:
    #
    # careaur_1 | careaur_2
    #     1           0
    #     0           1
    #
    # handle_unknown="ignore" es MUY importante porque permite
    # procesar categorías de 2025 que no hayan aparecido en 2023.
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    # ---------------------------------------------------------
    # TRANSFORMACIÓN DE VARIABLES NUMÉRICAS
    # ---------------------------------------------------------

    # StandardScaler estandariza las variables numéricas.
    #
    # Las transforma para que queden aproximadamente centradas
    # alrededor de 0 y con una escala comparable.
    #
    # Esto es especialmente útil para Regresión Lineal.
    numeric_transformer = StandardScaler()

    # ---------------------------------------------------------
    # COLUMN TRANSFORMER
    # ---------------------------------------------------------

    # ColumnTransformer permite aplicar diferentes tratamientos
    # dependiendo del tipo de variable.
    #
    # Categóricas -> OneHotEncoder
    # Numéricas   -> StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_VARIABLES
            ),
            (
                "numeric",
                numeric_transformer,
                NUMERIC_VARIABLES
            )
        ],

        # Cualquier columna que no esté declarada arriba
        # será ignorada.
        #
        # Esto evita que variables como "id" o "COSM2"
        # entren accidentalmente al modelo.
        remainder="drop"
    )

    return preprocessor