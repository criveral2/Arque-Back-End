from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from app.domain.model_features import (
    CATEGORICAL_VARIABLES,
    NUMERIC_VARIABLES
)


def build_preprocessor():

    # ---------------------------------------------------------
    # VARIABLES CATEGÓRICAS
    # ---------------------------------------------------------

    # Pipeline para variables categóricas.
    #
    # Primero reemplazamos posibles valores faltantes
    # utilizando la categoría más frecuente.
    #
    # Después convertimos las categorías mediante
    # One-Hot Encoding.
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # ---------------------------------------------------------
    # VARIABLES NUMÉRICAS
    # ---------------------------------------------------------

    # Pipeline para variables numéricas.
    #
    # Los NaN se reemplazan utilizando la mediana
    # calculada únicamente con el conjunto de entrenamiento.
    #
    # Luego las variables se estandarizan.
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # ---------------------------------------------------------
    # PREPROCESAMIENTO GENERAL
    # ---------------------------------------------------------

    # Aplicamos un tratamiento diferente dependiendo
    # del tipo de variable.
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

        # Cualquier columna no declarada se ignora.
        remainder="drop"
    )

    return preprocessor