# =========================================================
# CONFIGURACIÓN DE VARIABLES DEL MODELO
# =========================================================
#
# Este archivo define qué información utilizará Arqué
# para intentar predecir COSM2.
#
# Se mantiene en domain porque estas variables forman parte
# del concepto del modelo y no dependen de FastAPI, Pandas,
# scikit-learn ni de la forma física de almacenamiento.


# ---------------------------------------------------------
# VARIABLES DEL PRIMER EXPERIMENTO
# ---------------------------------------------------------

BASE_CATEGORICAL_VARIABLES = [
    "codcantf",   # Cantón
    "careaur",    # Área urbana / rural
    "CTIPOBR",    # Tipo de obra
    "cimi",       # Material de cimientos
    "piso",       # Material de piso
    "estru",      # Material de estructura
    "pared",      # Material de paredes
    "cubi",       # Material de cubierta
    "CASAEDIF",   # Casa / edificio
    "CDISPUSO"    # Uso de la edificación
]


BASE_NUMERIC_VARIABLES = [
    "NUPICAL",    # Número de pisos
    "CSUTE",      # Superficie del terreno
    "CARCO"       # Área total a construir
]


# ---------------------------------------------------------
# VARIABLES INCORPORADAS EN EL SEGUNDO EXPERIMENTO
# ---------------------------------------------------------

ADDITIONAL_CATEGORICAL_VARIABLES = [
    "mes",        # Mes del permiso
    "TOCUP",      # Tipo de ocupación
    "VALTERR"     # Rango de valor catastral del terreno
]


ADDITIONAL_NUMERIC_VARIABLES = [
    "NUCUCAL",    # Número de cuartos por vivienda
    "NUDOCAL"     # Número de dormitorios por vivienda
]


# ---------------------------------------------------------
# CONJUNTO ACTUAL DE VARIABLES
# ---------------------------------------------------------

CATEGORICAL_VARIABLES = (
    BASE_CATEGORICAL_VARIABLES
    + ADDITIONAL_CATEGORICAL_VARIABLES
)


NUMERIC_VARIABLES = (
    BASE_NUMERIC_VARIABLES
    + ADDITIONAL_NUMERIC_VARIABLES
)


# Variable que intentamos predecir.
TARGET_VARIABLE = "COSM2"


# Nombre que nos permite identificar esta versión
# del conjunto de características en los experimentos.
FEATURE_SET_NAME = "extended_v2"