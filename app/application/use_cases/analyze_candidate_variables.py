class AnalyzeCandidateVariablesUseCase:

    # Variables categóricas:
    # Aunque en el CSV muchas vienen almacenadas como números,
    # estos números representan códigos o categorías, no cantidades.
    #
    # Ejemplo:
    # cimi = tipo de material utilizado en los cimientos.
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
        "CORES",
        "COAMS"
    ]

    # Variables numéricas:
    # Estas sí representan cantidades sobre las que tiene sentido
    # calcular mínimo, máximo, promedio y mediana.
    NUMERIC_VARIABLES = [
        "NUPICAL",
        "CSUTE",
        "CARCO",
        "COSM2"
    ]

    # Constructor del caso de uso.
    # Recibe un repositorio encargado de acceder al dataset.
    #
    # De esta forma, esta clase no necesita saber si los datos
    # provienen de un CSV, Excel, base de datos, etc.
    def __init__(self, dataset_repository):
        self.dataset_repository = dataset_repository

    # Ejecuta el análisis exploratorio inicial
    # de las variables candidatas.
    def execute(self):

        # Carga el dataset utilizando el repositorio.
        dataset = self.dataset_repository.load()

        # Diccionario donde se almacenarán los resultados.
        # Se separan variables categóricas y numéricas
        # porque requieren análisis diferentes.
        analysis = {
            "categorical": {},
            "numeric": {}
        }

        # ---------------------------------------------------------
        # ANÁLISIS DE VARIABLES CATEGÓRICAS
        # ---------------------------------------------------------

        for column in self.CATEGORICAL_VARIABLES:

            # Obtiene todos los valores de la columna actual.
            data = dataset[column]

            analysis["categorical"][column] = {

                # Cantidad de categorías diferentes presentes.
                "unique_values": int(data.nunique()),

                # Cantidad de valores nulos (NaN).
                "null_values": int(data.isnull().sum()),

                # Cantidad de registros cuyo valor es 0.
                #
                # En ESED este valor puede representar
                # "no aplica" o "no respuesta" dependiendo
                # de la variable.
                "zero_values": int((data == 0).sum()),

                # Obtiene las 10 categorías más frecuentes.
                #
                # value_counts() cuenta cuántas veces aparece
                # cada valor.
                #
                # head(10) limita el resultado a los 10 valores
                # más frecuentes.
                "frequencies": {
                    str(key): int(value)
                    for key, value
                    in data.value_counts().head(10).items()
                }
            }

        # ---------------------------------------------------------
        # ANÁLISIS DE VARIABLES NUMÉRICAS
        # ---------------------------------------------------------

        for column in self.NUMERIC_VARIABLES:

            # Obtiene todos los valores de la columna actual.
            data = dataset[column]

            analysis["numeric"][column] = {

                # Cantidad de valores nulos.
                "null_values": int(data.isnull().sum()),

                # Cantidad de registros con valor igual a cero.
                "zero_values": int((data == 0).sum()),

                # Valor mínimo encontrado.
                "min": float(data.min()),

                # Valor máximo encontrado.
                "max": float(data.max()),

                # Promedio de los valores.
                "mean": float(data.mean()),

                # Mediana:
                # valor central de la distribución.
                # Es especialmente útil cuando existen
                # valores extremos.
                "median": float(data.median())
            }

        # Devuelve todos los resultados del análisis.
        # FastAPI posteriormente convierte este diccionario
        # automáticamente a JSON.
        return analysis