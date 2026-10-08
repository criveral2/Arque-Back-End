class InspectTargetOutliersUseCase:

    TARGET_VARIABLE = "COSM2"

    # Variables que nos ayudarán a entender
    # por qué un registro tiene un COSM2 muy alto.
    ANALYSIS_VARIABLES = [
        "codcantf",
        "CTIPOBR",
        "CSUTE",
        "CARCO",
        "NUPICAL",
        "cimi",
        "piso",
        "estru",
        "pared",
        "cubi",
        "CVAE",
        "COSM2"
    ]

    def __init__(self, dataset_repository):
        # Repositorio utilizado para acceder al dataset ESED.
        self.dataset_repository = dataset_repository

    def execute(self):

        # Cargamos el dataset completo.
        dataset = self.dataset_repository.load()

        # Conservamos únicamente las columnas que realmente
        # existan en el archivo.
        #
        # Esto evita errores si alguna variable cambia
        # entre distintas versiones de ESED.
        available_columns = [
            column
            for column in self.ANALYSIS_VARIABLES
            if column in dataset.columns
        ]

        # Seleccionamos únicamente las variables necesarias
        # para estudiar los valores extremos.
        analysis_dataset = dataset[available_columns].copy()

        # ---------------------------------------------------------
        # CONTROL DE COHERENCIA DEL COSTO POR M²
        # ---------------------------------------------------------
        # Calculamos un costo por m² de referencia dividiendo
        # el valor total calculado de la edificación (CVAE)
        # para el área total a construir (CARCO).
        #
        # IMPORTANTE:
        # Este valor NO reemplaza a COSM2.
        # Únicamente se utiliza como indicador auxiliar para
        # detectar posibles inconsistencias en los datos.
        analysis_dataset["COSM2_REFERENCE"] = (
                analysis_dataset["CVAE"]
                / analysis_dataset["CARCO"]
        )

        # Calculamos la diferencia porcentual entre el COSM2
        # reportado y el valor de referencia.
        #
        # abs() permite obtener la magnitud de la diferencia
        # independientemente de si el valor es mayor o menor.
        analysis_dataset["COSM2_DIFFERENCE_PERCENT"] = (
                (
                        analysis_dataset["COSM2"]
                        - analysis_dataset["COSM2_REFERENCE"]
                ).abs()
                / analysis_dataset["COSM2_REFERENCE"]
                * 100
        )

        # Ordenamos los registros desde el COSM2 más alto
        # hasta el más bajo.
        highest_values = (
            analysis_dataset
            .sort_values(
                by=self.TARGET_VARIABLE,
                ascending=False
            )
            .head(30)
        )

        # Convertimos los registros a diccionarios normales
        # para que FastAPI pueda devolverlos como JSON.
        records = highest_values.to_dict(orient="records")

        return {
            "records_analyzed": len(records),

            # Registros con los 30 valores más altos de COSM2.
            "highest_cosm2_records": records
        }

       