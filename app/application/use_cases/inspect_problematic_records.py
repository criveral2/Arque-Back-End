class InspectProblematicRecordsUseCase:

    def __init__(self, dataset_repository):
        # Repositorio encargado de cargar la base ESED.
        self.dataset_repository = dataset_repository

    def execute(self):

        # Cargamos el dataset original.
        dataset = self.dataset_repository.load()

        # ---------------------------------------------------------
        # GRUPO A: COSM2 IGUAL A CERO
        # ---------------------------------------------------------

        # Estos registros no proporcionan una etiqueta útil
        # para entrenar un modelo cuyo objetivo es predecir COSM2.
        zero_target = dataset[
            dataset["COSM2"] == 0
        ].copy()

        # ---------------------------------------------------------
        # GRUPO B: REGISTROS COMPARABLES
        # ---------------------------------------------------------

        # Para calcular CVAE / CARCO necesitamos que:
        # - CARCO sea mayor que cero
        # - CVAE sea mayor que cero
        # - COSM2 sea mayor que cero
        valid_dataset = dataset[
            (dataset["CARCO"] > 0)
            & (dataset["CVAE"] > 0)
            & (dataset["COSM2"] > 0)
        ].copy()

        # Calculamos un costo por m² auxiliar.
        #
        # Se utiliza únicamente como control de coherencia,
        # NO como variable predictora.
        valid_dataset["COSM2_REFERENCE"] = (
            valid_dataset["CVAE"]
            / valid_dataset["CARCO"]
        )

        # Diferencia porcentual entre COSM2 reportado
        # y el indicador auxiliar.
        valid_dataset["DIFFERENCE_PERCENT"] = (
            (
                valid_dataset["COSM2"]
                - valid_dataset["COSM2_REFERENCE"]
            ).abs()
            / valid_dataset["COSM2_REFERENCE"]
            * 100
        )

        # Identificamos registros cuya diferencia
        # supera el 100 %.
        high_difference = valid_dataset[
            valid_dataset["DIFFERENCE_PERCENT"] > 100
        ].copy()

        # ---------------------------------------------------------
        # VARIABLES QUE QUEREMOS OBSERVAR
        # ---------------------------------------------------------

        columns = [
            "id",
            "codcantf",
            "CTIPOBR",
            "CSUTE",
            "CARCO",
            "NUPICAL",
            "CVAE",
            "COSM2"
        ]

        # Conservamos solo las columnas que realmente
        # existan en el dataset.
        available_columns = [
            column
            for column in columns
            if column in dataset.columns
        ]

        # Para el segundo grupo también queremos mostrar
        # los valores calculados.
        difference_columns = (
            available_columns
            + [
                "COSM2_REFERENCE",
                "DIFFERENCE_PERCENT"
            ]
        )

        return {

            # Cantidad total de casos detectados.
            "summary": {
                "COSM2_zero": int(len(zero_target)),
                "difference_over_100": int(len(high_difference))
            },

            # Mostramos solamente los primeros 30 registros
            # para no devolver cientos de filas en Swagger.
            "COSM2_zero_examples": (
                zero_target[available_columns]
                .head(30)
                .to_dict(orient="records")
            ),

            # Ordenamos los casos más inconsistentes primero.
            "difference_over_100_examples": (
                high_difference[difference_columns]
                .sort_values(
                    by="DIFFERENCE_PERCENT",
                    ascending=False
                )
                .head(30)
                .to_dict(orient="records")
            )
        }