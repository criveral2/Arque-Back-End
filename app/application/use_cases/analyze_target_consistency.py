class AnalyzeTargetConsistencyUseCase:

    def __init__(self, dataset_repository):
        # Repositorio encargado de cargar la base ESED.
        self.dataset_repository = dataset_repository

    def execute(self):

        # Cargamos el dataset completo.
        dataset = self.dataset_repository.load()

        # ---------------------------------------------------------
        # IDENTIFICACIÓN DE CASOS QUE NO PUEDEN SER COMPARADOS
        # ---------------------------------------------------------

        # Si CARCO es 0, no podemos calcular CVAE / CARCO
        # porque produciría una división por cero.
        zero_carco = int((dataset["CARCO"] == 0).sum())

        # También registramos cuántos valores de CVAE y COSM2
        # son iguales a cero para analizarlos posteriormente.
        zero_cvae = int((dataset["CVAE"] == 0).sum())
        zero_cosm2 = int((dataset["COSM2"] == 0).sum())

        # ---------------------------------------------------------
        # SELECCIÓN DE REGISTROS VÁLIDOS PARA LA COMPARACIÓN
        # ---------------------------------------------------------

        # Utilizamos únicamente registros donde las tres variables
        # sean mayores que cero.
        #
        # Esto NO elimina registros del dataset original.
        # Solo crea un subconjunto temporal para analizar coherencia.
        valid_dataset = dataset[
            (dataset["CARCO"] > 0)
            & (dataset["CVAE"] > 0)
            & (dataset["COSM2"] > 0)
        ].copy()

        # ---------------------------------------------------------
        # CÁLCULO DEL COSTO POR M² DE REFERENCIA
        # ---------------------------------------------------------

        # Calculamos un indicador auxiliar:
        # valor total de la edificación / área construida.
        valid_dataset["COSM2_REFERENCE"] = (
            valid_dataset["CVAE"]
            / valid_dataset["CARCO"]
        )

        # Calculamos la diferencia porcentual absoluta entre
        # COSM2 reportado y el valor de referencia.
        valid_dataset["DIFFERENCE_PERCENT"] = (
            (
                valid_dataset["COSM2"]
                - valid_dataset["COSM2_REFERENCE"]
            ).abs()
            / valid_dataset["COSM2_REFERENCE"]
            * 100
        )

        difference = valid_dataset["DIFFERENCE_PERCENT"]

        # ---------------------------------------------------------
        # DISTRIBUCIÓN DE LAS DIFERENCIAS
        # ---------------------------------------------------------

        return {
            "total_records": int(len(dataset)),

            "valid_records_for_comparison": int(len(valid_dataset)),

            "special_cases": {
                "CARCO_zero": zero_carco,
                "CVAE_zero": zero_cvae,
                "COSM2_zero": zero_cosm2
            },

            # Estos rangos son descriptivos.
            # Todavía NO representan reglas de eliminación.
            "difference_ranges": {
                "<=5%": int((difference <= 5).sum()),

                "5%-10%": int(
                    ((difference > 5) & (difference <= 10)).sum()
                ),

                "10%-25%": int(
                    ((difference > 10) & (difference <= 25)).sum()
                ),

                "25%-50%": int(
                    ((difference > 25) & (difference <= 50)).sum()
                ),

                "50%-100%": int(
                    ((difference > 50) & (difference <= 100)).sum()
                ),

                ">100%": int((difference > 100).sum())
            },

            # Percentiles de la diferencia.
            "difference_percentiles": {
                "50": float(difference.quantile(0.50)),
                "75": float(difference.quantile(0.75)),
                "90": float(difference.quantile(0.90)),
                "95": float(difference.quantile(0.95)),
                "99": float(difference.quantile(0.99))
            }
        }