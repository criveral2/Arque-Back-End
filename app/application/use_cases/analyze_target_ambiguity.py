from app.infrastructure.ml.preprocessing import (
    CATEGORICAL_VARIABLES,
    NUMERIC_VARIABLES
)


class AnalyzeTargetAmbiguityUseCase:

    TARGET_VARIABLE = "COSM2"

    def __init__(self, dataset_repository):

        # Repositorio encargado de cargar
        # ESED 2023 previamente preparado.
        self.dataset_repository = dataset_repository

    def execute(self):

        # ---------------------------------------------------------
        # 1. CARGA DEL DATASET
        # ---------------------------------------------------------

        dataset = self.dataset_repository.load()

        # Las mismas 13 variables utilizadas como entrada
        # en nuestros modelos.
        predictor_variables = (
            CATEGORICAL_VARIABLES
            + NUMERIC_VARIABLES
        )

        # ---------------------------------------------------------
        # 2. AGRUPACIÓN POR VARIABLES PREDICTORAS
        # ---------------------------------------------------------

        # Agrupamos registros que tengan exactamente
        # los mismos valores en las 13 variables X.
        #
        # Para cada combinación calculamos:
        # - número de registros
        # - cantidad de COSM2 diferentes
        # - mínimo y máximo de COSM2
        groups = (
            dataset
            .groupby(
                predictor_variables,
                dropna=False
            )
            .agg(
                records=(self.TARGET_VARIABLE, "size"),
                unique_target_values=(
                    self.TARGET_VARIABLE,
                    "nunique"
                ),
                min_cosm2=(
                    self.TARGET_VARIABLE,
                    "min"
                ),
                max_cosm2=(
                    self.TARGET_VARIABLE,
                    "max"
                ),
                mean_cosm2=(
                    self.TARGET_VARIABLE,
                    "mean"
                )
            )
            .reset_index()
        )

        # ---------------------------------------------------------
        # 3. COMBINACIONES REPETIDAS
        # ---------------------------------------------------------

        # Existen varias filas con exactamente
        # las mismas variables predictoras.
        repeated_groups = groups[
            groups["records"] > 1
        ].copy()

        # ---------------------------------------------------------
        # 4. GRUPOS AMBIGUOS
        # ---------------------------------------------------------

        # Una combinación es ambigua cuando:
        #
        # - aparece más de una vez
        # - pero tiene diferentes valores de COSM2
        #
        # Para un modelo, esos registros son indistinguibles
        # porque reciben exactamente las mismas variables X.
        ambiguous_groups = repeated_groups[
            repeated_groups[
                "unique_target_values"
            ] > 1
        ].copy()

        # Diferencia entre el COSM2 máximo y mínimo
        # dentro de una misma combinación de variables.
        ambiguous_groups["target_range"] = (
            ambiguous_groups["max_cosm2"]
            - ambiguous_groups["min_cosm2"]
        )

        # Ordenamos primero los casos donde existe
        # mayor diferencia en COSM2.
        ambiguous_groups = (
            ambiguous_groups
            .sort_values(
                by="target_range",
                ascending=False
            )
        )

        # ---------------------------------------------------------
        # 5. REGISTROS AFECTADOS
        # ---------------------------------------------------------

        # Cuántos registros forman parte de grupos repetidos.
        repeated_records = int(
            repeated_groups["records"].sum()
        )

        # Cuántos registros forman parte específicamente
        # de grupos ambiguos.
        ambiguous_records = int(
            ambiguous_groups["records"].sum()
        )

        total_records = int(len(dataset))

        # ---------------------------------------------------------
        # 6. RESULTADO
        # ---------------------------------------------------------

        return {
            "total_records": total_records,

            "predictor_variables": int(
                len(predictor_variables)
            ),

            "repeated_predictor_groups": int(
                len(repeated_groups)
            ),

            "records_in_repeated_groups":
                repeated_records,

            "ambiguous_predictor_groups": int(
                len(ambiguous_groups)
            ),

            "records_in_ambiguous_groups":
                ambiguous_records,

            "ambiguous_records_percentage": float(
                (
                    ambiguous_records
                    / total_records
                    * 100
                )
                if total_records > 0
                else 0
            ),

            # Mostramos los 20 grupos con mayor diferencia
            # entre sus valores de COSM2.
            "largest_ambiguities": (
                ambiguous_groups
                .head(20)
                .to_dict(orient="records")
            )
        }