import numpy as np

from app.domain.model_features import (
    CATEGORICAL_VARIABLES,
    NUMERIC_VARIABLES,
    TARGET_VARIABLE
)


class AnalyzeTemporalShiftUseCase:

    def __init__(
        self,
        training_dataset_repository,
        test_dataset_repository
    ):
        # Dataset utilizado durante el desarrollo del modelo.
        # Corresponde a ESED 2023.
        self.training_dataset_repository = (
            training_dataset_repository
        )

        # Dataset temporal posterior.
        # Corresponde a ESED 2025.
        self.test_dataset_repository = (
            test_dataset_repository
        )

    def execute(self):

        # ---------------------------------------------------------
        # 1. CARGA DE DATASETS
        # ---------------------------------------------------------

        dataset_2023 = (
            self.training_dataset_repository.load()
        )

        dataset_2025 = (
            self.test_dataset_repository.load()
        )

        # ---------------------------------------------------------
        # 2. ANÁLISIS DE LA VARIABLE OBJETIVO
        # ---------------------------------------------------------

        target_2023 = dataset_2023[
            TARGET_VARIABLE
        ]

        target_2025 = dataset_2025[
            TARGET_VARIABLE
        ]

        target_summary = {
            "2023": self._target_summary(
                target_2023
            ),

            "2025": self._target_summary(
                target_2025
            )
        }

        # ---------------------------------------------------------
        # 3. CAMBIO DE MEDIA Y MEDIANA DE COSM2
        # ---------------------------------------------------------

        mean_2023 = float(
            target_2023.mean()
        )

        mean_2025 = float(
            target_2025.mean()
        )

        median_2023 = float(
            target_2023.median()
        )

        median_2025 = float(
            target_2025.median()
        )

        target_shift = {
            "mean_absolute_change": float(
                mean_2025 - mean_2023
            ),

            "mean_percentage_change": float(
                (
                    (mean_2025 - mean_2023)
                    / mean_2023
                ) * 100
            ),

            "median_absolute_change": float(
                median_2025 - median_2023
            ),

            "median_percentage_change": float(
                (
                    (median_2025 - median_2023)
                    / median_2023
                ) * 100
            )
        }

        # ---------------------------------------------------------
        # 4. DISTRIBUCIÓN DE COSTOS ALTOS
        # ---------------------------------------------------------

        # Analizamos específicamente la presencia de
        # proyectos con costos elevados por metro cuadrado.
        #
        # Esto es relevante porque la evaluación temporal
        # mostró varios casos de 2025 entre 1400 y 1500 USD/m².
        high_cost_distribution = {
            "2023": self._cost_thresholds(
                target_2023
            ),

            "2025": self._cost_thresholds(
                target_2025
            )
        }

        # ---------------------------------------------------------
        # 5. CAMBIO EN VARIABLES NUMÉRICAS
        # ---------------------------------------------------------

        numeric_shift = {}

        for variable in NUMERIC_VARIABLES:

            values_2023 = dataset_2023[
                variable
            ]

            values_2025 = dataset_2025[
                variable
            ]

            median_numeric_2023 = float(
                values_2023.median()
            )

            median_numeric_2025 = float(
                values_2025.median()
            )

            numeric_shift[variable] = {
                "2023": {
                    "mean": self._safe_float(
                        values_2023.mean()
                    ),

                    "median": self._safe_float(
                        values_2023.median()
                    ),

                    "std": self._safe_float(
                        values_2023.std()
                    )
                },

                "2025": {
                    "mean": self._safe_float(
                        values_2025.mean()
                    ),

                    "median": self._safe_float(
                        values_2025.median()
                    ),

                    "std": self._safe_float(
                        values_2025.std()
                    )
                },

                # Diferencia directa entre las medianas.
                "median_absolute_change": float(
                    median_numeric_2025
                    - median_numeric_2023
                )
            }

        # ---------------------------------------------------------
        # 6. CATEGORÍAS NUEVAS EN 2025
        # ---------------------------------------------------------

        categorical_shift = {}

        for variable in CATEGORICAL_VARIABLES:

            series_2023 = dataset_2023[
                variable
            ]

            series_2025 = dataset_2025[
                variable
            ]

            # Categorías conocidas durante el entrenamiento.
            categories_2023 = set(
                series_2023
                .dropna()
                .unique()
                .tolist()
            )

            # Categorías encontradas posteriormente.
            categories_2025 = set(
                series_2025
                .dropna()
                .unique()
                .tolist()
            )

            # Categorías presentes en 2025 que nunca
            # aparecieron en ESED 2023.
            new_categories = (
                categories_2025
                - categories_2023
            )

            # Identificamos cuántos registros de 2025
            # contienen una categoría que el modelo
            # nunca observó durante su entrenamiento.
            valid_2025 = series_2025.notna()

            unseen_mask = (
                valid_2025
                & ~series_2025.isin(
                    categories_2023
                )
            )

            unseen_records = int(
                unseen_mask.sum()
            )

            categorical_shift[variable] = {
                "categories_2023": int(
                    len(categories_2023)
                ),

                "categories_2025": int(
                    len(categories_2025)
                ),

                "new_categories_2025_count": int(
                    len(new_categories)
                ),

                # Limitamos la respuesta para evitar
                # devolver listas demasiado grandes.
                "new_categories_2025": sorted(
                    [
                        str(value)
                        for value in new_categories
                    ]
                )[:20],

                "records_with_unseen_category": (
                    unseen_records
                ),

                "unseen_percentage": float(
                    (
                        unseen_records
                        / len(dataset_2025)
                    ) * 100
                )
            }

        # ---------------------------------------------------------
        # 7. VALORES AUSENTES
        # ---------------------------------------------------------

        predictor_variables = (
            CATEGORICAL_VARIABLES
            + NUMERIC_VARIABLES
        )

        missing_values = {}

        for variable in predictor_variables:

            missing_2023 = int(
                dataset_2023[
                    variable
                ].isna().sum()
            )

            missing_2025 = int(
                dataset_2025[
                    variable
                ].isna().sum()
            )

            missing_values[variable] = {
                "2023": {
                    "records": missing_2023,

                    "percentage": float(
                        (
                            missing_2023
                            / len(dataset_2023)
                        ) * 100
                    )
                },

                "2025": {
                    "records": missing_2025,

                    "percentage": float(
                        (
                            missing_2025
                            / len(dataset_2025)
                        ) * 100
                    )
                }
            }

        # ---------------------------------------------------------
        # 8. RESPUESTA
        # ---------------------------------------------------------

        return {
            "analysis": (
                "Temporal distribution shift 2023-2025"
            ),

            "records": {
                "2023": int(
                    len(dataset_2023)
                ),

                "2025": int(
                    len(dataset_2025)
                )
            },

            "target_summary": (
                target_summary
            ),

            "target_shift": (
                target_shift
            ),

            "high_cost_distribution": (
                high_cost_distribution
            ),

            "numeric_shift": (
                numeric_shift
            ),

            "categorical_shift": (
                categorical_shift
            ),

            "missing_values": (
                missing_values
            )
        }

    # -------------------------------------------------------------
    # MÉTODOS AUXILIARES
    # -------------------------------------------------------------

    def _target_summary(
        self,
        target
    ):

        return {
            "mean": float(
                target.mean()
            ),

            "median": float(
                target.median()
            ),

            "std": float(
                target.std()
            ),

            "min": float(
                target.min()
            ),

            "max": float(
                target.max()
            ),

            "p25": float(
                target.quantile(0.25)
            ),

            "p75": float(
                target.quantile(0.75)
            ),

            "p90": float(
                target.quantile(0.90)
            ),

            "p95": float(
                target.quantile(0.95)
            ),

            "p99": float(
                target.quantile(0.99)
            )
        }

    def _cost_thresholds(
        self,
        target
    ):

        thresholds = [
            500,
            800,
            1000,
            1200,
            1400
        ]

        result = {}

        for threshold in thresholds:

            count = int(
                (
                    target > threshold
                ).sum()
            )

            result[
                f">{threshold}"
            ] = {
                "records": count,

                "percentage": float(
                    (
                        count
                        / len(target)
                    ) * 100
                )
            }

        return result

    @staticmethod
    def _safe_float(
        value
    ):

        # Evita devolver NaN dentro del JSON.
        if value is None:
            return None

        if np.isnan(value):
            return None

        return float(value)