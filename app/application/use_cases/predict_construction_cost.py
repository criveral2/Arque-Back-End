import pandas as pd

from app.domain.model_features import (
    CATEGORICAL_VARIABLES,
    NUMERIC_VARIABLES
)


class PredictConstructionCostUseCase:

    def __init__(self, model):

        # El modelo ya se encuentra entrenado
        # y cargado en memoria.
        self.model = model

    def execute(self, input_data: dict):

        # ---------------------------------------------------------
        # 1. VALIDACIÓN DEL MODELO
        # ---------------------------------------------------------

        if self.model is None:
            raise ValueError(
                "El modelo de predicción no está cargado."
            )

        # ---------------------------------------------------------
        # 2. CONSTRUCCIÓN DEL REGISTRO
        # ---------------------------------------------------------

        predictor_variables = (
            CATEGORICAL_VARIABLES
            + NUMERIC_VARIABLES
        )

        # Creamos un DataFrame porque el Pipeline fue
        # entrenado utilizando nombres de columnas.
        input_dataframe = pd.DataFrame(
            [
                {
                    variable: input_data.get(variable)
                    for variable in predictor_variables
                }
            ]
        )

        # ---------------------------------------------------------
        # 3. PREDICCIÓN
        # ---------------------------------------------------------

        # El Pipeline realiza automáticamente:
        #
        # - imputación
        # - OneHotEncoding
        # - escalado
        # - Random Forest
        # - transformación inversa expm1()
        prediction = self.model.predict(
            input_dataframe
        )

        predicted_cost_m2 = float(
            prediction[0]
        )

        # ---------------------------------------------------------
        # 4. COSTO TOTAL ESTIMADO
        # ---------------------------------------------------------

        construction_area = float(
            input_data["CARCO"]
        )

        estimated_total_cost = (
            predicted_cost_m2
            * construction_area
        )

        # ---------------------------------------------------------
        # 5. RESPUESTA
        # ---------------------------------------------------------

        return {
            "estimated_cost_m2": round(
                predicted_cost_m2,
                2
            ),

            "construction_area_m2": round(
                construction_area,
                2
            ),

            "estimated_total_cost": round(
                estimated_total_cost,
                2
            ),

            "currency": "USD"
        }