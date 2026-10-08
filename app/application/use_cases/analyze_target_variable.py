class AnalyzeTargetVariableUseCase:

    # Variable que nuestro modelo intentará predecir.
    TARGET_VARIABLE = "COSM2"

    def __init__(self, dataset_repository):
        # Repositorio encargado de cargar la base ESED.
        self.dataset_repository = dataset_repository

    def execute(self):

        # Cargamos el dataset completo.
        dataset = self.dataset_repository.load()

        # Obtenemos únicamente la variable objetivo COSM2.
        target = dataset[self.TARGET_VARIABLE]

        # Calculamos los cuartiles.
        #
        # Q1 = valor por debajo del cual se encuentra el 25 % de los datos.
        # Q3 = valor por debajo del cual se encuentra el 75 % de los datos.
        q1 = target.quantile(0.25)
        q3 = target.quantile(0.75)

        # Rango intercuartílico (IQR).
        #
        # Se utiliza comúnmente para detectar posibles valores atípicos.
        iqr = q3 - q1

        # Límites teóricos del criterio IQR.
        lower_limit = q1 - (1.5 * iqr)
        upper_limit = q3 + (1.5 * iqr)

        return {

            # Número total de observaciones.
            "records": int(target.count()),

            # Registros cuyo costo estimado es cero.
            "zero_values": int((target == 0).sum()),

            # Estadísticos descriptivos principales.
            "min": float(target.min()),
            "max": float(target.max()),
            "mean": float(target.mean()),
            "median": float(target.median()),

            # Cuartiles.
            "q1": float(q1),
            "q3": float(q3),

            # Rango intercuartílico.
            "iqr": float(iqr),

            # Límites obtenidos mediante el criterio IQR.
            "lower_iqr_limit": float(lower_limit),
            "upper_iqr_limit": float(upper_limit),

            # Percentiles que nos permiten observar
            # la parte superior de la distribución.
            "percentiles": {
                "90": float(target.quantile(0.90)),
                "95": float(target.quantile(0.95)),
                "99": float(target.quantile(0.99)),
                "99.5": float(target.quantile(0.995)),
                "99.9": float(target.quantile(0.999))
            },

            # Contamos cuántos registros superan
            # algunos valores de referencia.
            #
            # Esto NO significa que vayamos a eliminarlos.
            # Solo nos permite estudiar la cola de la distribución.
            "high_value_counts": {
                ">1000": int((target > 1000).sum()),
                ">1500": int((target > 1500).sum()),
                ">2000": int((target > 2000).sum()),
                ">3000": int((target > 3000).sum()),
                ">5000": int((target > 5000).sum())
            }
        }