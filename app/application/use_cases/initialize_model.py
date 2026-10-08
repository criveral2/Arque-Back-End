class InitializeModelUseCase:

    def __init__(self, model_repository):
        self.model_repository = model_repository

    def execute(self):
        if self.model_repository.exists():
            self.print_message("Modelo predictivo encontrado.")
            return self.model_repository.load()

        self.print_message("No existe un modelo predictivo entrenado.")
        return None

    def print_message(self, message: str):
        print("\n" + "=" * 55)
        print(" ARQUÉ - MODEL SERVICE")
        print("-" * 55)
        print(f" {message}")
        print("=" * 55 + "\n")