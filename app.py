from health_assistant.inference import load_models
from health_assistant.ui import run_app


if __name__ == "__main__":
    models = load_models()
    run_app(models)
