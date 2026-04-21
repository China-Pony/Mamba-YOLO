from ultralytics.engine.model import Model
from ultralytics.nn.tasks import MambaYOLORTDETRModel
from .predict import MambaYOLORTDETRPredictor
from .train import MambaYOLORTDETRTrainer
from .val import MambaYOLORTDETRValidator


class MambaYOLORTDETR(Model):

    def __init__(self, model="Mamba-YOLO-L-rtdetr.yaml") -> None:
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        return {
            "detect": {
                "predictor": MambaYOLORTDETRPredictor,
                "validator": MambaYOLORTDETRValidator,
                "trainer": MambaYOLORTDETRTrainer,
                "model": MambaYOLORTDETRModel,
            }
        }
