import yaml
from pydantic import BaseModel, Field, DirectoryPath


class Config(BaseModel):
    batch_size: int = Field(64, ge=1, description="Batch size for training")
    learning_rate: float = Field(0.02, ge=0, description="Initial learning rate")
    epochs: int = Field(300, ge=1, description="Number of training epochs")
    warm_up: int = Field(30, ge=0, description="Warmup epochs")
    num_classes: int = Field(100, ge=1, description="Number of classes in the dataset")
    p_threshold: float = Field(0.5, ge=0, le=1, description="Clean probability threshold")
    alpha: float = Field(4.0, ge=0, description="Beta distribution alpha parameter")
    lambda_u: float = Field(25.0, ge=0, description="Weight for unsupervised loss")
    num_workers: int = Field(5, ge=0, description="Number of DataLoader workers")
    temperature: float = Field(0.5, ge=0.1, le=10, description="Sharpening temperature")

    root_dir: DirectoryPath = Field("./datasets", description="Root directory for the dataset")

    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)
        super().__init__(**config_data)
