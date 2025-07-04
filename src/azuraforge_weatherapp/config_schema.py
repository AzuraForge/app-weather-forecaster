# app-weather-forecaster/src/azuraforge_weatherapp/config_schema.py
from pydantic import BaseModel, Field
from typing import List, Literal

class DataSourcingConfig(BaseModel):
    latitude: float
    longitude: float
    hourly_vars: List[str] = Field(..., min_length=1)

class FeatureEngineeringConfig(BaseModel):
    target_col: str
    target_col_transform: Literal['log', 'none'] = 'none'

class ModelParamsConfig(BaseModel):
    sequence_length: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)

class TrainingParamsConfig(BaseModel):
    epochs: int = Field(..., gt=0)
    lr: float = Field(..., gt=0)
    optimizer: Literal['adam', 'sgd'] = 'adam'
    test_size: float = Field(..., gt=0, lt=1)
    validate_every: int = Field(..., gt=0)

class SystemConfig(BaseModel):
    caching_enabled: bool = True
    cache_max_age_hours: int = 6

class WeatherForecasterConfig(BaseModel):
    """Pydantic model for the weather forecaster pipeline configuration."""
    pipeline_name: Literal['weather_forecaster']
    data_sourcing: DataSourcingConfig
    feature_engineering: FeatureEngineeringConfig
    model_params: ModelParamsConfig
    training_params: TrainingParamsConfig
    system: SystemConfig

    class Config:
        # Pydantic'e, modelde tanımlanmayan ekstra alanları
        # (experiment_id, task_id gibi) görmezden gelmesini söylüyoruz.
        extra = 'ignore'