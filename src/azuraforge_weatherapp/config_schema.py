# app-weather-forecaster/src/azuraforge_weatherapp/config_schema.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Union, Literal, Optional

class DataSourcingConfig(BaseModel):
    latitude: float
    longitude: float
    hourly_vars: Union[List[str], str]
    data_limit: Optional[int] = Field(None, gt=0, description="Limits the data to the last N rows for faster training.")

    @field_validator('hourly_vars')
    @classmethod
    def convert_str_to_list(cls, v: Union[List[str], str]) -> List[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(',') if item.strip()]
        return v

# ... Diğer sınıflar aynı kalıyor ...
class FeatureEngineeringConfig(BaseModel):
    target_col: str
    target_col_transform: Literal['none', 'log'] = 'none'

class ModelParamsConfig(BaseModel):
    sequence_length: int = 60
    hidden_size: int = 50

class TrainingParamsConfig(BaseModel):
    epochs: int = 50
    lr: float = 0.001
    optimizer: Literal['adam', 'sgd'] = 'adam'
    test_size: float = 0.2
    validate_every: int = 5
    batch_size: int = Field(32, gt=0, description="Number of samples per gradient update.") # <-- YENİ SATIR

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
        extra = 'forbid'