# app-weather-forecaster/src/azuraforge_weatherapp/pipeline.py

import logging
from typing import Any, Dict, Tuple, List, Optional
import yaml
from importlib import resources
import pandas as pd
import requests
from pydantic import BaseModel

from azuraforge_learner.pipelines import TimeSeriesPipeline
from azuraforge_learner import Sequential, LSTM, Linear

def get_default_config() -> Dict[str, Any]:
    """Eklentinin varsayılan YAML konfigürasyonunu yükler."""
    try:
        with resources.open_text("azuraforge_weatherapp.config", "weather_forecaster_config.yml") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load default config for weather app: {e}", exc_info=True)
        return {"error": f"Default config could not be loaded: {e}"}

class WeatherForecastPipeline(TimeSeriesPipeline):
    """
    Open-Meteo API'sinden alınan hava durumu verileriyle sıcaklık tahmini yapar.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("WeatherForecastPipeline (Plugin) initialized successfully.")
    
    def get_config_model(self) -> Optional[type[BaseModel]]:
        """
        Bu pipeline için Pydantic konfigürasyon modelini döndürür.
        Basitlik adına, şimdilik herhangi bir ekstra doğrulama yapmıyoruz.
        """
        # Daha karmaşık doğrulamalar için buraya özel bir Pydantic modeli eklenebilir.
        return None

    def _load_data_from_source(self) -> pd.DataFrame:
        """Open-Meteo API'sinden geçmiş hava durumu verilerini çeker."""
        params = self.config.get("data_sourcing", {})
        self.logger.info(f"'_load_data_from_source' called. Params: {params}")

        hourly_vars = params.get("hourly_vars", [])
        if isinstance(hourly_vars, str):
            hourly_vars = [var.strip() for var in hourly_vars.split(',')]
        
        if not hourly_vars:
            raise ValueError("No hourly variables specified in data_sourcing config.")
            
        hourly_vars_str = ",".join(hourly_vars)
        
        api_url = "https://archive-api.open-meteo.com/v1/archive"
        api_params = {
            "latitude": params.get("latitude"),
            "longitude": params.get("longitude"),
            "start_date": "2020-01-01",
            "end_date": pd.to_datetime("today").strftime('%Y-%m-%d'),
            "hourly": hourly_vars_str,
            "timezone": "auto"
        }
        
        self.logger.info(f"Requesting Open-Meteo API with params: {api_params}")
        response = requests.get(api_url, params=api_params)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        if df.empty:
            raise ValueError("Open-Meteo API returned no data.")
            
        self.logger.info(f"Downloaded {len(df)} rows of weather data.")
        return df

    def get_caching_params(self) -> Dict[str, Any]:
        """Önbellek anahtarı oluşturmak için kullanılacak parametreleri belirtir."""
        ds_config = self.config.get("data_sourcing", {})
        hourly_vars = ds_config.get("hourly_vars", [])
        if isinstance(hourly_vars, str):
            hourly_vars = [var.strip() for var in hourly_vars.split(',')]
            
        sorted_vars = sorted(hourly_vars)
        caching_params = {
            "latitude": ds_config.get("latitude"),
            "longitude": ds_config.get("longitude"),
            "hourly_vars": ",".join(sorted_vars)
        }
        self.logger.info(f"'get_caching_params' called. Cache key params: {caching_params}")
        return caching_params

    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        """Modelin hedef değişkenini ve girdi özelliklerini belirtir."""
        target_col = self.config.get("feature_engineering", {}).get("target_col", "temperature_2m")
        feature_cols = self.config.get("data_sourcing", {}).get("hourly_vars", [])
        if isinstance(feature_cols, str):
            feature_cols = [var.strip() for var in feature_cols.split(',')]
        
        self.logger.info(f"'_get_target_and_feature_cols' called. Target: {target_col}, Features: {feature_cols}")
        return target_col, feature_cols

    def _create_model(self, input_shape: Tuple) -> Sequential:
        """LSTM ve bir Linear katmandan oluşan modeli oluşturur."""
        self.logger.info(f"'_create_model' called. Input shape: {input_shape}")
        
        # Bu pipeline'da tek bir değişken (target) ile sekans oluşturduğumuz için
        # input_size her zaman 1 olacaktır.
        input_size = input_shape[2] 
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 50)
        
        model = Sequential(
            LSTM(input_size=input_size, hidden_size=hidden_size),
            Linear(hidden_size, 1)
        )
        self.logger.info("LSTM model for weather created successfully.")
        return model