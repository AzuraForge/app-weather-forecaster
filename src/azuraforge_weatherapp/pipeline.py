# app-weather-forecaster/src/azuraforge_weatherapp/pipeline.py

import logging
from typing import Any, Dict, Tuple, List, Optional
import yaml
from importlib import resources
import pandas as pd
import requests
from pydantic import BaseModel

from azuraforge_learner.pipelines.timeseries import TimeSeriesPipeline
from azuraforge_learner import Sequential, LSTM, Linear

# Yeni oluşturduğumuz config şemasını import ediyoruz.
from .config_schema import WeatherForecasterConfig

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
    
    # === KRİTİK DÜZELTME: Soyut metodu doğru şekilde implemente ediyoruz ===
    def get_config_model(self) -> Optional[type[BaseModel]]:
        """Bu pipeline için Pydantic konfigürasyon modelini döndürür."""
        return WeatherForecasterConfig

    def _load_data_from_source(self) -> pd.DataFrame:
        """Open-Meteo API'sinden geçmiş hava durumu verilerini çeker."""
        params = self.config.get("data_sourcing", {})
        self.logger.info(f"'_load_data_from_source' called with params: {params}")

        hourly_vars = params.get("hourly_vars", [])
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
        ds_config = self.config.get("data_sourcing", {})
        hourly_vars = ds_config.get("hourly_vars", [])
        sorted_vars = sorted(hourly_vars)
        return {
            "latitude": ds_config.get("latitude"),
            "longitude": ds_config.get("longitude"),
            "hourly_vars": ",".join(sorted_vars)
        }

    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        target_col = self.config.get("feature_engineering", {}).get("target_col", "temperature_2m")
        # Bu pipeline'da, tüm 'hourly_vars' özellik olarak kullanılır.
        feature_cols = self.config.get("data_sourcing", {}).get("hourly_vars", [])
        return target_col, feature_cols

    def _create_model(self, input_shape: Tuple) -> Sequential:
        self.logger.info(f"'_create_model' called. Input shape: {input_shape}")
        # Girdi şekli (batch, seq_len, features)
        input_size = input_shape[2] 
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 50)
        
        model = Sequential(
            LSTM(input_size=input_size, hidden_size=hidden_size),
            Linear(hidden_size, 1) # Çıktı tek bir değer (tahmin)
        )
        self.logger.info("LSTM model for weather created successfully.")
        return model