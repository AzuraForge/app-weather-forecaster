# ========== DOSYA: app-weather-forecaster/src/azuraforge_weatherapp/pipeline.py (TAM HALİ) ==========

import logging
from typing import Any, Dict, Tuple, List, Optional
import yaml
from importlib import resources
import pandas as pd
import requests
from pydantic import BaseModel
import numpy as np

from azuraforge_learner import Sequential, LSTM, Linear, TimeSeriesPipeline, Dropout
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
    Open-Meteo API'sinden alınan hava durumu verileriyle tahmin yapar.
    """
    def get_config_model(self) -> type[BaseModel]:
        return WeatherForecasterConfig

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("WeatherForecastPipeline (Eklenti) başarıyla başlatıldı.")

    def _load_data_from_source(self) -> pd.DataFrame:
        """
        Veriyi Open-Meteo API'sinden çeker, zaman özellikleri ekler ve veriyi temizler.
        """
        params = self.config.get("data_sourcing", {})
        self.logger.info(f"'_load_data_from_source' çağrıldı. Parametreler: {params}")

        # _get_target_and_feature_cols'u daha sonra kullanacağımız için burada çağırıp saklayalım
        # Not: Bu, _get_target_and_feature_cols'un _load_data_from_source'a bağımlı olmadığını varsayar.
        temp_target_col, temp_feature_cols_config = self._get_target_and_feature_cols_from_config()
        all_required_vars = set(temp_feature_cols_config)
        all_required_vars.add(temp_target_col)
        hourly_vars_str = ",".join(sorted(list(all_required_vars)))
        
        api_url = "https://archive-api.open-meteo.com/v1/archive"
        api_params = {
            "latitude": params.get("latitude"),
            "longitude": params.get("longitude"),
            "start_date": "2020-01-01",
            "end_date": pd.to_datetime("today").strftime('%Y-%m-%d'),
            "hourly": hourly_vars_str,
            "timezone": "auto"
        }
        
        self.logger.info(f"Open-Meteo API'sine istek gönderiliyor: {api_url} with params {api_params}")
        response = requests.get(api_url, params=api_params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        if df.empty:
            raise ValueError("Open-Meteo API'sinden veri indirilemedi.")
            
        # === YENİ: ZAMAN ÖZELLİKLERİ EKLEME (FEATURE ENGINEERING) ===
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 23.0)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 23.0)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.0)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.0)
        self.logger.info("Döngüsel zaman özellikleri eklendi (sin/cos for hour and day of year).")
        # === YENİ ADIM SONU ===

        df.ffill(inplace=True)
        df.bfill(inplace=True)

        self.logger.info(f"{len(df)} satır hava durumu verisi indirildi ve işlendi.")
        return df
    
    def get_caching_params(self) -> Dict[str, Any]:
        """Önbellek anahtarı için kullanılacak parametreleri döndürür."""
        ds_config = self.config.get("data_sourcing", {})
        # _get_target_and_feature_cols'u kullanarak tüm gerekli değişkenleri al
        target_col, all_required_vars = self._get_target_and_feature_cols()
        caching_params = {
            "latitude": ds_config.get("latitude"),
            "longitude": ds_config.get("longitude"),
            "hourly_vars": ",".join(sorted(all_required_vars)) # Sıralayarak tutarlılık sağla
        }
        self.logger.info(f"Cache anahtar parametreleri: {caching_params}")
        return caching_params

    def _get_target_and_feature_cols_from_config(self) -> Tuple[str, List[str]]:
        """Sadece konfigürasyona bakarak hedef ve özellik sütunlarını alır."""
        target_col = self.config.get("feature_engineering", {}).get("target_col", "temperature_2m")
        feature_cols_raw = self.config.get("data_sourcing", {}).get("hourly_vars", [])
        
        if isinstance(feature_cols_raw, str):
            return target_col, [item.strip() for item in feature_cols_raw.split(',') if item.strip()]
        return target_col, feature_cols_raw

    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        """
        Nihai hedef ve özellik sütunlarını belirler (zaman özellikleri dahil).
        """
        target_col, feature_cols_config = self._get_target_and_feature_cols_from_config()
        
        # === YENİ: ZAMAN ÖZELLİKLERİNİ DE DAHİL ET ===
        base_features = set(feature_cols_config)
        base_features.add(target_col)
        
        temporal_features = ['hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos']
        all_features = sorted(list(base_features.union(temporal_features)))
        # === DEĞİŞİKLİK SONU ===
        
        self.logger.info(f"Nihai Hedef: {target_col}, Nihai Özellikler: {all_features}")
        return target_col, all_features

    def _create_model(self, input_shape: Tuple) -> Sequential:
        self.logger.info(f"Creating LSTM model with Dropout. Input shape: {input_shape}")
        num_features = input_shape[2] 
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 64)
        
        # === YENİ: DROPOUT İLE GÜÇLENDİRİLMİŞ MİMARİ ===
        model = Sequential(
            LSTM(input_size=num_features, hidden_size=hidden_size),
            Dropout(0.2), # LSTM çıkışına %20 oranında dropout uygula
            Linear(hidden_size, 1)
        )
        self.logger.info("LSTM with Dropout model created successfully.")
        return model