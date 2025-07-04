# app-weather-forecaster/src/azuraforge_weatherapp/pipeline.py

import logging
from typing import Any, Dict, Tuple, List
import yaml
from importlib import resources
import pandas as pd
import requests
from pydantic import BaseModel

from azuraforge_learner import Sequential, LSTM, Linear, TimeSeriesPipeline
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
        Veriyi Open-Meteo API'sinden çeker, ön işler ve eksik verileri doldurur.
        """
        params = self.config.get("data_sourcing", {})
        self.logger.info(f"'_load_data_from_source' çağrıldı. Parametreler: {params}")

        # === KRİTİK DÜZELTME BAŞLANGICI: Hedef sütunu her zaman dahil et ===
        # _get_target_and_feature_cols'tan gelen birleştirilmiş listeyi kullanıyoruz.
        _, all_required_vars = self._get_target_and_feature_cols()
        hourly_vars_str = ",".join(all_required_vars)
        # === KRİTİK DÜZELTME SONU ===
        
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
            
        # === KRİTİK DÜZELTME BAŞLANGICI: NaN (Not-a-Number) hatasını önle ===
        # API'den gelen verilerde olası boşlukları dolduruyoruz.
        # 'ffill' (forward-fill) metodu, bir önceki geçerli değeri kullanarak boşluğu doldurur.
        df.fillna(method='ffill', inplace=True)
        # Eğer ilk satırlarda hala NaN varsa, 'bfill' (backward-fill) ile doldur.
        df.fillna(method='bfill', inplace=True)
        # === KRİTİK DÜZELTME SONU ===

        self.logger.info(f"{len(df)} satır hava durumu verisi indirildi ve işlendi.")
        return df

    def get_caching_params(self) -> Dict[str, Any]:
        ds_config = self.config.get("data_sourcing", {})
        # _get_target_and_feature_cols'u kullanarak tüm gerekli değişkenleri al
        _, all_required_vars = self._get_target_and_feature_cols()
        caching_params = {
            "latitude": ds_config.get("latitude"),
            "longitude": ds_config.get("longitude"),
            "hourly_vars": ",".join(sorted(all_required_vars))
        }
        self.logger.info(f"'get_caching_params' çağrıldı. Cache anahtar parametreleri: {caching_params}")
        return caching_params

    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        """
        Hedef ve özellik sütunlarını belirler. Hedef sütunun özellikler
        listesinde yer almasını garanti eder.
        """
        target_col = self.config.get("feature_engineering", {}).get("target_col", "temperature_2m")
        feature_cols_from_config = self.config.get("data_sourcing", {}).get("hourly_vars", [])
        
        # === KRİTİK DÜZELTME BAŞLANGICI ===
        # Hedef sütunun özellik listesinde olduğundan emin ol.
        # Set kullanarak kopyaları otomatik olarak engelliyoruz.
        all_cols = set(feature_cols_from_config)
        all_cols.add(target_col)
        feature_cols = sorted(list(all_cols))
        # === KRİTİK DÜZELTME SONU ===
        
        self.logger.info(f"'_get_target_and_feature_cols' çağrıldı. Hedef: {target_col}, Özellikler: {feature_cols}")
        return target_col, feature_cols

    def _create_model(self, input_shape: Tuple) -> Sequential:
        self.logger.info(f"'_create_model' çağrıldı. Girdi şekli: {input_shape}")
        num_features = input_shape[2] 
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 50)
        model = Sequential(
            LSTM(input_size=num_features, hidden_size=hidden_size),
            Linear(hidden_size, 1)
        )
        self.logger.info("LSTM modeli (hava durumu için) başarıyla oluşturuldu.")
        return model