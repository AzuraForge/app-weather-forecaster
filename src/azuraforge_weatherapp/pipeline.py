# app-weather-forecaster/src/azuraforge_weatherapp/pipeline.py

import logging
from typing import Any, Dict, Tuple, List
import yaml
from importlib import resources
import pandas as pd
import requests

from azuraforge_learner import Sequential, LSTM, Linear, TimeSeriesPipeline

# --- Varsayılan Konfigürasyonu Yükleme Fonksiyonu ---
def get_default_config() -> Dict[str, Any]:
    """Eklentinin varsayılan YAML konfigürasyonunu yükler."""
    try:
        with resources.open_text("azuraforge_weatherapp.config", "weather_forecaster_config.yml") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Hava durumu uygulaması için varsayılan config yüklenemedi: {e}", exc_info=True)
        return {"error": f"Varsayılan konfigürasyon yüklenemedi: {e}"}

# --- Ana Pipeline Sınıfı ---
class WeatherForecastPipeline(TimeSeriesPipeline):
    """
    Open-Meteo API'sinden alınan hava durumu verileriyle sıcaklık tahmini yapar.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("WeatherForecastPipeline (Eklenti) başarıyla başlatıldı.")

    def _load_data_from_source(self) -> pd.DataFrame:
        """
        Open-Meteo API'sinden geçmiş hava durumu verilerini çeker.
        """
        params = self.config.get("data_sourcing", {})
        self.logger.info(f"'_load_data_from_source' çağrıldı. Parametreler: {params}")

        # === DÜZELTME BAŞLANGICI ===
        # 'hourly_vars' parametresinin bir liste olduğundan emin olalım.
        # Eğer config'den string olarak gelirse (eski bir kalıntı), onu listeye çevirelim.
        hourly_vars = params.get("hourly_vars", [])
        if isinstance(hourly_vars, str):
            hourly_vars = [var.strip() for var in hourly_vars.split(',')]
        
        # Open-Meteo'nun beklediği format: "temperature_2m,precipitation,..."
        hourly_vars_str = ",".join(hourly_vars)
        # === DÜZELTME SONU ===
        
        api_url = "https://archive-api.open-meteo.com/v1/archive"
        api_params = {
            "latitude": params.get("latitude"),
            "longitude": params.get("longitude"),
            "start_date": "2020-01-01",
            "end_date": pd.to_datetime("today").strftime('%Y-%m-%d'),
            "hourly": hourly_vars_str, # Düzeltilmiş string'i kullan
            "timezone": "auto"
        }
        
        self.logger.info(f"Open-Meteo API'sine istek gönderiliyor: {api_url} with params {api_params}")

        response = requests.get(api_url, params=api_params)
        
        # API'den gelen hatayı daha detaylı logla
        if not response.ok:
            try:
                error_details = response.json()
                self.logger.error(f"Open-Meteo API Hatası ({response.status_code}): {error_details}")
            except Exception:
                self.logger.error(f"Open-Meteo API Hatası ({response.status_code}): {response.text}")

        response.raise_for_status() # HTTP hatası varsa exception fırlat
        
        data = response.json()
        
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        if df.empty:
            self.logger.error(f"Open-Meteo'dan boş veri döndü. Parametreler: {api_params}")
            raise ValueError("Open-Meteo API'sinden veri indirilemedi.")
            
        self.logger.info(f"{len(df)} satır hava durumu verisi indirildi.")
        return df

    def get_caching_params(self) -> Dict[str, Any]:
        """
        Önbellek anahtarı oluşturmak için hangi parametrelerin önemli olduğunu belirtir.
        """
        ds_config = self.config.get("data_sourcing", {})
        
        # hourly_vars'ın her zaman bir liste olmasını sağla
        hourly_vars = ds_config.get("hourly_vars", [])
        if isinstance(hourly_vars, str):
            hourly_vars = [var.strip() for var in hourly_vars.split(',')]
            
        sorted_vars = sorted(hourly_vars)
        caching_params = {
            "latitude": ds_config.get("latitude"),
            "longitude": ds_config.get("longitude"),
            "hourly_vars": ",".join(sorted_vars)
        }
        self.logger.info(f"'get_caching_params' çağrıldı. Cache anahtar parametreleri: {caching_params}")
        return caching_params

    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        """Modelin hedef değişkenini ve girdi özelliklerini belirtir."""
        target_col = self.config.get("feature_engineering", {}).get("target_col", "temperature_2m")
        
        feature_cols = self.config.get("data_sourcing", {}).get("hourly_vars", [])
        if isinstance(feature_cols, str):
            feature_cols = [var.strip() for var in feature_cols.split(',')]

        self.logger.info(f"'_get_target_and_feature_cols' çağrıldı. Hedef: {target_col}, Özellikler: {feature_cols}")
        return target_col, feature_cols

    def _create_model(self, input_shape: Tuple) -> Sequential:
        """LSTM ve bir Linear katmandan oluşan modeli oluşturur."""
        self.logger.info(f"'_create_model' çağrıldı. Girdi şekli: {input_shape}")
        
        num_features = input_shape[2] 
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 50)
        
        model = Sequential(
            LSTM(input_size=num_features, hidden_size=hidden_size),
            Linear(hidden_size, 1)
        )
        self.logger.info("LSTM modeli (hava durumu için) başarıyla oluşturuldu.")
        return model

# Eklentinin şemasını ve config'ini sağlayan fonksiyonlar
def get_form_schema():
    """UI'da dinamik form oluşturmak için şema (gelecek için)."""
    # Bu fonksiyon, gelecekteki dinamik UI mimarimiz için bir yer tutucudur.
    # Şimdilik boş bırakabilir veya temel bir yapı döndürebiliriz.
    return {"message": "Dynamic schema not yet implemented for this plugin."}