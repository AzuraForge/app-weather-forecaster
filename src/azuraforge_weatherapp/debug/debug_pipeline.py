# ========== DOSYA: app-weather-forecaster/debug/debug_pipeline.py ==========

import logging
import json
from pprint import pprint
import sys
import os

# Proje kök dizinini Python yoluna ekleyerek kardeş repolara erişimi sağlıyoruz.
# Bu, 'azuraforge_learner' gibi paketleri bulmasını sağlar.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Artık kardeş repo'lardan import yapabiliriz.
from azuraforge_learner import Callback
from azuraforge_weatherapp.pipeline import WeatherForecastPipeline, get_default_config

# Redis'e olan bağımlılığı kaldırmak için sahte (mock) bir callback
class MockProgressCallback(Callback):
    """Redis'e bağlanmak yerine ilerlemeyi sadece konsola yazdıran sahte callback."""
    def on_epoch_end(self, event):
        payload = event.payload
        epoch = payload.get('epoch', '?')
        total_epochs = payload.get('total_epochs', '?')
        loss = payload.get('loss', float('nan'))
        print(f"  [DEBUG_CALLBACK] Epoch {epoch}/{total_epochs} -> Loss: {loss:.6f}")

def run_isolated_test():
    """Pipeline'ı izole bir şekilde çalıştıran ana fonksiyon."""
    
    print("--- 🔬 AzuraForge İzole Pipeline Testi Başlatılıyor ---")

    # 1. Pipeline'ın varsayılan konfigürasyonunu yükle
    default_config = get_default_config()
    print("✅ Varsayılan konfigürasyon yüklendi.")

    # 2. Test için parametreleri burada kolayca değiştirin
    override_params = {
        "model_params": {
            "sequence_length": 168,
            "hidden_size": 32
        },
        "training_params": {
            "epochs": 10,
            "lr": 0.001,
            "optimizer": "adam",
            "test_size": 0.2,
            "validate_every": 2,
            "batch_size": 64
        },
        "data_sourcing": {
            "data_limit": 4380
        }
    }
    
    final_config = default_config.copy()
    for key, value in override_params.items():
        if key in final_config and isinstance(final_config[key], dict):
            final_config[key].update(value)
        else:
            final_config[key] = value

    print("\n🔧 Çalıştırılacak Konfigürasyon:")
    print(json.dumps(final_config, indent=2))

    # 3. Pipeline örneğini ve sahte callback'i oluştur
    final_config['experiment_dir'] = './.tmp/debug_run'
    pipeline = WeatherForecastPipeline(final_config)
    mock_callback = MockProgressCallback()

    print("\n🚀 Pipeline çalıştırılıyor...")
    
    try:
        # === DEĞİŞİKLİK BURADA BAŞLIYOR ===
        # Önce veriyi pipeline'ın kendi metoduyla yükle
        print("\nVeri kaynaktan yükleniyor...")
        raw_data = pipeline._load_data_from_source()
        print(f"✅ Veri yüklendi. {len(raw_data)} satır bulundu.")

        # Şimdi yüklenen veriyi `run` metoduna argüman olarak ver
        results = pipeline.run(raw_data=raw_data, callbacks=[mock_callback])
        # === DEĞİŞİKLİK SONU ===

        print("\n--- ✅ Test Başarıyla Tamamlandı ---")
        print("Sonuç Metrikleri:")
        pprint(results.get('metrics', {}))
        print("\nSon Kayıp Değeri:")
        pprint(results.get('final_loss'))

    except Exception as e:
        print(f"\n--- ❌ Test Sırasında Hata Oluştu ---")
        logging.error("Pipeline hatası:", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    run_isolated_test()