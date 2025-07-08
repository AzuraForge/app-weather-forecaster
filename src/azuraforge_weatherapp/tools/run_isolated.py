# ========== DOSYA: app-weather-forecaster/tools/run_isolated.py ==========
import logging
import json
from pprint import pprint
import sys
import os
import argparse  # Argüman parse etmek için argparse eklendi

# Proje kök dizinini Python yoluna ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from azuraforge_learner import Callback
from azuraforge_weatherapp.pipeline import WeatherForecastPipeline, get_default_config

class MockProgressCallback(Callback):
    """Redis yerine ilerlemeyi konsola yazdıran sahte callback."""
    def on_epoch_end(self, event):
        payload = event.payload
        epoch = payload.get('epoch', '?')
        total_epochs = payload.get('total_epochs', '?')
        loss = payload.get('loss', float('nan'))
        print(f"  [DEBUG] Epoch {epoch}/{total_epochs} -> Loss: {loss:.6f}")

def run_isolated_test(args):
    """Pipeline'ı izole bir şekilde çalıştıran ana fonksiyon."""
    print("--- 🔬 AzuraForge İzole Pipeline Testi Başlatılıyor ---")

    default_config = get_default_config()
    print("✅ Varsayılan konfigürasyon yüklendi.")

    # Komut satırı argümanları ile konfigürasyonu override et
    final_config = default_config.copy()
    if args.hidden_size:
        final_config['model_params']['hidden_size'] = args.hidden_size
    if args.epochs:
        final_config['training_params']['epochs'] = args.epochs
    if args.lr:
        final_config['training_params']['lr'] = args.lr
    if args.batch_size:
        final_config['training_params']['batch_size'] = args.batch_size
    if args.data_limit:
        final_config['data_sourcing']['data_limit'] = args.data_limit

    print("\n🔧 Çalıştırılacak Nihai Konfigürasyon:")
    print(json.dumps(final_config, indent=2))

    final_config['experiment_dir'] = os.path.join(os.path.dirname(__file__), '..', '.tmp', 'isolated_run')
    pipeline = WeatherForecastPipeline(final_config)
    mock_callback = MockProgressCallback()

    print("\n🚀 Pipeline çalıştırılıyor...")
    
    try:
        print("\nVeri kaynaktan yükleniyor...")
        raw_data = pipeline._load_data_from_source()
        print(f"✅ Veri yüklendi. {len(raw_data)} satır bulundu.")

        results = pipeline.run(raw_data=raw_data, callbacks=[mock_callback])

        print("\n--- ✅ Test Başarıyla Tamamlandı ---")
        print("Sonuç Metrikleri:")
        pprint(results.get('metrics', {}))
        print("\nSon Kayıp Değeri:")
        pprint(results.get('final_loss'))

    except Exception:
        print(f"\n--- ❌ Test Sırasında Hata Oluştu ---")
        logging.error("Pipeline hatası:", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="AzuraForge Weather Forecaster - Isolated Runner")
    parser.add_argument("--hidden-size", type=int, help="LSTM gizli katman boyutu.")
    parser.add_argument("--epochs", type=int, help="Eğitim için epoch sayısı.")
    parser.add_argument("--lr", type=float, help="Öğrenme oranı (learning rate).")
    parser.add_argument("--batch-size", type=int, help="Yığın boyutu (batch size).")
    parser.add_argument("--data-limit", type=int, help="Kullanılacak son N veri satırı.")
    
    args = parser.parse_args()
    run_isolated_test(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    main()