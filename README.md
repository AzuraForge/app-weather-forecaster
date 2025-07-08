# AzuraForge: Hava Durumu Tahmin Eklentisi (Rol Model)

Bu proje, AzuraForge platformu için **rol model bir uygulama eklentisidir**. Open-Meteo API'sinden aldığı geçmiş hava durumu verilerini kullanarak, `azuraforge-learner` kütüphanesindeki `LSTM` modeli ile gelecekteki hava sıcaklığını tahmin eder.

Bu eklenti, AzuraForge ekosistemine yeni bir uygulama eklerken takip edilmesi gereken **"altın standartları"** (proje yapısı, test edilebilirlik, dokümantasyon) belirler.

## 🎯 Ana Sorumluluklar

*   Python `entry_points` mekanizması aracılığıyla kendisini AzuraForge ekosistemine bir "pipeline" olarak kaydeder.
*   Kullanıcı arayüzünde (`Dashboard`) dinamik olarak bir form oluşturulabilmesi için gerekli konfigürasyon (`config/weather_forecaster_config.yml`) ve şema (`form_schema.json`) dosyalarını sağlar.
*   `TimeSeriesPipeline` soyut sınıfını miras alarak veri çekme, ön işleme, model oluşturma ve eğitim adımlarını kendi uzmanlık alanına göre (hava durumu verisi) uygular.

---

## 🏛️ Ekosistemdeki Yeri

Bu eklenti, AzuraForge ekosisteminin modüler ve genişletilebilir yapısının canlı bir örneğidir. Projenin genel mimarisini, vizyonunu ve geliştirme rehberini anlamak için lütfen ana **[AzuraForge Platform Dokümantasyonuna](https://github.com/AzuraForge/platform/tree/main/docs)** başvurun.

---

## 🛠️ İzole Geliştirme ve Hızlı Test

Bu eklenti, tüm AzuraForge platformunu (`Docker`) çalıştırmadan, tamamen bağımsız olarak test edilebilir ve geliştirilebilir. Bu, hiperparametre optimizasyonu gibi işlemleri dakikalar içinde yapmanızı sağlar.

### Gereksinimler
1.  Ana platformun **[Geliştirme Rehberi](https://github.com/AzuraForge/platform/blob/main/docs/DEVELOPMENT_GUIDE.md)**'ne göre Python sanal ortamınızın kurulu ve aktif olduğundan emin olun.
2.  Bu reponun kök dizininde olduğunuzdan emin olun.

### Testi Çalıştırma

Aşağıdaki komut, pipeline'ı varsayılan (ve yüksek performanslı) ayarlarla çalıştıracaktır:
```bash
python tools/run_isolated.py
```

### Parametreleri Değiştirerek Test Etme

Farklı hiperparametreleri denemek için komut satırı argümanlarını kullanabilirsiniz. Örneğin, `hidden_size`'ı 128 yapmak ve sadece 10 epoch eğitmek için:
```bash
python tools/run_isolated.py --hidden-size 128 --epochs 10 --batch-size 16
```
Bu komut, varsayılan konfigürasyonu bu argümanlarla geçersiz kılarak deneyi çalıştırır.
