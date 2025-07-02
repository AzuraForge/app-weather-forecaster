# app-weather-forecaster

1.  **İmajı Build Etme:**
    ```bash
    docker build . -t azuraforge/app-weather-forecaster-test
    ```
    Eğer bu komut hatasız tamamlanırsa, `pyproject.toml` dosyanızdaki tüm bağımlılıkların doğru olduğunu ve paketin kurulabildiğini doğrulamış olursunuz.

2.  **Container'ı Çalıştırma (Başarı Mesajı):**
    ```bash
    docker run --rm azuraforge/app-weather-forecaster-test
    ```
    Çıktı olarak `✅ AzuraForge App Weather Forecaster library built successfully!` mesajını görmelisiniz.

3.  **Container İçinde Testleri Çalıştırma (Gelecek İçin):**
    `app-weather-forecaster` reposuna bir `tests` klasörü ve `pytest` testleri eklediğimizde, bu testleri tamamen izole bir ortamda şöyle çalıştırabiliriz:
    ```bash
    # CMD'yi override ederek pytest'i çalıştır
    docker run --rm azuraforge/app-weather-forecaster-test pytest
    ```

