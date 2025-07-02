# ========== app-weather-forecaster/Dockerfile ==========
# Bu Dockerfile, kütüphanenin bağımsız olarak build edilip
# test edilebildiğini doğrulamak içindir.

# --- Aşama 1: Builder ---
# Bu aşama, tüm bağımlılıkları kurar ve paketi hazırlar.
FROM python:3.10-slim-bullseye AS builder

# Geliştirme için gerekli sistem paketleri (örn: git)
RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini ayarla
WORKDIR /app

# Önce sadece paket yönetimi dosyalarını kopyala.
# Bu, Docker'ın katman önbelleklemesini (layer caching) optimize eder.
# Bağımlılıklar değişmediği sürece bu katman yeniden build edilmez.
COPY pyproject.toml setup.py ./

# Bağımlılıkları kur. --no-cache-dir imaj boyutunu küçültür.
# --mount=type=cache,target=/root/.cache/pip build'leri hızlandırır.
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -e .

# Geri kalan tüm proje kaynak kodunu kopyala
COPY src ./src


# --- Aşama 2: Final/Runtime ---
# Bu aşama, sadece gerekli olanları içeren temiz ve küçük bir imaj oluşturur.
FROM python:3.10-slim-bullseye AS final

WORKDIR /app

# Builder aşamasından sadece kurulu paketleri ve kaynak kodunu kopyala.
# Bu, build araçlarını ve gereksiz sistem kütüphanelerini son imajdan uzak tutar.
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /app/src /app/src

# PYTHONPATH'i ayarla ki Python, /app/src içindeki paketleri bulabilsin.
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Bu komut, 'docker run' ile çalıştırıldığında build'in başarılı olduğunu gösterir.
# CI/CD ortamında bu komut 'pytest' ile override edilebilir.
CMD ["python", "-c", "print('✅ AzuraForge App Weather Forecaster library built successfully!')"]