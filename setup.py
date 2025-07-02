# app-weather-forecaster/setup.py

from setuptools import setup, find_packages

setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # Paket kurulduğunda .yml gibi Python dışı dosyaların da
    # kopyalanmasını sağlar. Bu çok önemlidir.
    include_package_data=True, 
    package_data={
        # "azuraforge_weatherapp" paketi içindeki tüm .yml dosyalarını dahil et.
        "azuraforge_weatherapp": ["config/*.yml"], 
    },
)