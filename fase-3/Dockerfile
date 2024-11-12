# Usar una imagen base de Python más ligera
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . /app

# Instalar las dependencias adicionales necesarias

# Instalar las dependencias necesarias para Flask, pandas y loguru
RUN pip install --no-cache-dir Flask

# Instalar pandas para la manipulación de datos
RUN pip install --no-cache-dir pandas 

# Instalar loguru para manejo de logs
RUN pip install --no-cache-dir loguru

# Instalar category-encoders para manejo de características categóricas
RUN pip install --no-cache-dir category-encoders

# Instalar XGBoost para modelos de clasificación y regresión
RUN pip install --no-cache-dir xgboost

# Exponer el puerto para la API (5000)
EXPOSE 5000

# Comando para ejecutar la API Flask
CMD ["python", "apirest.py"]
