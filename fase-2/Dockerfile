# Usar una imagen base de Python con TensorFlow preinstalado
FROM tensorflow/tensorflow:2.17.0 

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . /app

# Instalar las dependencias adicionales necesarias
# Instalar pandas para la manipulación de datos
RUN pip install --no-cache-dir pandas 

# Instalar loguru para manejo de logs
RUN pip install --no-cache-dir loguru

# Instalar category-encoders para manejo de características categóricas
RUN pip install --no-cache-dir category-encoders

# Instalar XGBoost para modelos de clasificación y regresión
RUN pip install --no-cache-dir xgboost

# Exponer puerto (por si se tiene alguna API)
EXPOSE 5000




