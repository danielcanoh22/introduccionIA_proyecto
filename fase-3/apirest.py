from flask import Flask, request, jsonify
import pandas as pd
from model.train import train_model  # Importa la función de entrenamiento del script train
from model.predict import make_prediction  # Importa la función de predicción del script predict
from loguru import logger
import os
import zipfile

app = Flask(__name__)

# Ruta temporal para archivos de entrada y salida
INPUT_FILE = "input_test.csv"
DEFAULT_TEST_FILE = "data/test.csv"
OUTPUT_FILE = "predicts_mushroom.csv"
MODEL_FILE = "mushroom_model.pkl"
ZIP_TRAIN_FILE = "data/train.zip"
ZIP_TEST_FILE = "data/test.zip"
EXTRACTED_FOLDER = "data/"

# Función para extraer el archivo ZIP
def extract_zip_file(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    logger.info(f"Extracted files in {extract_to}")

# Endpoint para entrenar el modelo
@app.route('/train', methods=['POST'])
def train():
    """
    Endpoint para entrenar el modelo.
    Ejecuta la función train_model() que entrena y guarda el modelo.
    
    Retorna:
    - JSON con el estado de éxito o error del entrenamiento.
    """
    try:
        # Extraer los archivos ZIP antes de entrenar
        extract_zip_file(ZIP_TRAIN_FILE, EXTRACTED_FOLDER)

        # Ruta al archivo de entrenamiento extraído
        train_file_path = os.path.join(EXTRACTED_FOLDER, "train.csv")

        # Llamar a la función train_model para entrenar el modelo
        train_model(data_file=train_file_path, model_file=MODEL_FILE, overwrite=True)

        return jsonify({"status": "Trained and saved model successfully"})
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({"error": str(e)}), 500

# Endpoint para predecir con el modelo
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar predicciones usando el modelo entrenado.
    - Si se recibe un archivo CSV en la solicitud, se utiliza ese archivo.
    - Si no se recibe un archivo, se utiliza el archivo predeterminado `test.csv`.
    Ejecuta la función make_prediction() y procesa las predicciones generadas.
    
    Retorna:
    - JSON con el mensaje de archivo utilizado y las predicciones generadas o un mensaje de error.
    """
    try:
        # Extraer archivos antes de predecir
        extract_zip_file(ZIP_TEST_FILE, EXTRACTED_FOLDER)

        # Ruta del archivo de prueba extraído
        test_file_path = os.path.join(EXTRACTED_FOLDER, "test.csv")

        # Verifica si se envió un archivo; si no, usa el archivo predeterminado
        if 'file' in request.files:
            # Guarda el archivo recibido temporalmente
            file = request.files['file']
            file.save(INPUT_FILE)
            input_file_used = INPUT_FILE
            message = "Predictions made with the provided file"
        else:
            # Usa el archivo predeterminado
            input_file_used = test_file_path
            message = "Predictions made with the default file"

        # Llamar a la función make_prediction para realizar las predicciones
        make_prediction(input_file=input_file_used, model_file=MODEL_FILE, output_file=OUTPUT_FILE)

        # Leer el archivo de salida y devolver las predicciones
        df_resultado = pd.read_csv(OUTPUT_FILE)
        predictions = df_resultado.to_dict(orient="records")
        return jsonify({"message": message, "predictions": predictions})
    except Exception as e:
        logger.error(f"Error when making predictions: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
