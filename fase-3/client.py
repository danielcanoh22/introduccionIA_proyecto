import requests

# URL base de la API
BASE_URL = 'http://localhost:5000'

# Archivo de prueba con pocos datos para testear la API
PREDICTION_FILE = 'test_api/test_final.csv'

def train_model():
    """
    Llama al endpoint `/train` para entrenar el modelo.
    """
    try:
        response = requests.post(f"{BASE_URL}/train")
        if response.status_code == 200:
            print("Successfully trained model")
            print("Response:", response.json())
        else:
            print("Error training model")
            print("Response:", response.json())
    except Exception as e:
        print(f"Training request error: {e}")

def make_prediction(file_path=None):
    """
    Llama al endpoint `/predict` para realizar predicciones.
    
    Parámetros:
    - file_path (str): Ruta del archivo CSV para enviar en la solicitud.
                       Si es None, se utilizará el archivo predeterminado.
    """
    try:
        # Si se sube un archivo diferente al default, utilizarlo para las predicciones
        files = {'file': open(file_path, 'rb')} if file_path else None
        response = requests.post(f"{BASE_URL}/predict", files=files)

        # Verificar el estado de la respuesta
        if response.status_code == 200:
            data = response.json()
            print(data['message'])
            print("Predictions:", data['predictions'])
        else:
            print("Error when making predictions")
            print("Response:", response.json())
    except Exception as e:
        print(f"Prediction request failed: {e}")
    finally:
        # Cerrar el archivo
        if files:
            files['file'].close()

# Llamada a las funciones
if __name__ == "__main__":
    print("Training the model...")
    train_model()

    print("\nMaking predictions...")
    make_prediction(PREDICTION_FILE)
