import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from loguru import logger
import os

def update(df):
    """
    Actualiza las columnas categóricas del DataFrame, llenando valores faltantes y reemplazando valores poco frecuentes con 'noise'.

    Args:
        df (pd.DataFrame): DataFrame con datos de entrada.

    Returns:
        pd.DataFrame: DataFrame actualizado con columnas categóricas modificadas.
    """

    threshold = 100
    cat_c = [
        'cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
        'stem-root', 'stem-surface', 'stem-color', 'veil-type', 'veil-color', 'has-ring', 'ring-type',
        'spore-print-color', 'habitat', 'season',
    ]

    for col in cat_c:
        df[col] = df[col].fillna('missing')
        df.loc[df[col].value_counts(dropna=False)[df[col]].values < threshold, col] = "noise"
        df[col] = df[col].astype('category')

    return df

def make_prediction(input_file, model_file, output_file):
    """
    Realiza predicciones utilizando el modelo entrenado.

    Args:
        input_file (str): Ruta del archivo CSV con los datos de entrada.
        model_file (str): Ruta del archivo del modelo entrenado.
        output_file (str): Ruta del archivo donde se guardarán las predicciones.

    Retorna:
        None
    """
    # Comprobación de la existencia del archivo del modelo
    if not os.path.isfile(model_file):
        logger.error(f"model file {model_file} does not exist")
        raise FileNotFoundError(f"Model file {model_file} does not exist")

    # Cargar el archivo CSV en un DataFrame
    logger.info("loading input data")
    input_df = pd.read_csv(input_file)

    # Cargar el modelo desde el archivo PKL
    logger.info("loading model")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Llenar valores faltantes
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    # Llenar valores faltantes en todas las columnas del DataFrame
    for col in input_df.columns:
        input_df[col] = imp.fit_transform(input_df[[col]])[:, 0]

    # Actualizar las columnas categóricas
    input_df = update(input_df)

    # Realizar predicciones con el modelo
    logger.info("making predictions")
    preds = model.predict(input_df)

    # Crear un DataFrame con las predicciones
    predictions_file = pd.DataFrame(preds, columns=['class'])
    predictions_file['class'] = predictions_file['class'].replace({1: 'p', 0: 'e'})

    # Mostrar las predicciones realizadas
    logger.info(f"predictions: {predictions_file}")

    # Guardar el archivo de predicciones
    try:
        logger.info(f"Output file will be saved at: {os.path.abspath(output_file)}")
        logger.info(f"saving predictions to {output_file}")
        predictions_file.to_csv(output_file, index=False)
    except Exception as e:
        logger.error(f"error saving file: {e}")
        raise

