# Importar bibliotecas y recursos necesarios
import numpy as np
import pandas as pd
import argparse
from sklearn.impute import SimpleImputer
from category_encoders.target_encoder import TargetEncoder
from loguru import logger
import os
import pickle

# Configurar el parser de argumentos para la línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', required=True, type=str, help='a csv file with input data (no targets)')
parser.add_argument('--model_file', required=True, type=str, help='a pkl file with a model already stored (see train.py)')
parser.add_argument('--output_file', default='predictions.csv', type=str, help='the output file where predictions will be stored')

# Parsear los argumentos de la línea de comandos
args = parser.parse_args()

# Asignar los argumentos a variables
model_file = args.model_file
input_file = args.input_file
output_file = args.output_file

# Comprobación de la existencia del archivo del modelo
if not os.path.isfile(model_file):
    logger.error(f"model file {model_file} does not exist")
    exit(-1)

# Comprobación de la existencia del archivo de entrada
if not os.path.isfile(input_file):
    logger.error(f"input file {input_file} does not exist")
    exit(-1)

# Cargar el archivo CSV en un DataFrame
logger.info("loading input data")
input_df = pd.read_csv(input_file)  

# Cargar el modelo desde el archivo PKL
logger.info("loading model")
with open(model_file, 'rb') as f:
    m = pickle.load(f)
 
 # Llenar valores faltantes en la columna 'cap-diameter' usando la moda de la columna
input_df['cap-diameter'] = input_df['cap-diameter'].fillna(input_df['cap-diameter'].mode()[0])
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# Llenar valores faltantes en todas las columnas del DataFrame
for col in input_df.columns:
    input_df[col] = imp.fit_transform(input_df[[col]])[:, 0]

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

# Actualizar el DataFrame de entrada con la función update
input_df = update(input_df)

# Hacer predicciones usando el modelo
logger.info("making predictions")
preds = m.predict(input_df)

# Crear un DataFrame con las predicciones
predictions_file = pd.DataFrame(preds, columns=['class'])
predictions_file['class'] = predictions_file['class'].replace({1: 'p', 0: 'e'})

# Mostrar las predicciones realizadas
logger.info(f"predicciones: {predictions_file}")

# Generar el archivo csv con las predicciones
try:
    logger.info(f"Output file will be saved at: {os.path.abspath(output_file)}")
    logger.info(f"saving predictions to {output_file}")
    predictions_file.to_csv(output_file, index=False)
except Exception as e:
    logger.error(f"error saving file: {e}")