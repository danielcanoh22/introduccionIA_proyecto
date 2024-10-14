# Importar bibliotecas y recursos necesarios
import numpy as np
import pandas as pd
import argparse
from sklearn.impute import SimpleImputer
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import train_test_split
import os
from loguru import logger
import pickle

# Configurar el parser de argumentos para la línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', required=True, type=str, help='a csv file with train data')
parser.add_argument('--model_file', type=str, help='where the trained model will be stored')
parser.add_argument('--overwrite_model', default=False, action='store_true', help='if sets overwrites the model file if it exists')

# Parsear los argumentos de la línea de comandos
args = parser.parse_args()

# Asignar los argumentos a variables
model_file = args.model_file
data_file  = args.data_file
overwrite = args.overwrite_model

# Comprobación de la existencia del archivo del modelo
if os.path.isfile(model_file):
    if overwrite:
        logger.info(f"overwriting existing model file {model_file}")
    else:
        logger.info(f"model file {model_file} exists. exitting. use --overwrite_model option")
        exit(-1)

# Cargar el archivo CSV en un DataFrame
logger.info("loading train data")
train = pd.read_csv(data_file)

# Llenar valores faltantes en la columna 'cap-diameter' usando la moda de la columna
logger.info("preprocessing train data")
train['cap-diameter'] =train['cap-diameter'] .fillna(train['cap-diameter'].mode()[0])
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# Llenar valores faltantes en todas las columnas del DataFrame
for col in train.columns:
    train[col] = imp.fit_transform(train[[col]])[:, 0]

def update(df):
    """
    Actualiza las columnas categóricas del DataFrame, llenando valores faltantes y reemplazando valores poco frecuentes con 'noise'.

    Args:
        df (pd.DataFrame): DataFrame con datos de entrenamiento.

    Returns:
        pd.DataFrame: DataFrame actualizado con columnas categóricas modificadas.
    """

    threshold = 100
    cat_c = [
        'cap-shape','cap-surface','cap-color','does-bruise-or-bleed','gill-attachment','gill-spacing','gill-color',
        'stem-root','stem-surface','stem-color','veil-type','veil-color','has-ring','ring-type',
        'spore-print-color','habitat','season',
            ]

    for col in cat_c:
        df[col] = df[col].fillna('missing')
        df.loc[df[col].value_counts(dropna=False)[df[col]].values < threshold, col] = "noise"
        df[col] = df[col].astype('category')

    return df

# Actualizar el DataFrame de entrenamiento con la función update
train  = update(train)

# Convertir la columna 'class' a tipo booleano (True si es 'p', False en caso contrario)
train['class'].value_counts()
train['class'] = train['class'] =='p'

# Inicializar el codificador de características categóricas
encoder  = TargetEncoder()

# Identificar las características categóricas del DataFrame
cat_features = [val for val in train.drop(columns = 'class').select_dtypes(exclude ='number').columns]
for feature in cat_features:
    train[feature] = encoder.fit_transform(train[feature], train['class'])

# Separar los datos en variables predictivas (X) y la variable objetivo (y)
X = train.drop(['class'], axis=1)
y = train['class']

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5)

# Cargar el modelo desde un archivo PKL
with open('mushroom_model.pkl', 'rb') as file:
    clf = pickle.load(file)

# Entrenar el modelo
logger.info("training model")
clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# Guardar el modelo entrenado en un archivo PKL
logger.info("saving model")
with open('mushroom_model.pkl', 'wb') as f:
  pickle.dump(clf, f)