# Importar bibliotecas y recursos necesarios
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import train_test_split
import os
from loguru import logger
import pickle

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


# Función para entrenar el modelo
def train_model(data_file: str, model_file: str, overwrite: bool = False):
    """
    Entrena un modelo a partir de los datos proporcionados y guarda el modelo entrenado.

    Args:
        data_file (str): Ruta al archivo CSV con los datos de entrenamiento.
        model_file (str): Ruta donde se guardará el modelo entrenado.
        overwrite (bool): Si es True, sobrescribe el archivo del modelo si ya existe.
    
    Returns:
        None
    """
    # Comprobación de la existencia del archivo del modelo
    if os.path.isfile(model_file):
        if overwrite:
            logger.info(f"overwriting existing model file {model_file}")
        else:
            logger.info(f"model file {model_file} exists. Exiting. Use --overwrite_model option")
            return

    # Cargar el archivo CSV en un DataFrame
    logger.info("Loading train data")
    train = pd.read_csv(data_file)

    # Llenar valores faltantes en la columna 'cap-diameter' usando la moda de la columna
    logger.info("Preprocessing train data")
    train['cap-diameter'] = train['cap-diameter'].fillna(train['cap-diameter'].mode()[0])
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    # Llenar valores faltantes en todas las columnas del DataFrame
    for col in train.columns:
        train[col] = imp.fit_transform(train[[col]])[:, 0]

    # Función para actualizar columnas categóricas
    def update(df):
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

    # Actualizar el DataFrame de entrenamiento
    train = update(train)

    # Convertir la columna 'class' a tipo booleano
    train['class'] = train['class'] == 'p'

    # Inicializar el codificador de características categóricas
    encoder = TargetEncoder()

    # Identificar las características categóricas del DataFrame
    cat_features = [val for val in train.drop(columns='class').select_dtypes(exclude='number').columns]
    for feature in cat_features:
        train[feature] = encoder.fit_transform(train[feature], train['class'])

    # Separar las características (X) y la variable objetivo (y)
    X = train.drop(['class'], axis=1)
    y = train['class']

    # Dividir los datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5)

    # Crear el clasificador (usando XGBoost por ejemplo)
    from xgboost import XGBClassifier
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Entrenar el modelo
    logger.info("Training model")
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    # Guardar el modelo entrenado en un archivo PKL
    logger.info("Saving model")
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)

    logger.info(f"Model saved to {model_file}")
