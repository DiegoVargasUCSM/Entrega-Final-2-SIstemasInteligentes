import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor

# CARGAR CSV
df = pd.read_csv("ventas_productos.csv")

# LIMPIEZA
df = df.dropna()
df = df.drop_duplicates()

# VARIABLE OBJETIVO
objetivo = "Ventas"

# ELIMINAR COLUMNAS INNECESARIAS
columnas_eliminar = [objetivo]

if "ID" in df.columns:
    columnas_eliminar.append("ID")

X = df.drop(columns=columnas_eliminar)
y = df[objetivo]

# COLUMNAS
columnas_numericas = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

columnas_categoricas = X.select_dtypes(
    include=["object"]
).columns.tolist()

# PREPROCESAMIENTO
preprocesador = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            columnas_numericas
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            columnas_categoricas
        )
    ]
)

# MODELO
modelo = Pipeline(steps=[
    ("preprocesador", preprocesador),
    ("modelo", DecisionTreeRegressor(
        random_state=42
    ))
])

# TRAIN TEST
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ENTRENAR
modelo.fit(X_train, y_train)

# GUARDAR
paquete = {
    "modelo": modelo,
    "columnas_entrada": X.columns.tolist()
}

joblib.dump(
    paquete,
    "modelo_prediccion_ventas.pkl"
)

print("Modelo guardado correctamente")