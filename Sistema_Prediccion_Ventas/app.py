import streamlit as st
import pandas as pd
import joblib

# CARGAR MODELO
modelo_data = joblib.load("modelo_prediccion_ventas.pkl")

modelo = modelo_data["modelo"]
columnas_entrada = modelo_data["columnas_entrada"]

# CONFIGURACIÓN
st.set_page_config(
    page_title="Sistema Predictivo de Ventas",
    layout="centered"
)

st.title("Sistema Predictivo de Ventas")
st.write("Predicción de ventas usando Machine Learning")

# FORMULARIO
producto = st.text_input("Producto")

categoria = st.selectbox(
    "Categoría",
    ["Tecnologia", "Hogar", "Ropa", "Deportes"]
)

precio = st.number_input(
    "Precio",
    min_value=0.0,
    value=100.0
)

descuento = st.number_input(
    "Descuento (%)",
    min_value=0,
    max_value=100,
    value=0
)

stock = st.number_input(
    "Stock",
    min_value=0,
    value=10
)

publicidad = st.selectbox(
    "Publicidad",
    ["Si", "No"]
)

calificacion = st.slider(
    "Calificación",
    1.0,
    5.0,
    4.0
)

visitas = st.number_input(
    "Visitas Web",
    min_value=0,
    value=100
)

mes = st.selectbox(
    "Mes",
    [
        "Enero","Febrero","Marzo","Abril",
        "Mayo","Junio","Julio","Agosto",
        "Septiembre","Octubre","Noviembre","Diciembre"
    ]
)

dia = st.selectbox(
    "Día Semana",
    [
        "Lunes","Martes","Miércoles",
        "Jueves","Viernes","Sábado","Domingo"
    ]
)

temporada = st.selectbox(
    "Temporada",
    ["Alta", "Baja", "Normal"]
)

# BOTÓN PREDICCIÓN
if st.button("Predecir Ventas"):

    try:

        nuevo_producto = pd.DataFrame([{
            "Producto": producto,
            "Categoria": categoria,
            "Precio": precio,
            "Descuento": descuento,
            "Stock": stock,
            "Publicidad": publicidad,
            "Calificacion": calificacion,
            "Visitas_Web": visitas,
            "Mes": mes,
            "Dia_Semana": dia,
            "Temporada": temporada
        }])

        nuevo_producto = nuevo_producto[columnas_entrada]

        prediccion = modelo.predict(nuevo_producto)

        st.success(
            f"Ventas estimadas: {round(prediccion[0])} unidades"
        )

    except Exception as e:
        st.error(f"Error: {e}")