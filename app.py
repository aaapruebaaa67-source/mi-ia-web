import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="EPSA - Control de Rendimiento", layout="centered")

st.title("Sistema Predictivo de Aprobación de Estudiantes")
st.subheader("Herramienta Auxiliar de Apoyo Pedagógico — EPSA")
st.write("Introduzca los parámetros del estudiante para evaluar el riesgo de suspensión.")

@st.cache_resource
def load_model():
    return joblib.load('modelo_arbol_1.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"No se encontró el modelo: {e}")
    st.stop()

st.markdown("---")
st.header("Parámetros de Evaluación Académica")

col1, col2 = st.columns(2)

with col1:
    g1 = st.slider("Nota del Primer Parcial (G1)", min_value=0, max_value=20, value=10, step=1)

with col2:
    g2 = st.slider("Nota del Segundo Parcial (G2)", min_value=0, max_value=20, value=10, step=1)

st.subheader("Asistencia y Edad")

col3, col4 = st.columns(2)

with col3:
    absences = st.number_input("Faltas de Asistencia", min_value=0, max_value=100, value=2, step=1)

with col4:
    age = st.number_input("Edad", min_value=15, max_value=30, value=18, step=1)

features = pd.DataFrame(
    [[g1, g2, absences, age]],
    columns=["G1.mat", "G2.mat", "absences.mat", "age"]
)

st.markdown("---")

try:
    prediccion = model.predict(features)[0]
    probabilidades = model.predict_proba(features)[0]
    prob_suspenso = probabilidades[0]
    prob_aprobado = probabilidades[1]

    st.header("Resultado del Análisis")

    if prediccion == 0:
        st.error("Riesgo de Suspension Detectado")
        st.write("El modelo estima una alta probabilidad de que el alumno no supere la asignatura. Se sugiere aplicar medidas de refuerzo preventivas.")
    else:
        st.success("Trayectoria Conforme (Aprobado)")
        st.write("El patron de rendimiento actual se asocia con los registros historicos de alumnos aprobados.")

    st.subheader("Nivel de Confianza")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Probabilidad de Suspenso", f"{prob_suspenso:.1%}")
        st.progress(float(prob_suspenso))
    with col_b:
        st.metric("Probabilidad de Aprobado", f"{prob_aprobado:.1%}")
        st.progress(float(prob_aprobado))

    chart_data = pd.DataFrame(
        {"Probabilidad": [prob_suspenso, prob_aprobado]},
        index=["Suspenso", "Aprobado"]
    )
    st.bar_chart(chart_data)

except Exception as e:
    st.error(f"Error durante el procesamiento: {e}")

st.markdown("---")
st.caption("Nota de responsabilidad: Este sistema predictivo es exclusivamente una herramienta "
           "estadística auxiliar de apoyo institucional. No sustituye el criterio evaluador "
           "de los docentes de la Escuela Politécnica Superior de Alcoy.")
