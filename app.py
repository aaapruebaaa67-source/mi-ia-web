import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="EPSA - Control de Rendimiento", layout="centered")

st.title("Sistema Predictivo de Aprobación de Estudiantes")
st.subheader("Herramienta Auxiliar de Apoyo Pedagógico — EPSA")
st.write("Introduzca los parámetros del estudiante para evaluar el riesgo de suspensión.")

@st.cache_resource
def load_epsa_model():
    return joblib.load('modelo_arbol 1.pkl')

try:
    model = load_epsa_model()
except Exception as e:
    st.error(f"No se encontró el archivo 'modelo_arbol 1.pkl' en el directorio actual: {e}")
    st.stop()

st.markdown("---")

st.header("Parámetros de Evaluación Académica")

col1, col2 = st.columns(2)
with col1:
    g1 = st.slider("Nota del Primer Parcial (G1)", min_value=0.0, max_value=20.0, value=10.0, step=0.5)
with col2:
    g2 = st.slider("Nota del Segundo Parcial (G2)", min_value=0.0, max_value=20.0, value=10.0, step=0.5)

st.subheader("Asistencia y Edad")
col3, col4 = st.columns(2)
with col3:
    absences = st.number_input("Faltas de Asistencia", min_value=0, max_value=100, value=2, step=1)
with col4:
    age = st.number_input("Edad", min_value=15, max_value=30, value=18, step=1)

features = np.array([[g1, g2, absences, age]])

st.markdown("---")

if st.button("Calcular Predicción"):
    try:
        prediccion = model.predict(features)[0]
        
        st.header("Resultado del Análisis")
        
        if prediccion == 0:
            st.error("Riesgo de Suspensión Detectado")
            st.write("El modelo estima una alta probabilidad de que el alumno no supere la asignatura. Se sugiere aplicar medidas de refuerzo preventivas.")
        else:
            st.success("Trayectoria Conforme (Aprobado)")
            st.write("El patrón de rendimiento actual se asocia con los registros históricos de alumnos aprobados.")
            
        if hasattr(model, "predict_proba"):
            probabilidades = model.predict_proba(features)[0]
            st.subheader("Nivel de Confianza")
            st.bar_chart(probabilidades)
            
            for idx, prob in enumerate(probabilidades):
                estado_texto = "Riesgo de Suspenso" if idx == 0 else "Aprobación"
                st.write(f"Probabilidad de {estado_texto}: {prob:.2%}")
                
    except Exception as e:
        st.error(f"Error durante el procesamiento: {e}")

st.markdown("---")
st.caption("Nota de responsabilidad: Este sistema predictivo es exclusivamente una herramienta "
           "estadística auxiliar de apoyo institucional. No sustituye el criterio evaluador "
           "de los docentes de la Escuela Politécnica Superior de Alcoy.")
