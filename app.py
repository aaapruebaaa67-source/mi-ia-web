
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="EPSA - Control de Rendimiento",
    layout="centered",
    page_icon="🎓"
)

st.title("🎓 Sistema Predictivo de Aprobación de Estudiantes")
st.subheader("Herramienta Auxiliar de Apoyo Pedagógico — EPSA")
st.write("Introduzca los parámetros del estudiante para evaluar el riesgo de suspensión.")

@st.cache_resource
def load_epsa_model():
    return joblib.load('modelo_arbol_1.pkl')

try:
    model = load_epsa_model()
except Exception as e:
    st.error(f"No se encontró el archivo 'modelo_arbol_1.pkl': {e}")
    st.stop()

# ── Información del modelo ──────────────────────────────────────────────────
with st.expander("ℹ️ Información del modelo"):
    st.markdown("""
    **Tipo:** Árbol de Decisión (profundidad máx. 3)  
    **Variables predictoras:**
    - `G1.mat` — Nota primer parcial (0–20)  
    - `G2.mat` — Nota segundo parcial (0–20) *(mayor peso)*  
    - `absences.mat` — Faltas de asistencia  
    - `age` — Edad del alumno  

    **Umbral de aprobación:** G2 ≥ 13 (sobre 20) es la variable más determinante.
    """)

st.markdown("---")
st.header("Parámetros de Evaluación Académica")

col1, col2 = st.columns(2)

with col1:
    g1 = st.slider(
        "Nota del Primer Parcial (G1)",
        min_value=0, max_value=20, value=10, step=1,
        help="Escala 0–20. El modelo fue entrenado con notas portuguesas."
    )

with col2:
    g2 = st.slider(
        "Nota del Segundo Parcial (G2)",
        min_value=0, max_value=20, value=10, step=1,
        help="Variable con mayor peso predictivo. Nota ≥ 13 favorece la aprobación."
    )

# Indicador visual del umbral
if g2 < 10:
    st.warning(f"⚠️ G2 = {g2} — Rendimiento muy bajo")
elif g2 < 13:
    st.info(f"ℹ️ G2 = {g2} — Zona de riesgo (umbral de aprobación: 13)")
else:
    st.success(f"✅ G2 = {g2} — Por encima del umbral histórico de aprobación")

st.subheader("Asistencia y Edad")
col3, col4 = st.columns(2)

with col3:
    absences = st.number_input(
        "Faltas de Asistencia", min_value=0, max_value=100, value=2, step=1
    )

with col4:
    age = st.number_input(
        "Edad", min_value=15, max_value=30, value=18, step=1
    )

# ── Construcción del DataFrame con los nombres correctos ───────────────────
# El modelo fue entrenado con estas columnas exactas:
features = pd.DataFrame(
    [[g1, g2, absences, age]],
    columns=["G1.mat", "G2.mat", "absences.mat", "age"]
)

st.markdown("---")

# ── Predicción en tiempo real ───────────────────────────────────────────────
try:
    prediccion = model.predict(features)[0]
    probabilidades = model.predict_proba(features)[0]  # [P(suspenso), P(aprobado)]
    prob_suspenso = probabilidades[0]
    prob_aprobado = probabilidades[1]

    st.header("Resultado del Análisis")

    if prediccion == 0:
        st.error("🔴 Riesgo de Suspensión Detectado")
        st.write(
            "El modelo estima una alta probabilidad de que el alumno no supere "
            "la asignatura. Se sugiere aplicar medidas de refuerzo preventivas."
        )
    else:
        st.success("🟢 Trayectoria Conforme (Aprobado)")
        st.write(
            "El patrón de rendimiento actual se asocia con los registros "
            "históricos de alumnos aprobados."
        )

    # Barra de confianza más clara
    st.subheader("Nivel de Confianza")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Probabilidad de Suspenso", f"{prob_suspenso:.1%}")
        st.progress(float(prob_suspenso))
    with col_b:
        st.metric("Probabilidad de Aprobado", f"{prob_aprobado:.1%}")
        st.progress(float(prob_aprobado))

    # Gráfico de barras con etiquetas correctas
    chart_data = pd.DataFrame(
        {"Probabilidad": [prob_suspenso, prob_aprobado]},
        index=["Suspenso", "Aprobado"]
    )
    st.bar_chart(chart_data)

    # Resumen de inputs
    with st.expander("📋 Datos introducidos"):
        st.dataframe(features.rename(columns={
            "G1.mat": "Nota G1",
            "G2.mat": "Nota G2",
            "absences.mat": "Faltas",
            "age": "Edad"
        }), use_container_width=True)

except Exception as e:
    st.error(f"Error durante el procesamiento: {e}")

st.markdown("---")
st.caption(
    "Nota de responsabilidad: Este sistema predictivo es exclusivamente una herramienta "
    "estadística auxiliar de apoyo institucional. No sustituye el criterio evaluador "
    "de los docentes de la Escuela Politécnica Superior de Alcoy."
)
