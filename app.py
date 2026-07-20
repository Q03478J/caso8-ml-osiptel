# ══════════════════════════════════════════════════════
# CASO 8 - Clasificación de Reclamos OSIPTEL
# Jack Yhems Coronel Guevara - UPLA 2026
# Algoritmo: SVM Lineal (Accuracy: 91.4%)
# ══════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from supabase import create_client, Client

# ── CONFIG DE PÁGINA ───────────────────────────────
st.set_page_config(
    page_title="Clasificador de Reclamos OSIPTEL",
    page_icon="📡",
    layout="centered"
)

# ── CONEXIÓN SUPABASE ──────────────────────────────
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# ── CARGA DEL MODELO ───────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load("models/modelo_entrenado.pkl")
    columns = joblib.load("models/columnas_ohe.pkl")
    return model, columns

model, columnas_ohe = load_model()

# ── INTERFAZ ───────────────────────────────────────
st.title("📡 Clasificador de Reclamos OSIPTEL")
st.subheader("Caso 8 — Enrutamiento Automático de Reclamos")
st.markdown("**Algoritmo:** SVM Lineal · **Accuracy:** 91.4% · **Dataset:** OSIPTEL")
st.divider()

st.markdown("### Ingresa los datos del reclamo:")

col1, col2 = st.columns(2)

with col1:
    empresa = st.selectbox(
        "Empresa Operadora:",
        ["Movistar", "Claro", "Entel", "Bitel"]
    )
    tipo_servicio = st.selectbox(
        "Tipo de Servicio:",
        ["Movil", "Internet_Fijo", "Telefonia_Fija", "TV_Paga"]
    )
    motivo = st.selectbox(
        "Motivo del Reclamo:",
        ["Facturacion", "Calidad_Servicio", "Velocidad", "Cobertura", "Portabilidad"]
    )

with col2:
    region = st.selectbox(
        "Región:",
        ["Lima", "Junin", "Cusco", "Arequipa", "Piura",
         "La_Libertad", "Lambayeque", "Ica", "Ancash", "Puno"]
    )
    instancia = st.selectbox(
        "Instancia:",
        ["Primera", "Segunda"]
    )
    cantidad = st.number_input(
        "Cantidad de reclamos similares:",
        min_value=1, max_value=1000, value=50
    )
    tasa = st.slider(
        "Tasa de resolución histórica (%):",
        min_value=0.0, max_value=100.0, value=65.0
    )

st.divider()

# ── PREDICCIÓN ─────────────────────────────────────
if st.button("🔍 Clasificar Reclamo", use_container_width=True, type="primary"):

    # Construir dataframe con los inputs
    input_dict = {
        'empresa': [empresa],
        'tipo_servicio': [tipo_servicio],
        'motivo': [motivo],
        'region': [region],
        'instancia': [instancia],
        'cantidad_reclamos': [float(cantidad)],
        'tasa_resolucion': [float(tasa)]
    }
    df_input = pd.DataFrame(input_dict)

    # One-Hot Encoding
    df_ohe = pd.get_dummies(df_input,
                             columns=['empresa','tipo_servicio',
                                      'motivo','region','instancia'])

    # Alinear columnas con las del modelo entrenado
    df_final = df_ohe.reindex(columns=columnas_ohe, fill_value=0)

    # Predicción
    prediccion = model.predict(df_final)[0]

    # Colores por área
    colores = {
        "Facturacion": "🟡",
        "Tecnica":     "🔵",
        "Comercial":   "🟢",
        "Legal":       "🔴"
    }
    icono = colores.get(prediccion, "⚪")

    st.success(f"### {icono} Área Resolutiva: **{prediccion}**")

    descripciones = {
        "Facturacion": "El reclamo debe derivarse al Área de Facturación para revisión de cobros y cargos.",
        "Tecnica":     "El reclamo debe derivarse al Área Técnica para diagnóstico de fallas de servicio.",
        "Comercial":   "El reclamo debe derivarse al Área Comercial para temas contractuales y ofertas.",
        "Legal":       "El reclamo debe derivarse al Área Legal por posibles infracciones normativas."
    }
    st.info(descripciones.get(prediccion, ""))

    # ── GUARDAR EN SUPABASE ────────────────────────
    payload = {
        "inputs_usuario": {
            "empresa": empresa,
            "tipo_servicio": tipo_servicio,
            "motivo": motivo,
            "region": region,
            "instancia": instancia,
            "cantidad": int(cantidad),
            "tasa_resolucion": float(tasa)
        },
        "resultado_prediccion": prediccion
    }

    try:
        supabase.table("predicciones_log").insert(payload).execute()
        st.caption("✅ Consulta registrada en Supabase.")
    except Exception as e:
        st.warning(f"No se pudo registrar en Supabase: {e}")

# ── HISTORIAL ──────────────────────────────────────
st.divider()
if st.button("📋 Ver últimas predicciones registradas"):
    try:
        resp = supabase.table("predicciones_log")\
                       .select("*")\
                       .order("fecha", desc=True)\
                       .limit(10)\
                       .execute()
        if resp.data:
            st.dataframe(pd.DataFrame(resp.data))
        else:
            st.info("Aún no hay predicciones registradas.")
    except Exception as e:
        st.error(f"Error al obtener historial: {e}")

st.divider()
st.caption("Jack Yhems Coronel Guevara · Ingeniería de Sistemas · UPLA 2026 · Caso 8 ML")
