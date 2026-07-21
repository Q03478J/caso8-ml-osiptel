# ══════════════════════════════════════════════════════
# CASO 8 - Clasificación de Reclamos OSIPTEL
# Jack Yhems Coronel Guevara - UPLA 2026
# Estilo: igloo.inc — glassmorphism + arctic dark UI
# ══════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from supabase import create_client, Client

st.set_page_config(
    page_title="OSIPTEL · Clasificador ML",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
.stApp { background: #020408; font-family: 'Space Grotesk', sans-serif; }
.stApp::before {
    content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,180,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(120,60,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 30% at 50% 50%, rgba(0,255,200,0.04) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}
.hero-container { text-align: center; padding: 3rem 1rem 2rem; }
.hero-tag {
    display: inline-block; background: rgba(0,180,255,0.1);
    border: 1px solid rgba(0,180,255,0.25); color: #00B4FF;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; padding: 0.35rem 1rem;
    border-radius: 100px; margin-bottom: 1.5rem;
}
.hero-title {
    font-size: 3.2rem; font-weight: 700; line-height: 1.1;
    color: #FFFFFF; letter-spacing: -0.03em; margin-bottom: 0.8rem;
}
.hero-title span {
    background: linear-gradient(135deg, #00B4FF 0%, #7B5FFF 50%, #00FFD1 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub { font-size: 1rem; font-weight: 300; color: rgba(255,255,255,0.45); margin-bottom: 2rem; }
.stats-row { display: flex; justify-content: center; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 2.5rem; }
.stat-pill {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px; padding: 0.45rem 1.1rem;
    font-size: 0.82rem; color: rgba(255,255,255,0.6);
}
.stat-pill strong { color: #00B4FF; font-weight: 600; }
.glass-card {
    background: rgba(255,255,255,0.033); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 2rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.glass-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,180,255,0.4), transparent);
}
.card-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.15em;
    text-transform: uppercase; color: rgba(0,180,255,0.7); margin-bottom: 1.2rem;
}
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: rgba(255,255,255,0.55) !important; font-size: 0.82rem !important;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0066FF 0%, #7B5FFF 100%) !important;
    border: none !important; border-radius: 12px !important; color: white !important;
    font-size: 0.95rem !important; font-weight: 600 !important;
    padding: 0.85rem 2rem !important;
    box-shadow: 0 0 30px rgba(0,102,255,0.25) !important;
}
.result-card {
    background: rgba(0,255,180,0.05); border: 1px solid rgba(0,255,180,0.2);
    border-radius: 16px; padding: 1.8rem; text-align: center; margin: 1.5rem 0;
    position: relative; overflow: hidden;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,180,0.6), transparent);
}
.result-area { font-size: 2rem; font-weight: 700; color: #00FFB4; margin: 0.5rem 0; }
.result-desc { font-size: 0.88rem; color: rgba(255,255,255,0.45); font-weight: 300; }
.arctic-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 2rem 0;
}
.footer-text {
    text-align: center; font-size: 0.75rem; color: rgba(255,255,255,0.2);
    letter-spacing: 0.05em; padding: 2rem 0 1rem;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 0 !important; max-width: 720px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <div class="hero-tag">📡 Machine Learning · UPLA 2026</div>
    <div class="hero-title">Clasificador de<br><span>Reclamos OSIPTEL</span></div>
    <div class="hero-sub">Caso 8 — Enrutamiento Automático de Reclamos en Servicios Públicos</div>
    <div class="stats-row">
        <div class="stat-pill"><strong>SVM Lineal</strong> · Modelo principal</div>
        <div class="stat-pill">Accuracy <strong>91.4%</strong></div>
        <div class="stat-pill">F1-Score <strong>90.9%</strong></div>
        <div class="stat-pill">Latencia <strong>0.8ms</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

@st.cache_resource
def load_model():
    model   = joblib.load("models/modelo_entrenado.pkl")
    columns = joblib.load("models/columnas_ohe.pkl")
    return model, columns

model, columnas_ohe = load_model()

st.markdown('<div class="glass-card"><div class="card-label">Datos del reclamo</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    empresa       = st.selectbox("Empresa Operadora", ["Movistar","Claro","Entel","Bitel"])
    tipo_servicio = st.selectbox("Tipo de Servicio", ["Movil","Internet_Fijo","Telefonia_Fija","TV_Paga"])
    motivo        = st.selectbox("Motivo del Reclamo", ["Facturacion","Calidad_Servicio","Velocidad","Cobertura","Portabilidad"])

with col2:
    region    = st.selectbox("Región", ["Lima","Junin","Cusco","Arequipa","Piura","La_Libertad","Lambayeque","Ica","Ancash","Puno"])
    instancia = st.selectbox("Instancia", ["Primera","Segunda"])
    cantidad  = st.number_input("Cantidad de reclamos similares", min_value=1, max_value=1000, value=50)
    tasa      = st.slider("Tasa de resolución histórica (%)", 0.0, 100.0, 65.0)

st.markdown('</div>', unsafe_allow_html=True)

if st.button("⚡ Clasificar y Enrutar Reclamo"):
    input_dict = {
        'empresa': [empresa], 'tipo_servicio': [tipo_servicio],
        'motivo': [motivo], 'region': [region], 'instancia': [instancia],
        'cantidad_reclamos': [float(cantidad)], 'tasa_resolucion': [float(tasa)]
    }
    df_input = pd.DataFrame(input_dict)
    df_ohe   = pd.get_dummies(df_input, columns=['empresa','tipo_servicio','motivo','region','instancia'])
    df_final = df_ohe.reindex(columns=columnas_ohe, fill_value=0)
    prediccion = model.predict(df_final)[0]

    iconos = {"Facturacion":"🟡","Tecnica":"🔵","Comercial":"🟢","Legal":"🔴"}
    descripciones = {
        "Facturacion": "Derivar al Área de Facturación · Revisión de cobros y cargos indebidos",
        "Tecnica":     "Derivar al Área Técnica · Diagnóstico y resolución de fallas de servicio",
        "Comercial":   "Derivar al Área Comercial · Gestión contractual y ofertas no cumplidas",
        "Legal":       "Derivar al Área Legal · Posibles infracciones normativas y regulatorias"
    }

    st.markdown(f"""
    <div class="result-card">
        <div style="font-size:0.75rem;letter-spacing:0.15em;text-transform:uppercase;
                    color:rgba(0,255,180,0.6);font-weight:600;margin-bottom:0.5rem;">
            Área Resolutiva Detectada
        </div>
        <div class="result-area">{iconos.get(prediccion,'⚪')} {prediccion}</div>
        <div class="result-desc">{descripciones.get(prediccion,'')}</div>
    </div>
    """, unsafe_allow_html=True)

    payload = {
        "inputs_usuario": {"empresa":empresa,"tipo_servicio":tipo_servicio,
                           "motivo":motivo,"region":region,"instancia":instancia,
                           "cantidad":int(cantidad),"tasa":float(tasa)},
        "resultado_prediccion": prediccion
    }
    try:
        supabase.table("predicciones_log").insert(payload).execute()
        st.markdown('<p style="text-align:center;font-size:0.75rem;color:rgba(0,255,180,0.4);margin-top:0.5rem;">✓ Registrado en Supabase</p>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Error Supabase: {e}")

st.markdown('<div class="arctic-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card"><div class="card-label">Historial de predicciones</div>', unsafe_allow_html=True)
if st.button("📋 Ver últimas 10 predicciones"):
    try:
        resp = supabase.table("predicciones_log").select("*").order("fecha", desc=True).limit(10).execute()
        if resp.data:
            df_hist = pd.DataFrame(resp.data)
            df_hist['fecha'] = pd.to_datetime(df_hist['fecha']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(df_hist[['id','fecha','resultado_prediccion']], use_container_width=True)
        else:
            st.markdown('<p style="color:rgba(255,255,255,0.3);font-size:0.85rem;">Sin predicciones aún.</p>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
    JACK YHEMS CORONEL GUEVARA · INGENIERÍA DE SISTEMAS · UPLA 2026 · CASO 8 ML
</div>
""", unsafe_allow_html=True)
