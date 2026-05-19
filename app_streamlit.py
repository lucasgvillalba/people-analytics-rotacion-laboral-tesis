# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime          # ⬅️ NUEVO
import os                              # ⬅️ NUEVO

# ---------- Clase usada dentro del pipeline ----------
class FeatureDerivadas(BaseEstimator, TransformerMixin):
    def __init__(self, col_salario="Ingreso_Mensual", col_horas="Horas_Extra", col_eq="Equilibrio_VidaTrabajo"):
        self.col_salario = col_salario
        self.col_horas = col_horas
        self.col_eq = col_eq
        self.p33_ = None
        self.p66_ = None

    def fit(self, X, y=None):
        s = pd.to_numeric(pd.Series(X[self.col_salario]), errors="coerce").dropna()
        self.p33_, self.p66_ = (0, 0) if len(s) == 0 else np.nanpercentile(s, [33, 66])
        return self

    def transform(self, X):
        X = X.copy()
        s = pd.to_numeric(X[self.col_salario], errors="coerce")
        horas_si = X[self.col_horas].astype(str).str.strip().str.lower().isin(["sí", "si", "true", "1"])
        eq_bajo = X[self.col_eq].astype(str).str.strip().str.lower().isin(["bajo", "muy bajo", "muy_bajo", "muy-bajo"])
        X["Extra_Pago_Bajo"] = ((horas_si) & (s <= self.p33_)).astype(int)
        X["HorasAltas_EquilibrioBajo"] = ((horas_si) & (eq_bajo)).astype(int)
        return X
# ---------------------------------------------------

st.set_page_config(page_title="Modelo de predicción de rotación laboral", layout="centered")

# ----------------------------
# 1) Cargar modelo
# ----------------------------
MODEL_PATH = Path("modelos/modelo_xgb_derivadas_v2.joblib")
LOG_PATH = "predicciones_log.csv"   # ⬅️ NUEVO: CSV acumulativo

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"No encuentro el modelo en: {MODEL_PATH}. Verificá la ruta.")
        st.stop()
    return joblib.load(MODEL_PATH)

pipe = load_model()

# ----------------------------
# 2) Catálogos
# ----------------------------
DEPTOS = sorted([
    "Docencia","Investigación y desarrollo","Tecnología","Contaduría",
    "RRHH","Marketing","Mantenimiento","Ventas","Dirección de Carreras"
])

PUESTOS = sorted([
    "Jefe/a de Trabajos Prácticos","Profesor/a Titular","Analista Funcional",
    "Soporte IT","Auxiliar Docente","Profesor/a Adjunto/a",
    "Coordinador/a de Cátedra","Becario/a de Investigación",
    "Desarrollador/a","Administrador/a de Sistemas",
    "Técnico/a de Laboratorio","Encargado/a de Servicios Generales",
    "Supervisor/a de Mantenimiento","Analista Contable",
    "Responsable Impositivo","Técnico/a de Mantenimiento",
    "Tesorero/a","Especialista en Selección","Analista de RRHH",
    "Generalista de RRHH","Especialista en Capacitación",
    "Representante Institucional","Asesor/a de Vinculación",
])

NIVEL_EDU_LABELS = ["Secundario","Terciario","Universitario","Posgrado","Doctorado"]
DISP = ["Completa","Media jornada","Flexible"]
SI_NO = ["No","Sí"]
SAT = ["Muy bajo","Bajo","Bueno","Excelente"]
EQV = ["Muy bajo","Bajo","Bueno","Excelente"]

# ----------------------------
# 3) Sidebar
# ----------------------------
st.sidebar.header("Configuración")
modo = st.sidebar.radio("Seleccioná el tipo de formulario", ["Candidato externo","Empleado actual"])

# 🔒 Umbral fijo (no editable por usuario)
umbral = 0.20

# ----------------------------
# Título
# ----------------------------
st.title("🧠 Modelo de predicción de rotación laboral")
st.markdown("---")

# ----------------------------
# 4) Formularios
# ----------------------------
def common_fields(defaults=None, modo="Empleado actual"):
    d = defaults or {}
    col1, col2 = st.columns(2)
    label_ingreso = "Ingreso Mensual (ARS)" if modo == "Empleado actual" else "Ingreso pretendido (ARS)"

    with col1:
        edad = st.number_input("Edad", min_value=18, max_value=65, value=d.get("Edad", 35))
        depto = st.selectbox("Departamento", DEPTOS, index=DEPTOS.index(d.get("Departamento","Docencia")))
        puesto = st.selectbox("Puesto", PUESTOS, index=PUESTOS.index(d.get("Puesto","Profesor/a Adjunto/a")))
        nivel_label = st.selectbox("Nivel Educativo", NIVEL_EDU_LABELS, index=2)
        nivel_edu = NIVEL_EDU_LABELS.index(nivel_label) + 1
        ingreso = st.number_input(label_ingreso, min_value=700000, max_value=3500000, step=5000, value=d.get("Ingreso_Mensual",1200000))
        disp = st.selectbox("Disponibilidad Horaria", DISP, index=DISP.index(d.get("Disponibilidad_Horaria","Completa")))
        movil = st.selectbox("Movilidad Propia", SI_NO, index=SI_NO.index(d.get("Movilidad_Propia","No")))
        dist = st.number_input("Distancia desde el hogar (km)", min_value=0.0, max_value=60.0, step=0.1, value=d.get("Distancia_KM",8.0))

    with col2:
        hijos = st.number_input("Cantidad de Hijos", min_value=0, max_value=6, value=d.get("Cantidad_Hijos",0))
        if modo == "Empleado actual":
            sat = st.selectbox("Satisfacción con el Puesto", SAT, index=SAT.index(d.get("Satisfaccion_Puesto","Bueno")))
            eq = st.selectbox("Equilibrio Vida-Trabajo", EQV, index=EQV.index(d.get("Equilibrio_VidaTrabajo","Bueno")))
            hextra = st.selectbox("¿Hace Horas Extra?", SI_NO, index=SI_NO.index(d.get("Horas_Extra","No")))
            cap = st.number_input("Capacitaciones en el último año", min_value=0, max_value=10, value=d.get("Capacitaciones_Ultimo_Anio",1))
            anios_emp = st.number_input("Años en la empresa", min_value=0, max_value=40, value=d.get("Anios_Empresa",0))
        else:
            sat = d.get("Satisfaccion_Puesto", "Bueno")
            eq = d.get("Equilibrio_VidaTrabajo", "Bueno")
            hextra = d.get("Horas_Extra", "No")
            cap = d.get("Capacitaciones_Ultimo_Anio", 0)
            anios_emp = d.get("Anios_Empresa", 0)
        prev = st.number_input("N° de empresas previas", min_value=0, max_value=15, value=d.get("Num_Empresas_Previas",0))

    fila = {
        "Edad": int(edad),
        "Departamento": depto,
        "Puesto": puesto,
        "Nivel_Educativo": int(nivel_edu),
        "Ingreso_Mensual": int(ingreso),
        "Disponibilidad_Horaria": disp,
        "Movilidad_Propia": movil,
        "Distancia_KM": float(dist),
        "Cantidad_Hijos": int(hijos),
        "Satisfaccion_Puesto": sat,
        "Equilibrio_VidaTrabajo": eq,
        "Horas_Extra": hextra,
        "Capacitaciones_Ultimo_Anio": int(cap),
        "Anios_Empresa": int(anios_emp),
        "Num_Empresas_Previas": int(prev)
    }
    return fila

if modo == "Candidato externo":
    st.subheader("📝 Formulario — Candidato externo")
    st.caption("Solo info disponible en CV/entrevista. Lo interno se infiere automáticamente para alimentar el modelo.")
    defaults_ext = {
        "Anios_Empresa": 0,"Horas_Extra": "No","Capacitaciones_Ultimo_Anio": 0,
        "Satisfaccion_Puesto": "Bueno","Equilibrio_VidaTrabajo": "Bueno"
    }
    fila = common_fields(defaults=defaults_ext, modo="Candidato externo")
else:
    st.subheader("🧾 Formulario — Empleado actual")
    fila = common_fields(modo="Empleado actual")

st.markdown("---")

# ----------------------------
# 4.1) Guardado de logs (utilidad)  ⬅️ NUEVO
# ----------------------------
def append_log(fila_dict, proba, umbral, clasificacion, modo):
    """Agrega una fila al CSV de logs (crea encabezados si no existe)."""
    registro = dict(fila_dict)  # copia de inputs
    registro.update({
        "probabilidad": round(float(proba), 6),
        "umbral": round(float(umbral), 4),
        "clasificacion": clasificacion,
        "modo": modo,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    df_reg = pd.DataFrame([registro])
    new_file = not os.path.exists(LOG_PATH)
    df_reg.to_csv(LOG_PATH, mode="a", header=new_file, index=False, encoding="utf-8-sig", sep=";", 
        decimal=",")

# ----------------------------
# 5) Predicción
# ----------------------------
df_input = pd.DataFrame([fila])

if st.button("🔮 Predecir probabilidad de rotación"):
    try:
        proba = pipe.predict_proba(df_input)[:, 1][0]

        # Semáforo de riesgo
        if proba < 0.20:
            color, label = "#2ecc71", "Riesgo BAJO"
        elif proba < 0.50:
            color, label = "#f1c40f", "Riesgo MEDIO"
        else:
            color, label = "#e74c3c", "Riesgo ALTO"

        # ⬇️ Persistir registro
        append_log(fila, proba, umbral, label, modo)
        st.caption("✅ Predicción registrada en predicciones_log.csv")

        st.metric("Probabilidad estimada de rotación", f"{proba:.3f}")
        st.markdown(
            f"<div style='padding:10px;background-color:{color};border-radius:8px;color:white;text-align:center;font-size:18px;'>"
            f"Clasificación: {label}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error al predecir: {e}")

# ----------------------------
# 6) Factores de riesgo
# ----------------------------
def factores_riesgo(f, es_externo=False):
    tips = []
    if str(f["Horas_Extra"]).strip().lower() in ["sí","si","true","1"]:
        tips.append("Hace horas extra frecuentemente.")
    if f["Satisfaccion_Puesto"] in ["Muy bajo","Bajo"]:
        tips.append(f"Satisfacción '{f['Satisfaccion_Puesto']}'.")
    if f["Equilibrio_VidaTrabajo"] in ["Muy bajo","Bajo"]:
        tips.append(f"Equilibrio vida-trabajo '{f['Equilibrio_VidaTrabajo']}'.")
    if f["Ingreso_Mensual"] <= 1_000_000:
        tips.append("Ingreso mensual en rango bajo (≤ 1M ARS).")
    if f["Movilidad_Propia"] == "No" and float(f["Distancia_KM"]) >= 10:
        tips.append("Sin movilidad propia y distancia ≥ 10 km.")
    if not es_externo and int(f["Anios_Empresa"]) <= 1:
        tips.append("Poca antigüedad en la empresa (≤ 1 año).")
    if f["Departamento"] in ["Docencia","Marketing"] or f["Puesto"] in ["Becario/a de Investigación","Supervisor/a de Mantenimiento","Desarrollador/a"]:
        tips.append("Área/puesto con mayor contribución a la rotación en el modelo (histórico).")
    return tips

st.markdown("### 🔎 Posibles factores de riesgo (indicativos)")
es_externo = (modo == "Candidato externo")
rs = factores_riesgo(fila, es_externo=es_externo)
if rs:
    for t in rs:
        st.write(f"- {t}")
else:
    st.write("- No se detectan señales fuertes (usar la probabilidad del modelo).")

st.markdown("---")
st.caption("⚠️ Prototipo académico: no reemplaza evaluación integral de RRHH. Sirve como apoyo en priorización post-entrevista y screening de CVs.")
st.caption("Modelo: XGB (features derivadas v2) • AUC≈0.70 • Umbral fijo 0.20")  # ⬅️ NUEVO pie de página
