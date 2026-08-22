import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Estimación de Constantes de Van der Waals",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Estimación de Constantes de Van der Waals ($a$ y $b$)")
st.markdown("""
Esta aplicación permite calcular los parámetros de **Van der Waals** para un gas o mezcla de gases recolectados 
sobre agua mediante desplazamiento en probeta invertida (reacción $\\text{NaHCO}_3 + \\text{CH}_3\\text{COOH}$).
""")

# ==========================================
# PANEL LATERAL: CONDICIONES AMBIENTALES
# ==========================================
st.sidebar.header("⚙️ Condiciones Experimentales")

T_C = st.sidebar.number_input("Temperatura del laboratorio (°C)", value=25.0, step=0.1, format="%.1f")
P_atm = st.sidebar.number_input("Presión atmosférica local (atm)", value=0.9930, step=0.0010, format="%.4f")
P_vap_H2O = st.sidebar.number_input("Presión de vapor de H₂O a T (atm)", value=0.0313, step=0.0005, format="%.4f")

# Corrección de presión del gas seco
P_gas = P_atm - P_vap_H2O
T_K = T_C + 273.15
R = 0.0820574  # L*atm / (mol*K)
RT = R * T_K

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Presión corregida ($P_{{gas}}$):** `{P_gas:.4f} atm`")
st.sidebar.markdown(f"**Temperatura absoluta ($T$):** `{T_K:.2f} K`")

# ==========================================
# ENTRADA DE DATOS EXPERIMENTALES
# ==========================================
st.subheader("📋 Datos Experimentales de los Ensayos")
st.caption("Ingresa al menos 2 ensayos. Si agregas más filas, el sistema aplicará un ajuste por Mínimos Cuadrados.")

# Datos iniciales precargados
df_inicial = pd.DataFrame({
    "Ensayo": [1, 2, 3],
    "n (mol)": [0.0520, 0.0680, 0.0800],
    "V (mL)": [1325.0, 1720.0, 2020.0]
})

df_editado = st.data_editor(
    df_inicial,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True
)

# Validar datos
datos_validos = df_editado.dropna()
datos_validos = datos_validos[(datos_validos["n (mol)"] > 0) & (datos_validos["V (mL)"] > 0)]

if len(datos_validos) < 2:
    st.error("⚠️ Debes ingresar al menos 2 filas con valores mayores a cero para resolver el sistema.")
    st.stop()

# Procesamiento de volúmenes y volúmenes molares
datos_validos["V (L)"] = datos_validos["V (mL)"] / 1000.0
datos_validos["Vm (L/mol)"] = datos_validos["V (L)"] / datos_validos["n (mol)"]

# ==========================================
# CÁLCULO DE CONSTANTES (Aproximación Lineal)
# ==========================================
# Ecuación: (1 / Vm) * a - P * b = RT - P * Vm
A = np.column_stack([
    1.0 / datos_validos["Vm (L/mol)"],
    -P_gas * np.ones(len(datos_validos))
])

Y = RT - P_gas * datos_validos["Vm (L/mol)"].values

# Solución matricial (Exacta para N=2, Mínimos Cuadrados para N>2)
solucion, residuos, rank, s = np.linalg.lstsq(A, Y, rcond=None)
a_calculado, b_calculado = solucion[0], solucion[1]

# ==========================================
# MÉTRICAS Y RESULTADOS
# ==========================================
st.markdown("---")
col_res1, col_res2, col_res3 = st.columns(3)

col_res1.metric(
    label="Parámetro 'a' (Atracción intermolecular)",
    value=f"{a_calculado:.4f} L²·atm/mol²",
    help="Valores mayores indican mayores fuerzas atractivas (típico de mezclas con vapor/CO2)."
)

col_res2.metric(
    label="Parámetro 'b' (Covolumen molar)",
    value=f"{b_calculado:.4f} L/mol",
    help="Representa el volumen efectivo ocupado por un mol de moléculas del gas."
)

# Factor de compresibilidad promedio Z = P * Vm / (RT)
Z_prom = np.mean((P_gas * datos_validos["Vm (L/mol)"]) / RT)
col_res3.metric(
    label="Factor Z Promedio",
    value=f"{Z_prom:.4f}",
    delta=f"{Z_prom - 1.0:.4f} respecto a Gas Ideal",
    delta_color="inverse"
)

# ==========================================
# VISUALIZACIÓN GRÁFICA
# ==========================================
st.markdown("---")
st.subheader("📈 Comportamiento Gráfico: Gas Real vs Gas Ideal")

col_graf, col_tabla = st.columns([3, 2])

with col_graf:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Rango de volúmenes molares para simulación de curvas
    Vm_min = max(0.05, b_calculado * 1.05 if b_calculado > 0 else 0.05, datos_validos["Vm (L/mol)"].min() * 0.7)
    Vm_max = datos_validos["Vm (L/mol)"].max() * 1.3
    Vm_teorico = np.linspace(Vm_min, Vm_max, 300)
    
    # Isotermas calculadas
    P_ideal = (RT) / Vm_teorico
    P_vdw = (RT / (Vm_teorico - b_calculado)) - (a_calculado / (Vm_teorico**2))
    
    # Curvas
    ax.plot(Vm_teorico, P_ideal, label="Gas Ideal ($P = RT / \\bar{V}$)", color="#1f77b4", linestyle="--", linewidth=2)
    ax.plot(Vm_teorico, P_vdw, label=f"Van der Waals Ajustado ($a={a_calculado:.2f}, b={b_calculado:.3f}$)", color="#d62728", linewidth=2.5)
    
    # Puntos experimentales
    ax.scatter(
        datos_validos["Vm (L/mol)"], 
        [P_gas] * len(datos_validos), 
        color="#2ca02c", 
        s=90, 
        zorder=5, 
        edgecolors="black", 
        label="Datos de Laboratorio"
    )
    
    ax.set_title(f"Isoterma a T = {T_C} °C", fontsize=13, pad=10)
    ax.set_xlabel("Volumen Molar $\\bar{V}$ (L/mol)", fontsize=11)
    ax.set_ylabel("Presión (atm)", fontsize=11)
    ax.set_ylim(bottom=max(0, P_gas * 0.5), top=P_gas * 1.5)
    ax.axhline(P_gas, color="gray", linestyle=":", alpha=0.6, label=f"$P_{{gas}}$ = {P_gas:.4f} atm")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    st.pyplot(fig)

with col_tabla:
    st.markdown("**Tabla Resumen de Ensayos:**")
    tabla_resumen = datos_validos[["Ensayo", "n (mol)", "V (mL)", "Vm (L/mol)"]].copy()
    tabla_resumen["Z"] = (P_gas * tabla_resumen["Vm (L/mol)"]) / RT
    st.dataframe(tabla_resumen.style.format({
        "n (mol)": "{:.4f}",
        "V (mL)": "{:.1f}",
        "Vm (L/mol)": "{:.4f}",
        "Z": "{:.4f}"
    }), use_container_width=True, hide_index=True)

    st.info("""
    **Interpretación química:**
    - Los gases reales con atracción molecular tienen $a > 0$.
    - En mezclas con $\\text{CO}_2$, vapor de $\\text{H}_2\\text{O}$ y aire, los valores difieren de una sustancia pura debido al efecto multicomponente.
    """)
