"""
streamlit_app/app.py
=====================
Dashboard de monitoreo de data drift y model drift para el pipeline MLOps
de prediccion de pago a tiempo.

Secciones:
  1. Resumen ejecutivo: KPIs clave + estado global del sistema
  2. Performance del modelo: comparacion historico vs actual
  3. Drift por variable: tabla con semaforo + filtros
  4. Distribuciones: comparacion visual hist vs actual de cualquier variable
  5. Evolucion temporal: tasa de impagos predicha en el tiempo
  6. Recomendaciones automaticas

Para correr:
    streamlit run streamlit_app/app.py

Estado: V1.1.1 - Avance 3.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Rutas (relativas al root del repo)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_processed"
MODELS_DIR = ROOT / "models"

# Configuracion de pagina
st.set_page_config(
    page_title="Drift Monitor - MLOps Credito",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Carga de datos (cached)
# ============================================================

@st.cache_data
def cargar_drift_metrics():
    return pd.read_csv(DATA_DIR / "drift_metrics.csv")


@st.cache_data
def cargar_drift_summary():
    with open(DATA_DIR / "drift_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_prediction_log():
    log = pd.read_csv(DATA_DIR / "prediction_log.csv")
    log['fecha_prediccion'] = pd.to_datetime(log['fecha_prediccion'])
    return log


@st.cache_data
def cargar_dataset_limpio():
    df = pd.read_parquet(DATA_DIR / "dataset_limpio.parquet")
    return df


@st.cache_data
def cargar_model_metrics():
    with open(MODELS_DIR / "model_metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Componentes UI auxiliares
# ============================================================

def semaforo(alerta):
    """Devuelve emoji/color segun nivel de alerta."""
    return {
        "SIN_DRIFT": ":large_green_circle: SIN DRIFT",
        "DRIFT_MODERADO": ":large_yellow_circle: MODERADO",
        "DRIFT_SEVERO": ":red_circle: SEVERO",
        "N/A": ":white_circle: N/A",
    }.get(alerta, alerta)


def kpi_card(label, value, delta=None, delta_color="normal", help_text=None):
    st.metric(label, value, delta=delta, delta_color=delta_color, help=help_text)


# ============================================================
# UI Principal
# ============================================================

def main():
    # ---- Sidebar ----
    st.sidebar.title(":bar_chart: Drift Monitor")
    st.sidebar.markdown("**MLOps - Prediccion de Pago a Tiempo**")
    st.sidebar.markdown("Modulo 5 - Henry")
    st.sidebar.markdown("---")

    seccion = st.sidebar.radio(
        "Seccion",
        [
            ":house: Resumen ejecutivo",
            ":chart_with_upwards_trend: Performance del modelo",
            ":bar_chart: Drift por variable",
            ":mag: Distribuciones",
            ":calendar: Evolucion temporal",
            ":bulb: Recomendaciones",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Fuente: drift_metrics.csv, drift_summary.json, prediction_log.csv")

    # ---- Cargar artifacts ----
    try:
        df_drift = cargar_drift_metrics()
        summary = cargar_drift_summary()
        log = cargar_prediction_log()
        df_limpio = cargar_dataset_limpio()
    except FileNotFoundError as e:
        st.error(f"No se encontraron artifacts del monitoring. Corre primero `python model_monitoring.py`. Error: {e}")
        st.stop()

    # ---- Header global ----
    st.title("Dashboard de Monitoreo - Pipeline MLOps Credito")
    st.caption(f"Ultimo analisis: {summary['fecha_analisis']}")

    # ============================================================
    # SECCION 1: Resumen ejecutivo
    # ============================================================
    if seccion.startswith(":house:"):
        st.header("Resumen ejecutivo")

        # KPIs principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_total = summary['data_drift']['total_variables_analizadas']
            kpi_card("Variables analizadas", n_total)
        with col2:
            n_severo = summary['data_drift']['variables_con_drift_severo']
            kpi_card("Drift severo", n_severo,
                     delta=f"{n_severo/n_total*100:.0f}% del total",
                     delta_color="inverse")
        with col3:
            roc_hist = summary['model_drift']['performance_historico']['roc_auc']
            roc_act = summary['model_drift']['performance_actual']['roc_auc']
            caida = summary['model_drift']['caida_roc_auc']
            kpi_card("ROC-AUC actual", f"{roc_act:.3f}",
                     delta=f"{-caida:+.3f} vs historico",
                     delta_color="inverse" if caida > 0 else "normal")
        with col4:
            alerta_mdrift = summary['model_drift']['alerta_model_drift']
            color_mdrift = "OK" if alerta_mdrift == "OK" else "DEGRADADO"
            kpi_card("Estado del modelo", alerta_mdrift,
                     delta=color_mdrift)

        st.markdown("---")

        # Recomendacion en banner
        recomendacion = summary['recomendacion']
        if recomendacion.startswith("OK"):
            st.success(f":white_check_mark: {recomendacion}")
        elif recomendacion.startswith("RETRAINING_URGENTE"):
            st.error(f":rotating_light: {recomendacion}")
        elif recomendacion.startswith("MONITOREO_REFORZADO"):
            st.warning(f":warning: {recomendacion}")
        else:
            st.info(f":information_source: {recomendacion}")

        # Periodos
        st.subheader("Periodos comparados")
        col_h, col_a = st.columns(2)
        with col_h:
            st.markdown("**Historico (referencia)**")
            ph = summary['periodo_historico']
            st.markdown(f"- Desde: `{ph['desde']}`")
            st.markdown(f"- Hasta: `{ph['hasta']}`")
            st.markdown(f"- Registros: `{ph['n_registros']:,}`")
        with col_a:
            st.markdown("**Actual (produccion simulada)**")
            pa = summary['periodo_actual']
            st.markdown(f"- Desde: `{pa['desde']}`")
            st.markdown(f"- Hasta: `{pa['hasta']}`")
            st.markdown(f"- Registros: `{pa['n_registros']:,}`")

        # Distribucion de alertas - barras
        st.subheader("Distribucion de alertas")
        conteo_alertas = df_drift['alerta'].value_counts()
        col_b1, col_b2 = st.columns([3, 2])
        with col_b1:
            fig, ax = plt.subplots(figsize=(8, 3))
            colores = {'SIN_DRIFT': '#2a9d8f', 'DRIFT_MODERADO': '#f4a261', 'DRIFT_SEVERO': '#e63946'}
            cols = [colores.get(a, 'gray') for a in conteo_alertas.index]
            ax.barh(conteo_alertas.index, conteo_alertas.values, color=cols, edgecolor='white')
            for i, (idx, v) in enumerate(conteo_alertas.items()):
                ax.text(v + 0.1, i, f'{v} variables', va='center', fontsize=11, fontweight='bold')
            ax.set_xlabel('Cantidad de variables')
            ax.set_title('Variables segun nivel de drift detectado')
            ax.set_xlim(0, conteo_alertas.max() * 1.2)
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
        with col_b2:
            st.markdown("**Variables con drift severo:**")
            severas = summary['data_drift'].get('variables_severas', [])
            if severas:
                for v in severas:
                    st.markdown(f"- :red_circle: `{v}`")
            else:
                st.success("Ninguna variable con drift severo :white_check_mark:")

    # ============================================================
    # SECCION 2: Performance del modelo
    # ============================================================
    elif seccion.startswith(":chart_with_upwards_trend:"):
        st.header("Performance del modelo")

        perf_h = summary['model_drift']['performance_historico']
        perf_a = summary['model_drift']['performance_actual']

        st.subheader("Comparacion historico vs actual")

        comparacion = pd.DataFrame({
            'Metrica': ['n', 'ROC-AUC', 'F1 macro', 'F1 clase 0 (impago)', 'Tasa impago real', 'Tasa impago predicha'],
            'Historico': [perf_h['n'], perf_h['roc_auc'], perf_h['f1_macro'], perf_h['f1_clase_0'],
                          perf_h['tasa_impago_real'], perf_h['tasa_impago_predicha']],
            'Actual': [perf_a['n'], perf_a['roc_auc'], perf_a['f1_macro'], perf_a['f1_clase_0'],
                       perf_a['tasa_impago_real'], perf_a['tasa_impago_predicha']],
        })
        st.dataframe(comparacion, use_container_width=True, hide_index=True)

        # Gráfico de barras comparativo
        fig, ax = plt.subplots(figsize=(10, 4))
        metricas = ['roc_auc', 'f1_macro', 'f1_clase_0']
        labels = ['ROC-AUC', 'F1 macro', 'F1 clase 0']
        x = np.arange(len(metricas))
        width = 0.35
        ax.bar(x - width/2, [perf_h[m] for m in metricas], width, label='Historico', color='#264653')
        ax.bar(x + width/2, [perf_a[m] for m in metricas], width, label='Actual', color='#e76f51')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_title('Performance del modelo: historico vs actual', fontweight='bold')
        ax.set_ylim(0, 1)
        for i, (h, a) in enumerate(zip([perf_h[m] for m in metricas], [perf_a[m] for m in metricas])):
            ax.text(i - width/2, h + 0.02, f'{h:.3f}', ha='center', fontsize=9, fontweight='bold')
            ax.text(i + width/2, a + 0.02, f'{a:.3f}', ha='center', fontsize=9, fontweight='bold')
        st.pyplot(fig)

        # Alerta de model drift
        caida = summary['model_drift']['caida_roc_auc']
        umbral = summary['model_drift']['umbral_alerta']
        st.markdown("---")
        st.subheader("Diagnostico de model drift")
        if abs(caida) > umbral:
            st.error(f":rotating_light: Caida de ROC-AUC de {caida:+.4f} excede el umbral configurado ({umbral}). Sugerido: reentrenar.")
        else:
            st.success(f":white_check_mark: Caida de ROC-AUC de {caida:+.4f} esta dentro del umbral aceptable ({umbral}).")

    # ============================================================
    # SECCION 3: Drift por variable
    # ============================================================
    elif seccion.startswith(":bar_chart:"):
        st.header("Drift por variable - tabla detallada")

        col1, col2 = st.columns([1, 3])
        with col1:
            filtro_alerta = st.multiselect(
                "Filtrar por nivel de alerta",
                options=['SIN_DRIFT', 'DRIFT_MODERADO', 'DRIFT_SEVERO'],
                default=['DRIFT_MODERADO', 'DRIFT_SEVERO'],
            )
            filtro_tipo = st.multiselect(
                "Filtrar por tipo",
                options=['numerica', 'categorica'],
                default=['numerica', 'categorica'],
            )

        df_filtrado = df_drift[
            df_drift['alerta'].isin(filtro_alerta) & df_drift['tipo'].isin(filtro_tipo)
        ].copy()
        df_filtrado['semaforo'] = df_filtrado['alerta'].apply(semaforo)

        cols_show = ['semaforo', 'variable', 'tipo', 'ks_pvalue', 'psi', 'psi_clasif',
                     'js_divergence', 'chi2_pvalue', 'criterios_disparados']
        st.dataframe(
            df_filtrado[cols_show].sort_values('alerta', ascending=False),
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        st.markdown("---")
        st.subheader("Top variables por PSI (variables numericas)")
        df_psi = df_drift[df_drift['tipo'] == 'numerica'].dropna(subset=['psi']).copy()
        df_psi = df_psi.sort_values('psi', ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 5))
        colores_psi = []
        for p in df_psi['psi']:
            if p < 0.1: colores_psi.append('#2a9d8f')
            elif p < 0.25: colores_psi.append('#f4a261')
            else: colores_psi.append('#e63946')
        ax.barh(df_psi['variable'], df_psi['psi'], color=colores_psi, edgecolor='white')
        ax.axvline(0.10, color='orange', linestyle='--', label='Umbral WARN (0.10)')
        ax.axvline(0.25, color='red', linestyle='--', label='Umbral ALERT (0.25)')
        ax.set_xlabel('PSI (Population Stability Index)')
        ax.set_title('Top 15 variables por PSI')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        for i, (idx, row) in enumerate(df_psi.iterrows()):
            ax.text(row['psi'] + 0.005, i, f'{row["psi"]:.3f}', va='center', fontsize=9)
        st.pyplot(fig)

    # ============================================================
    # SECCION 4: Distribuciones
    # ============================================================
    elif seccion.startswith(":mag:"):
        st.header("Comparacion de distribuciones: historico vs actual")

        # Selector de variable
        variables_disponibles = df_drift['variable'].tolist()
        variable = st.selectbox("Seleccionar variable", variables_disponibles)

        # Recrear el split temporal para visualizar
        df_limpio_sorted = df_limpio.sort_values('fecha_prestamo').reset_index(drop=True)
        n_hist = int(len(df_limpio_sorted) * 0.70)
        df_hist_v = df_limpio_sorted.iloc[:n_hist]
        df_act_v = df_limpio_sorted.iloc[n_hist:]

        info_var = df_drift[df_drift['variable'] == variable].iloc[0]

        # Mostrar metricas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tipo", info_var['tipo'])
        with col2:
            st.metric("Alerta", info_var['alerta'])
        with col3:
            if info_var['tipo'] == 'numerica':
                st.metric("PSI", f"{info_var['psi']:.4f}" if pd.notna(info_var['psi']) else "N/A")
            else:
                st.metric("Chi² p-value", f"{info_var['chi2_pvalue']:.4f}" if pd.notna(info_var['chi2_pvalue']) else "N/A")
        with col4:
            if info_var['tipo'] == 'numerica':
                st.metric("KS p-value", f"{info_var['ks_pvalue']:.4f}" if pd.notna(info_var['ks_pvalue']) else "N/A")
            else:
                st.metric("JS", "N/A")

        # Visualizacion
        if info_var['tipo'] == 'numerica' and variable in df_limpio.columns:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            # Histogramas superpuestos
            axes[0].hist(df_hist_v[variable].dropna(), bins=30, alpha=0.5, label='Historico', color='#264653', density=True)
            axes[0].hist(df_act_v[variable].dropna(), bins=30, alpha=0.5, label='Actual', color='#e76f51', density=True)
            axes[0].set_title(f'Distribucion de {variable}')
            axes[0].set_xlabel(variable)
            axes[0].set_ylabel('Densidad')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            # Boxplot comparativo
            axes[1].boxplot([df_hist_v[variable].dropna(), df_act_v[variable].dropna()],
                            labels=['Historico', 'Actual'], patch_artist=True,
                            boxprops=dict(facecolor='#2a9d8f', alpha=0.6))
            axes[1].set_title(f'Boxplot comparativo')
            axes[1].grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        elif info_var['tipo'] == 'categorica' and variable in df_limpio.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            hist_counts = df_hist_v[variable].value_counts(normalize=True).sort_index()
            act_counts = df_act_v[variable].value_counts(normalize=True).sort_index()
            cats = sorted(set(hist_counts.index) | set(act_counts.index))
            hist_pct = [hist_counts.get(c, 0) * 100 for c in cats]
            act_pct = [act_counts.get(c, 0) * 100 for c in cats]
            x = np.arange(len(cats))
            width = 0.35
            ax.bar(x - width/2, hist_pct, width, label='Historico', color='#264653')
            ax.bar(x + width/2, act_pct, width, label='Actual', color='#e76f51')
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in cats], rotation=45, ha='right')
            ax.set_ylabel('% del periodo')
            ax.set_title(f'Distribucion de {variable}')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)

    # ============================================================
    # SECCION 5: Evolucion temporal
    # ============================================================
    elif seccion.startswith(":calendar:"):
        st.header("Evolucion temporal de predicciones")

        # Agrupar log por mes
        log['mes'] = log['fecha_prediccion'].dt.to_period('M').dt.to_timestamp()
        agg = log.groupby('mes').agg(
            n_predicciones=('y_pred', 'count'),
            tasa_impago_predicha=('y_pred', lambda x: (x == 0).mean()),
            tasa_impago_real=('y_real', lambda x: (x == 0).mean()),
            accuracy_mes=('acertado', 'mean'),
            score_promedio=('y_proba', 'mean'),
        ).reset_index()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicciones en el log", f"{len(log):,}")
        with col2:
            st.metric("Accuracy global", f"{log['acertado'].mean():.4f}")

        st.subheader("Tasa de impagos: real vs predicha (mensual)")
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(agg['mes'], agg['tasa_impago_real'] * 100, 'o-', label='Tasa real', color='#264653', linewidth=2)
        ax.plot(agg['mes'], agg['tasa_impago_predicha'] * 100, 's--', label='Tasa predicha', color='#e76f51', linewidth=2)
        ax.set_xlabel('Mes')
        ax.set_ylabel('% impagos')
        ax.set_title('Evolucion mensual de tasa de impagos')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Accuracy del modelo en el tiempo")
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(agg['mes'], agg['accuracy_mes'] * 100, 'o-', color='#2a9d8f', linewidth=2)
        ax.axhline(95, color='gray', linestyle='--', alpha=0.5, label='Baseline trivial (95%)')
        ax.set_ylabel('Accuracy %')
        ax.set_xlabel('Mes')
        ax.set_title('Accuracy mensual del modelo')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Tabla resumen mensual")
        st.dataframe(agg, use_container_width=True, hide_index=True)

    # ============================================================
    # SECCION 6: Recomendaciones
    # ============================================================
    elif seccion.startswith(":bulb:"):
        st.header("Recomendaciones automaticas")

        recomendacion = summary['recomendacion']
        st.subheader("Diagnostico del sistema")
        if recomendacion.startswith("OK"):
            st.success(f":white_check_mark: {recomendacion}")
        elif recomendacion.startswith("RETRAINING_URGENTE"):
            st.error(f":rotating_light: {recomendacion}")
        elif recomendacion.startswith("MONITOREO_REFORZADO"):
            st.warning(f":warning: {recomendacion}")
        else:
            st.info(f":information_source: {recomendacion}")

        st.subheader("Detalle de criterios")
        criterios = {
            "Variables con drift severo": summary['data_drift']['variables_con_drift_severo'],
            "Variables con drift moderado": summary['data_drift']['variables_con_drift_moderado'],
            "Variables sin drift": summary['data_drift']['variables_sin_drift'],
            "Caida ROC-AUC del modelo": summary['model_drift']['caida_roc_auc'],
            "Estado del modelo": summary['model_drift']['alerta_model_drift'],
        }
        for k, v in criterios.items():
            st.markdown(f"- **{k}**: `{v}`")

        st.markdown("---")
        st.subheader("Umbrales aplicados")
        umbrales = summary['umbrales_aplicados']
        st.json(umbrales)

        st.markdown("---")
        st.subheader("Acciones sugeridas segun nivel de alerta")
        st.markdown("""
**Si OK** → Continuar monitoreo regular. Revisar mensualmente.

**Si OBSERVAR** → Documentar drift en variables aisladas. Programar retraining trimestral.

**Si MONITOREO_REFORZADO** → Multiples drifts pero modelo estable. Reentrenar preventivamente
en proximos 30 dias. Revisar features especificas con drift severo.

**Si RETRAINING_URGENTE** → Performance degradada. Reentrenar AHORA con datos del periodo actual.
Validar pipeline de features.
        """)


if __name__ == "__main__":
    main()
