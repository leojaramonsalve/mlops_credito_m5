"""
model_monitoring.py
====================
Monitoreo de data drift y model drift para el pipeline MLOps de prediccion
de pago a tiempo.

Estrategia:
  - Split temporal por fecha_prestamo: el 70% mas antiguo es 'historico'
    (referencia) y el 30% mas reciente es 'actual' (produccion simulada).
  - Para cada variable, calcular drift entre historico y actual.

Metricas implementadas:
  - Kolmogorov-Smirnov (KS test)     : variables numericas continuas
  - Population Stability Index (PSI) : variables numericas (bineado)
  - Jensen-Shannon divergence (JS)   : distribuciones
  - Chi-cuadrado                     : variables categoricas

Umbrales de alerta (estandar industria):
  - KS p-value < 0.05    -> drift estadisticamente significativo
  - PSI < 0.10           -> sin drift
  - PSI 0.10 - 0.25      -> drift moderado (WARN)
  - PSI > 0.25           -> drift severo (ALERT)
  - JS divergence > 0.10 -> drift moderado
  - Chi2 p-value < 0.05  -> drift en categoricas

Adicional - Model drift:
  - Calcula la performance del mejor modelo en el periodo 'actual' y la
    compara con la performance en el periodo 'historico'. Si cae > 5% en
    ROC-AUC, dispara alerta.

Outputs (en data_processed/):
  - drift_metrics.csv         : una fila por variable con todas las metricas
  - prediction_log.csv        : log de predicciones del periodo actual
                                (simula lo que el deploy produciria)
  - drift_summary.json        : resumen ejecutivo + alertas
  - model_performance_drift.json : caida de metricas del modelo

Uso:
    python model_monitoring.py

Estado: V1.1.1 - Avance 3.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import (
    load_clean_dataset,
    build_derived_features,
    get_feature_columns,
    MODELS_DIR,
    DATA_PROC_DIR,
    COLUMNAS_LEAKAGE,
)


# ============================================================
# Configuracion - umbrales de alerta
# ============================================================

UMBRAL_KS_PVALUE = 0.05
UMBRAL_PSI_WARN = 0.10
UMBRAL_PSI_ALERT = 0.25
UMBRAL_JS = 0.10
UMBRAL_CHI2_PVALUE = 0.05
UMBRAL_CAIDA_AUC = 0.05   # caida absoluta de AUC para alerta

PROPORCION_HISTORICA = 0.70  # 70% mas antiguo = historico de referencia


# ============================================================
# Calculo de metricas de drift
# ============================================================

def calcular_ks(serie_ref, serie_actual):
    """Kolmogorov-Smirnov para variables numericas."""
    s_ref = serie_ref.dropna()
    s_act = serie_actual.dropna()
    if len(s_ref) < 10 or len(s_act) < 10:
        return None, None
    try:
        stat, pval = stats.ks_2samp(s_ref, s_act)
        return float(stat), float(pval)
    except Exception:
        return None, None


def calcular_psi(serie_ref, serie_actual, n_bins=10):
    """
    Population Stability Index (estandar de la industria financiera).
    PSI = sum( (actual% - ref%) * ln(actual% / ref%) )
    """
    s_ref = serie_ref.dropna()
    s_act = serie_actual.dropna()
    if len(s_ref) < 10 or len(s_act) < 10 or s_ref.nunique() < 2:
        return None

    try:
        # Bineado basado en cuantiles del referencia
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.unique(np.quantile(s_ref, quantiles))
        if len(bins) < 3:
            return None
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts, _ = np.histogram(s_ref, bins=bins)
        act_counts, _ = np.histogram(s_act, bins=bins)

        ref_pct = ref_counts / len(s_ref)
        act_pct = act_counts / len(s_act)

        # Evitar log(0): minimo 0.0001
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        act_pct = np.where(act_pct == 0, 0.0001, act_pct)

        psi = np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct))
        return float(psi)
    except Exception:
        return None


def calcular_js_divergence(serie_ref, serie_actual, n_bins=20):
    """
    Jensen-Shannon divergence: distancia simetrica entre distribuciones.
    Acotada en [0, 1]. JS > 0.1 ya es divergencia notable.
    """
    s_ref = serie_ref.dropna()
    s_act = serie_actual.dropna()
    if len(s_ref) < 10 or len(s_act) < 10:
        return None
    try:
        all_min = min(s_ref.min(), s_act.min())
        all_max = max(s_ref.max(), s_act.max())
        if all_min == all_max:
            return 0.0
        bins = np.linspace(all_min, all_max, n_bins + 1)
        p, _ = np.histogram(s_ref, bins=bins, density=True)
        q, _ = np.histogram(s_act, bins=bins, density=True)
        # Normalizar a probabilidades
        p = p / p.sum() if p.sum() > 0 else p
        q = q / q.sum() if q.sum() > 0 else q
        # Evitar zeros
        p = np.where(p == 0, 1e-10, p)
        q = np.where(q == 0, 1e-10, q)
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        return float(js)
    except Exception:
        return None


def calcular_chi2(serie_ref, serie_actual):
    """Chi-cuadrado para variables categoricas."""
    s_ref = serie_ref.dropna().astype(str)
    s_act = serie_actual.dropna().astype(str)
    if len(s_ref) < 10 or len(s_act) < 10:
        return None, None
    try:
        # Frecuencias por categoria
        cats = sorted(set(s_ref.unique()) | set(s_act.unique()))
        ref_counts = s_ref.value_counts().reindex(cats, fill_value=0).values
        act_counts = s_act.value_counts().reindex(cats, fill_value=0).values

        # Tabla de contingencia 2 x len(cats)
        tabla = np.array([ref_counts, act_counts])
        chi2_stat, pval, _, _ = stats.chi2_contingency(tabla)
        return float(chi2_stat), float(pval)
    except Exception:
        return None, None


# ============================================================
# Clasificacion de alertas
# ============================================================

def clasificar_alerta_psi(psi):
    if psi is None:
        return "N/A"
    if psi < UMBRAL_PSI_WARN:
        return "SIN_DRIFT"
    if psi < UMBRAL_PSI_ALERT:
        return "DRIFT_MODERADO"
    return "DRIFT_SEVERO"


def clasificar_alerta_global(ks_pval, psi, js, chi2_pval, tipo_var):
    """
    Combina los criterios para emitir una alerta global por variable.
    """
    alertas = []
    if tipo_var == "numerica":
        if ks_pval is not None and ks_pval < UMBRAL_KS_PVALUE:
            alertas.append("KS")
        if psi is not None and psi >= UMBRAL_PSI_WARN:
            alertas.append("PSI")
        if js is not None and js > UMBRAL_JS:
            alertas.append("JS")
    else:  # categorica
        if chi2_pval is not None and chi2_pval < UMBRAL_CHI2_PVALUE:
            alertas.append("CHI2")

    if not alertas:
        return "SIN_DRIFT", []
    if len(alertas) >= 2 or (psi is not None and psi >= UMBRAL_PSI_ALERT):
        return "DRIFT_SEVERO", alertas
    return "DRIFT_MODERADO", alertas


# ============================================================
# Split temporal historico vs actual
# ============================================================

def split_temporal(df, proporcion_historica=PROPORCION_HISTORICA):
    """
    Divide el dataset en historico (mas antiguo) y actual (mas reciente)
    usando fecha_prestamo. Esto simula el flujo productivo donde tenemos
    un modelo entrenado con datos pasados y lo aplicamos sobre datos nuevos.
    """
    df = df.sort_values('fecha_prestamo').reset_index(drop=True)
    n_hist = int(len(df) * proporcion_historica)
    df_historico = df.iloc[:n_hist].copy()
    df_actual = df.iloc[n_hist:].copy()
    return df_historico, df_actual


# ============================================================
# Generacion del log de predicciones (simula produccion)
# ============================================================

def generar_log_predicciones(df_actual, modelo, target_var="Pago_atiempo"):
    """
    Aplica el modelo entrenado sobre el periodo 'actual' y genera un log
    que el sistema productivo registraria normalmente.
    """
    feature_cols = [c for c in df_actual.columns
                    if c not in [target_var, 'fecha_prestamo'] and c not in COLUMNAS_LEAKAGE]
    X = df_actual[feature_cols]
    y_real = df_actual[target_var].values
    y_pred = modelo.predict(X)
    y_proba = modelo.predict_proba(X)[:, 1]

    log = pd.DataFrame({
        'fecha_prediccion': df_actual['fecha_prestamo'].values,
        'y_real': y_real,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'acertado': (y_real == y_pred).astype(int),
    })
    return log


# ============================================================
# Drift de performance del modelo
# ============================================================

def calcular_model_drift(df_historico, df_actual, modelo, target_var="Pago_atiempo"):
    """
    Compara la performance del modelo en historico vs actual.
    Si cae mucho, indica que el modelo se degrado.
    """
    feature_cols = [c for c in df_historico.columns
                    if c not in [target_var, 'fecha_prestamo'] and c not in COLUMNAS_LEAKAGE]

    perf = {}
    for nombre, df_split in [('historico', df_historico), ('actual', df_actual)]:
        X = df_split[feature_cols]
        y = df_split[target_var].values
        y_pred = modelo.predict(X)
        y_proba = modelo.predict_proba(X)[:, 1]
        perf[nombre] = {
            'n': len(df_split),
            'roc_auc': round(roc_auc_score(y, y_proba), 4),
            'f1_macro': round(f1_score(y, y_pred, average='macro'), 4),
            'f1_clase_0': round(f1_score(y, y_pred, pos_label=0), 4),
            'tasa_impago_real': round((y == 0).mean(), 4),
            'tasa_impago_predicha': round((y_pred == 0).mean(), 4),
        }

    caida_auc = perf['historico']['roc_auc'] - perf['actual']['roc_auc']
    alerta = "ALERTA" if abs(caida_auc) > UMBRAL_CAIDA_AUC else "OK"

    return {
        'performance_historico': perf['historico'],
        'performance_actual': perf['actual'],
        'caida_roc_auc': round(caida_auc, 4),
        'umbral_alerta': UMBRAL_CAIDA_AUC,
        'alerta_model_drift': alerta,
    }


# ============================================================
# Entrypoint
# ============================================================

def main():
    print("=" * 70)
    print("MODEL MONITORING - V1.1.1 (Avance 3)")
    print("=" * 70)
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos y agregar features derivadas
    print("\n[1/5] Cargando dataset y aplicando feature engineering...")
    df = load_clean_dataset()
    df = build_derived_features(df)
    print(f"  Dataset: {len(df):,} filas, fechas {df['fecha_prestamo'].min()} a {df['fecha_prestamo'].max()}")

    # 2. Split temporal
    print(f"\n[2/5] Split temporal {int(PROPORCION_HISTORICA*100)}/{int((1-PROPORCION_HISTORICA)*100)}...")
    df_hist, df_act = split_temporal(df, PROPORCION_HISTORICA)
    print(f"  Historico:  n={len(df_hist):,}  ({df_hist['fecha_prestamo'].min()} a {df_hist['fecha_prestamo'].max()})")
    print(f"  Actual:     n={len(df_act):,}  ({df_act['fecha_prestamo'].min()} a {df_act['fecha_prestamo'].max()})")

    # 3. Cargar modelo y generar log de predicciones
    print("\n[3/5] Cargando modelo y generando log de predicciones del periodo actual...")
    modelo = joblib.load(MODELS_DIR / "best_model.joblib")
    log = generar_log_predicciones(df_act, modelo)
    log.to_csv(DATA_PROC_DIR / "prediction_log.csv", index=False)
    print(f"  Log persistido: {DATA_PROC_DIR}/prediction_log.csv  ({len(log):,} predicciones)")
    print(f"  Tasa de acierto del modelo en periodo actual: {log['acertado'].mean():.4f}")

    # 4. Calcular metricas de drift por variable
    print("\n[4/5] Calculando metricas de drift por variable...")
    feature_cols = get_feature_columns(df)
    todas_num = feature_cols['numericas_continuas'] + feature_cols['numericas_discretas']
    todas_cat = feature_cols['categoricas_nominales'] + feature_cols['categoricas_ordinales']

    filas_drift = []
    for var in todas_num:
        ks_stat, ks_pval = calcular_ks(df_hist[var], df_act[var])
        psi = calcular_psi(df_hist[var], df_act[var])
        js = calcular_js_divergence(df_hist[var], df_act[var])
        alerta, criterios = clasificar_alerta_global(ks_pval, psi, js, None, "numerica")
        filas_drift.append({
            'variable': var,
            'tipo': 'numerica',
            'ks_statistic': round(ks_stat, 4) if ks_stat else None,
            'ks_pvalue': round(ks_pval, 4) if ks_pval else None,
            'psi': round(psi, 4) if psi else None,
            'psi_clasif': clasificar_alerta_psi(psi),
            'js_divergence': round(js, 4) if js else None,
            'chi2_statistic': None,
            'chi2_pvalue': None,
            'alerta': alerta,
            'criterios_disparados': ','.join(criterios) if criterios else '',
        })

    for var in todas_cat:
        chi2_stat, chi2_pval = calcular_chi2(df_hist[var], df_act[var])
        alerta, criterios = clasificar_alerta_global(None, None, None, chi2_pval, "categorica")
        filas_drift.append({
            'variable': var,
            'tipo': 'categorica',
            'ks_statistic': None,
            'ks_pvalue': None,
            'psi': None,
            'psi_clasif': 'N/A',
            'js_divergence': None,
            'chi2_statistic': round(chi2_stat, 4) if chi2_stat else None,
            'chi2_pvalue': round(chi2_pval, 4) if chi2_pval else None,
            'alerta': alerta,
            'criterios_disparados': ','.join(criterios) if criterios else '',
        })

    df_drift = pd.DataFrame(filas_drift)
    df_drift.to_csv(DATA_PROC_DIR / "drift_metrics.csv", index=False)
    print(f"  Drift metrics persistido: {DATA_PROC_DIR}/drift_metrics.csv  ({len(df_drift)} variables)")
    print()
    print("  Resumen de alertas:")
    print(df_drift.groupby('alerta').size().to_string())

    # 5. Model drift y resumen ejecutivo
    print("\n[5/5] Calculando model drift y generando resumen ejecutivo...")
    model_drift = calcular_model_drift(df_hist, df_act, modelo)

    n_severo = int((df_drift['alerta'] == 'DRIFT_SEVERO').sum())
    n_moderado = int((df_drift['alerta'] == 'DRIFT_MODERADO').sum())
    n_sin_drift = int((df_drift['alerta'] == 'SIN_DRIFT').sum())

    resumen = {
        'fecha_analisis': pd.Timestamp.now().isoformat(),
        'periodo_historico': {
            'desde': str(df_hist['fecha_prestamo'].min()),
            'hasta': str(df_hist['fecha_prestamo'].max()),
            'n_registros': len(df_hist),
        },
        'periodo_actual': {
            'desde': str(df_act['fecha_prestamo'].min()),
            'hasta': str(df_act['fecha_prestamo'].max()),
            'n_registros': len(df_act),
        },
        'data_drift': {
            'variables_con_drift_severo': n_severo,
            'variables_con_drift_moderado': n_moderado,
            'variables_sin_drift': n_sin_drift,
            'total_variables_analizadas': len(df_drift),
            'variables_severas': df_drift[df_drift['alerta'] == 'DRIFT_SEVERO']['variable'].tolist(),
        },
        'model_drift': model_drift,
        'umbrales_aplicados': {
            'KS_pvalue': UMBRAL_KS_PVALUE,
            'PSI_warn': UMBRAL_PSI_WARN,
            'PSI_alert': UMBRAL_PSI_ALERT,
            'JS_divergence': UMBRAL_JS,
            'Chi2_pvalue': UMBRAL_CHI2_PVALUE,
            'caida_AUC_alerta': UMBRAL_CAIDA_AUC,
        },
        'recomendacion': _generar_recomendacion(n_severo, n_moderado, model_drift),
    }

    with open(DATA_PROC_DIR / "drift_summary.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)

    with open(DATA_PROC_DIR / "model_performance_drift.json", "w", encoding="utf-8") as f:
        json.dump(model_drift, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Drift summary: {DATA_PROC_DIR}/drift_summary.json")
    print(f"  Model perf:    {DATA_PROC_DIR}/model_performance_drift.json")
    print()
    print(f"  Performance historico:  ROC-AUC = {model_drift['performance_historico']['roc_auc']}")
    print(f"  Performance actual:     ROC-AUC = {model_drift['performance_actual']['roc_auc']}")
    print(f"  Caida ROC-AUC:          {model_drift['caida_roc_auc']:+.4f}  ({model_drift['alerta_model_drift']})")
    print()
    print(f"  RECOMENDACION: {resumen['recomendacion']}")
    print()
    print("Listo para Avance 4 (FastAPI + Docker)")


def _generar_recomendacion(n_severo, n_moderado, model_drift):
    if model_drift['alerta_model_drift'] == 'ALERTA':
        return ("RETRAINING_URGENTE: la performance del modelo cayo mas alla del umbral. "
                "Revisar drift en features clave y reentrenar con datos del periodo actual.")
    if n_severo >= 3:
        return ("MONITOREO_REFORZADO: multiples variables con drift severo aunque la "
                "performance se mantiene. Programar reentrenamiento preventivo.")
    if n_severo > 0 or n_moderado >= 5:
        return ("OBSERVAR: drift detectado en variables aisladas. Documentar y monitorear "
                "evolucion. Considerar retraining trimestral.")
    return "OK: sin drift significativo. Continuar monitoreo regular."


if __name__ == "__main__":
    main()
