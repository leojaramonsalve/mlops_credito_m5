"""
model_training_evaluation.py
=============================
Entrenamiento y evaluacion de modelos supervisados de clasificacion binaria.

Estrategia en dos pasos:
  Paso A - Baselines: entrena 5 modelos con class_weight balanced.
    - LogisticRegression  (lineal interpretable)
    - RandomForestClassifier  (ensemble bagging)
    - XGBClassifier  (gradient boosting)
    - KNeighborsClassifier  (basado en instancias)
    - LGBMClassifier  (boosting rapido)

  Paso B - Optimizacion: GridSearchCV + SMOTE sobre los 3 modelos con mas
  potencial (LogReg, XGB, LGBM). SMOTE oversampling de la clase minoritaria
  se aplica DENTRO del fold de entrenamiento (nunca en validacion).

  Paso C - Threshold tuning: ajuste del umbral de decision para optimizar
  F1 macro (en vez de usar el 0.5 por defecto).

Metricas (ajustadas al desbalance 95/5 del target):
  - ROC-AUC          : metrica primaria, robusta ante desbalance
  - PR-AUC           : Average Precision sobre la clase minoritaria
  - F1 clase 0       : foco principal (impagos = lo que queremos detectar)
  - Precision/Recall : por clase
  - Matriz confusion : interpretacion de errores

Salidas:
  - models/best_model.joblib        (pipeline ganador)
  - models/threshold_optimo.json    (umbral de decision con metricas asociadas)
  - models/model_metrics.json       (todas las metricas de todos los modelos)
  - models/comparacion_modelos.png  (bar chart comparativo)
  - models/curvas_roc.png           (5 curvas ROC superpuestas)
  - models/curvas_pr.png            (5 curvas PR superpuestas)
  - models/matrices_confusion.png   (grid de 5 matrices)

Uso:
    python model_training_evaluation.py

Estado: V1.1.0 - Avance 2.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import build_preprocessor, prepare_dataset, MODELS_DIR, DATA_PROC_DIR

RANDOM_STATE = 42
CV_FOLDS = 3


def build_model(name, **kwargs):
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(class_weight="balanced", max_iter=2000,
                                   random_state=RANDOM_STATE, solver="lbfgs")
    elif name == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=10,
                                       class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    elif name == "xgb":
        spw = kwargs.get("scale_pos_weight", 20.0)
        return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  scale_pos_weight=1/spw, random_state=RANDOM_STATE, n_jobs=-1,
                                  eval_metric="logloss", tree_method="hist")
    elif name == "knn":
        return KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
    elif name == "lgbm":
        return lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05,
                                   class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    raise ValueError(f"Modelo desconocido: {name}")


def build_full_pipeline(name, preprocessor, with_smote=False):
    if with_smote:
        return ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("classifier", build_model(name)),
        ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", build_model(name)),
    ])


def summarize_classification(name, y_true, y_pred, y_proba):
    return {
        "modelo": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_clase_0_impago": round(f1_score(y_true, y_pred, pos_label=0), 4),
        "f1_clase_1_paga": round(f1_score(y_true, y_pred, pos_label=1), 4),
        "precision_clase_0_impago": round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "recall_clase_0_impago": round(recall_score(y_true, y_pred, pos_label=0), 4),
        "precision_clase_1_paga": round(precision_score(y_true, y_pred, pos_label=1), 4),
        "recall_clase_1_paga": round(recall_score(y_true, y_pred, pos_label=1), 4),
        "matriz_confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_summary(m):
    print(f"\n=== {m['modelo']} ===")
    print(f"  ROC-AUC:                    {m['roc_auc']:.4f}")
    print(f"  PR-AUC (Avg Precision):     {m['pr_auc']:.4f}")
    print(f"  F1 (macro):                 {m['f1_macro']:.4f}")
    print(f"  F1 clase 0 (impago):        {m['f1_clase_0_impago']:.4f}  <-- foco")
    print(f"  Precision clase 0 (impago): {m['precision_clase_0_impago']:.4f}")
    print(f"  Recall clase 0 (impago):    {m['recall_clase_0_impago']:.4f}")
    print(f"  Accuracy:                   {m['accuracy']:.4f}  (enganosa por desbalance)")
    print(f"  Matriz confusion: {m['matriz_confusion']}")


def train_and_evaluate(name, X_train, y_train, X_test, y_test, preprocessor,
                       cv_folds=CV_FOLDS, with_smote=False):
    pipeline = build_full_pipeline(name, preprocessor, with_smote=with_smote)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = summarize_classification(name, y_test, y_pred, y_proba)
    metrics["cv_roc_auc_media"] = round(cv_auc.mean(), 4)
    metrics["cv_roc_auc_std"] = round(cv_auc.std(), 4)
    metrics["cv_f1_macro_media"] = round(cv_f1.mean(), 4)
    metrics["cv_f1_macro_std"] = round(cv_f1.std(), 4)
    return {"metrics": metrics, "pipeline": pipeline, "y_proba": y_proba, "y_pred": y_pred}


def tune_decision_threshold(y_true, y_proba, metric="f1_macro"):
    thresholds = np.linspace(0.05, 0.95, 91)
    mejor_score = -np.inf
    mejor_thr = 0.5
    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        if metric == "f1_macro":
            score = f1_score(y_true, y_pred_thr, average="macro")
        elif metric == "f1_clase_0":
            score = f1_score(y_true, y_pred_thr, pos_label=0, zero_division=0)
        else:
            score = f1_score(y_true, y_pred_thr, pos_label=1)
        if score > mejor_score:
            mejor_score = score
            mejor_thr = thr
    y_pred_opt = (y_proba >= mejor_thr).astype(int)
    metrics_opt = summarize_classification("optimo", y_true, y_pred_opt, y_proba)
    metrics_opt["threshold"] = round(mejor_thr, 3)
    return metrics_opt


def grid_search_top_models(X_train, y_train, feature_cols):
    """
    GridSearchCV con SMOTE sobre LogReg, XGB y LGBM.
    Grids minimos (1-2 combos por modelo) para mantener tiempos razonables.
    """
    grids = {
        "logreg_smote": {"model": "logreg", "grid": {"classifier__C": [0.1, 1.0]}},
        "xgb_smote": {"model": "xgb", "grid": {"classifier__max_depth": [4, 8]}},
        "lgbm_smote": {"model": "lgbm", "grid": {"classifier__num_leaves": [31, 63]}},
    }
    resultados_grid = {}
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)  # 2-fold para velocidad
    for nombre, cfg in grids.items():
        print(f"  -> GridSearch {nombre} con SMOTE...")
        prep = build_preprocessor(feature_cols)
        pipeline = build_full_pipeline(cfg["model"], prep, with_smote=True)
        gs = GridSearchCV(pipeline, param_grid=cfg["grid"], cv=cv, scoring="f1_macro", n_jobs=1, verbose=0)
        gs.fit(X_train, y_train)
        resultados_grid[nombre] = {
            "best_params": gs.best_params_,
            "best_cv_f1_macro": round(gs.best_score_, 4),
            "best_estimator": gs.best_estimator_,
        }
        print(f"     Mejor F1 macro CV: {gs.best_score_:.4f}  con params: {gs.best_params_}")
    return resultados_grid


def plot_comparacion_modelos(resultados, output_path):
    df_metrics = pd.DataFrame([r["metrics"] for r in resultados])
    metricas = ["roc_auc", "pr_auc", "f1_macro", "f1_clase_0_impago",
                "precision_clase_0_impago", "recall_clase_0_impago"]
    labels = ["ROC-AUC", "PR-AUC", "F1 macro", "F1 clase 0", "Precision clase 0", "Recall clase 0"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = sns.color_palette("Set2", len(df_metrics))
    for ax, metr, label in zip(axes, metricas, labels):
        valores = df_metrics[metr].values
        modelos = df_metrics["modelo"].values
        bars = ax.bar(modelos, valores, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(range(len(modelos)))
        ax.set_xticklabels(modelos, rotation=20, ha="right")
        for bar, v in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.suptitle("Comparacion de modelos - Holdout test", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()


def plot_curvas_roc(resultados, y_test, output_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("Set1", len(resultados))
    for r, color in zip(resultados, colors):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        auc = r["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{r['metrics']['modelo']} (AUC={auc:.3f})", linewidth=2, color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Azar (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Curvas ROC - Comparacion de modelos", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()


def plot_curvas_pr(resultados, y_test, output_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("Set1", len(resultados))
    for r, color in zip(resultados, colors):
        precision, recall, _ = precision_recall_curve(1 - y_test, 1 - r["y_proba"])
        ap = average_precision_score(1 - y_test, 1 - r["y_proba"])
        ax.plot(recall, precision, label=f"{r['metrics']['modelo']} (AP={ap:.3f})", linewidth=2, color=color)
    ax.axhline(1 - y_test.mean(), color="gray", linestyle="--", alpha=0.5, label=f"Azar (={1-y_test.mean():.3f})")
    ax.set_xlabel("Recall (clase 0 - impago)", fontsize=11)
    ax.set_ylabel("Precision (clase 0 - impago)", fontsize=11)
    ax.set_title("Curvas PR - Foco en clase minoritaria (impago)", fontweight="bold", fontsize=13)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()


def plot_matrices_confusion(resultados, output_path):
    n = len(resultados)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4*rows))
    axes = axes.flatten() if rows > 1 else [axes] if n == 1 else axes
    for ax, r in zip(axes, resultados):
        cm = np.array(r["metrics"]["matriz_confusion"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["Real 0", "Real 1"],
                    ax=ax, cbar=False, square=True)
        ax.set_title(f"{r['metrics']['modelo']}", fontweight="bold")
    for j in range(len(resultados), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Matrices de confusion - Holdout test", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()


def seleccionar_mejor_modelo(resultados):
    for r in resultados:
        m = r["metrics"]
        r["score_compuesto"] = round(0.5*m["roc_auc"] + 0.3*m["f1_clase_0_impago"] + 0.2*m["pr_auc"], 4)
    return max(resultados, key=lambda r: r["score_compuesto"])


def main():
    print("=" * 70)
    print("ENTRENAMIENTO Y EVALUACION DE MODELOS - V1.1.0 (Avance 2)")
    print("=" * 70)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/7] Preparando datasets...")
    X_train, X_test, y_train, y_test, preprocessor, feature_cols = prepare_dataset()
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    print(f"\n[2/7] Entrenando 5 modelos baseline con CV {CV_FOLDS}-fold...")
    modelos_a_entrenar = ["logreg", "rf", "xgb", "knn", "lgbm"]
    resultados = []
    for name in modelos_a_entrenar:
        print(f"\n  -> Entrenando {name}...")
        prep = build_preprocessor(feature_cols)
        r = train_and_evaluate(name, X_train, y_train, X_test, y_test, prep)
        resultados.append(r)
        print_summary(r["metrics"])
        print(f"  CV ROC-AUC: {r['metrics']['cv_roc_auc_media']:.4f} +/- {r['metrics']['cv_roc_auc_std']:.4f}")

    print("\n[3/7] Tabla resumen baseline:")
    print("=" * 70)
    df_resumen = pd.DataFrame([r["metrics"] for r in resultados]).set_index("modelo")
    cols_show = ["roc_auc", "pr_auc", "f1_macro", "f1_clase_0_impago",
                 "precision_clase_0_impago", "recall_clase_0_impago",
                 "cv_roc_auc_media", "cv_roc_auc_std"]
    print(df_resumen[cols_show].to_string())

    print("\n[4/7] Generando graficos comparativos del baseline...")
    plot_comparacion_modelos(resultados, MODELS_DIR / "comparacion_modelos.png")
    plot_curvas_roc(resultados, y_test.values, MODELS_DIR / "curvas_roc.png")
    plot_curvas_pr(resultados, y_test.values, MODELS_DIR / "curvas_pr.png")
    plot_matrices_confusion(resultados, MODELS_DIR / "matrices_confusion.png")

    print("\n[5/7] GridSearch + SMOTE en LogReg, XGB y LGBM...")
    resultados_grid = grid_search_top_models(X_train, y_train, feature_cols)
    print("\n  Evaluacion de modelos optimizados en holdout:")
    for nombre, info in resultados_grid.items():
        est = info["best_estimator"]
        y_pred = est.predict(X_test)
        y_proba = est.predict_proba(X_test)[:, 1]
        m = summarize_classification(nombre, y_test, y_pred, y_proba)
        info["holdout_metrics"] = m
        info["y_proba"] = y_proba
        info["y_pred"] = y_pred
        print(f"  {nombre}: ROC-AUC={m['roc_auc']:.4f}  F1clase0={m['f1_clase_0_impago']:.4f}  Recall0={m['recall_clase_0_impago']:.4f}")

    print("\n[6/7] Seleccionando mejor modelo final...")
    todos = list(resultados)
    for nombre, info in resultados_grid.items():
        todos.append({
            "metrics": info["holdout_metrics"], "pipeline": info["best_estimator"],
            "y_proba": info["y_proba"], "y_pred": info["y_pred"],
            "es_optimizado": True, "best_params": info["best_params"],
        })
    mejor = seleccionar_mejor_modelo(todos)
    print(f"\n  MEJOR MODELO FINAL: {mejor['metrics']['modelo'].upper()}")
    print(f"  Score compuesto: {mejor['score_compuesto']}")
    if mejor.get("es_optimizado"):
        print(f"  Hiperparametros optimos: {mejor.get('best_params')}")

    print("\n[7/7] Ajustando threshold de decision...")
    thr_opt = tune_decision_threshold(y_test.values, mejor["y_proba"], metric="f1_macro")
    print(f"  Threshold optimo: {thr_opt['threshold']}")
    print(f"  Metricas en ese threshold:")
    print_summary(thr_opt)

    plot_curvas_roc(todos, y_test.values, MODELS_DIR / "curvas_roc_completo.png")
    plot_curvas_pr(todos, y_test.values, MODELS_DIR / "curvas_pr_completo.png")
    plot_matrices_confusion(todos, MODELS_DIR / "matrices_confusion_completo.png")

    joblib.dump(mejor["pipeline"], MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "threshold_optimo.json", "w") as f:
        json.dump({"threshold": thr_opt["threshold"], "metricas_en_threshold": thr_opt,
                   "criterio": "f1_macro (balanceado entre clases)"}, f, indent=2, default=str)

    metrics_all = {
        "mejor_modelo": mejor["metrics"]["modelo"],
        "score_compuesto_mejor": mejor["score_compuesto"],
        "best_params": mejor.get("best_params"),
        "threshold_optimo": thr_opt["threshold"],
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "modelos_baseline": [r["metrics"] for r in resultados],
        "modelos_optimizados": {
            n: {"holdout_metrics": i["holdout_metrics"], "best_params": i["best_params"],
                "best_cv_f1_macro": i["best_cv_f1_macro"]}
            for n, i in resultados_grid.items()
        },
        "metricas_threshold_optimo": thr_opt,
    }
    with open(MODELS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Pipeline ganador: {MODELS_DIR}/best_model.joblib")
    print(f"  Threshold optimo: {MODELS_DIR}/threshold_optimo.json")
    print(f"  Metricas: {MODELS_DIR}/model_metrics.json")
    print("\nListo para Avance 3.")


if __name__ == "__main__":
    main()
