# Pipeline MLOps - Prediccion de Pago a Tiempo

**Proyecto Integrador - Modulo 5 - Henry**  
**Rol simulado:** Cientifico de Datos Jr Advanced - Sector financiero  
**Variable objetivo:** `Pago_atiempo` (binario: 1 = paga, 0 = impago)  
**Dataset:** 10.763 registros historicos de creditos, 23 variables originales

---

## Caso de negocio

Una empresa financiera necesita anticipar el comportamiento crediticio de **nuevos clientes** antes de otorgar prestamos. El equipo de Datos y Analitica desarrolla un modelo predictivo que se integra en el flujo operativo: recibe la solicitud, calcula la probabilidad de pago a tiempo y devuelve un score que alimenta la decision crediticia.

El modelo opera bajo principios **MLOps**: estructura de carpetas estricta, versionamiento con tres ramas (developer/certification/master), pipelines automatizados de feature engineering, monitoreo de drift continuo y despliegue dockerizado de la API.

---

## Resumen de hallazgos clave

### Calidad de datos
- Dataset balanceado en variables explicativas, sin nulos disfrazados ni variables irrelevantes.
- **Nulos correlacionados** (~14% de las filas) en `promedio_ingresos_datacredito` y `tendencia_ingresos`: clientes sin historial en datacredito. Tratamiento: NO imputar con media, sino como categoria informativa.
- **Desbalance severo del target**: 95.25% pago a tiempo / 4.75% impago. Implica metricas robustas (ROC-AUC, F1 clase minoritaria) y tecnicas de manejo de clases (class_weight, SMOTE).

### Leakage detectado (decisiones criticas)
- **Grupo 1 (EDA):** las variables de saldo (`saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor`) tienen AUC univariada > 0.90. Representan estado post-otorgamiento del prestamo, no disponibles al evaluar nuevos clientes. **Excluidas.**
- **Grupo 2 (post-modelado):** la variable `puntaje` (score interno) tenia AUC univariada = 1.0 y dominaba 74% de la importancia del Random Forest. Hipotesis: score post-hoc o output de modelo previo entrenado sobre los mismos datos. **Excluida.**
- `puntaje_datacredito` (score externo del buro) se mantiene: AUC 0.62, legitima.

### Modelado
Se entrenaron 5 modelos baseline + 3 optimizados con SMOTE + GridSearch:

| Modelo | ROC-AUC | F1 clase 0 | Notas |
|--------|---------|------------|-------|
| **LGBM baseline** | 0.6482 | **0.1860** | Mejor balance final (seleccionado) |
| LogReg | 0.6613 | 0.1345 | Mejor recall |
| XGBoost | 0.6578 | 0.1606 | |
| Random Forest | 0.6436 | 0.0877 | |
| KNN | 0.5855 | 0.0192 | Peor en desbalance |
| lgbm_smote (GridSearch) | 0.6851 | 0.0545 | Mejor ROC pero F1 bajo |
| xgb_smote (GridSearch) | 0.6637 | 0.1538 | |
| logreg_smote (GridSearch) | 0.6433 | 0.1290 | |

**Modelo final:** LGBM baseline con `class_weight='balanced'`.  
**Threshold de decision optimizado:** 0.35 (F1 macro), en lugar del 0.5 por defecto.

### Monitoreo
Sobre el split temporal historico (70% mas antiguo) vs actual (30% mas reciente):
- **24 variables analizadas** con KS test, PSI, Jensen-Shannon, Chi-cuadrado.
- 6 variables con drift severo, 10 moderado, 8 sin drift.
- Performance del modelo se mantiene estable (caida ROC-AUC ~0.02, dentro del umbral).
- **Recomendacion automatica:** MONITOREO_REFORZADO + reentrenamiento preventivo.

---

## Estructura del repositorio

```
PI/
|-- mlops_pipeline/
|   |-- set_up.bat                       # Crea venv e instala dependencias
|   |-- requirements.txt
|   `-- src/
|       |-- config.json                  # Metadatos del proyecto
|       |-- Cargar_datos.ipynb           # Avance 1 - Carga y validacion
|       |-- comprension_eda.ipynb        # Avance 1 - EDA completo
|       |-- ft_engineering.py            # Avance 2 - Feature engineering
|       |-- model_training_evaluation.py # Avance 2 - 5 modelos + SMOTE + GridSearch
|       |-- model_deploy.py              # Avance 4 - FastAPI
|       `-- model_monitoring.py          # Avance 3 - Drift detection
|-- models/
|   |-- best_model.joblib                # Pipeline ganador (preproc + LGBM)
|   |-- preprocessor.joblib              # ColumnTransformer ajustado
|   |-- threshold_optimo.json            # Umbral 0.35
|   |-- model_metrics.json               # Todas las metricas
|   |-- comparacion_modelos.png          # Visualizaciones
|   |-- curvas_roc.png
|   |-- curvas_pr.png
|   `-- matrices_confusion.png
|-- data_processed/
|   |-- dataset_limpio.parquet           # Output Cargar_datos
|   |-- X_train.parquet, X_test.parquet  # Splits
|   |-- y_train.parquet, y_test.parquet
|   |-- X_train_transformed.parquet      # Post-preprocessor
|   |-- X_test_transformed.parquet
|   |-- feature_cols.json                # Catalogo de features
|   |-- reglas_validacion.json           # Contrato de datos
|   |-- prediction_log.csv               # Log para drift
|   |-- drift_metrics.csv                # Metricas por variable
|   |-- drift_summary.json               # Resumen ejecutivo
|   `-- model_performance_drift.json
|-- streamlit_app/
|   `-- app.py                           # Dashboard de monitoreo
|-- tests/                               # Tests unitarios (Extra Credit)
|-- .github/workflows/                   # GitHub Actions (Extra Credit)
|-- Base_de_datos.csv                    # Dataset fuente
|-- requirements.txt
|-- .gitignore
|-- Dockerfile                           # Avance 4
|-- readme.md
|-- init_git.bat                         # Setup Git inicial
`-- set_up.bat                           # Setup venv (alternativo)
```

---

## Setup local (Windows)

### Paso 1 - Clonar e inicializar Git
```cmd
git clone <URL_DEL_REPO>
cd PI
init_git.bat
```

### Paso 2 - Crear entorno virtual y registrar kernel Jupyter
```cmd
cd mlops_pipeline
.\set_up.bat
cd ..
```
Crea `mlops_pipeline/mlops_credito_m5-venv/` e instala todas las dependencias. Toma 3-5 min.

### Paso 3 - Activar el entorno
```cmd
mlops_pipeline\mlops_credito_m5-venv\Scripts\activate
```

### Paso 4 - Ejecutar el pipeline en orden
```cmd
cd mlops_pipeline\src
jupyter notebook Cargar_datos.ipynb           # Avance 1 - carga
jupyter notebook comprension_eda.ipynb        # Avance 1 - EDA
python ft_engineering.py                      # Avance 2 - features
python model_training_evaluation.py           # Avance 2 - modelado
python model_monitoring.py                    # Avance 3 - drift
cd ..\..
streamlit run streamlit_app\app.py            # Avance 3 - dashboard
uvicorn mlops_pipeline.src.model_deploy:app   # Avance 4 - API
```

---

## Branches y versionado

```
                  V1.0.0       V1.0.1       V1.1.0       V1.1.1
master:           o------------o------------o------------o
                  |            |            |            |
certification: ---o------------o------------o------------o
                  |            |            |            |
developer:    ----o------------o------------o------------o
                  estructura   carga+EDA    FE+modelado  drift+API+Docker
```

| Version | Avance | Contenido |
|---------|--------|-----------|
| V1.0.0  | - | Estructura inicial del proyecto |
| V1.0.1  | 1 | Cargar_datos + comprension_eda |
| V1.1.0  | 2 | ft_engineering + model_training_evaluation |
| V1.1.1  | 3 + 4 | model_monitoring + Streamlit + FastAPI + Dockerfile |

Flujo de PR: developer → certification (validacion) → master (produccion estable).

---

## Tech stack

- **Lenguaje:** Python 3.10+
- **Datos:** pandas, numpy, pyarrow, openpyxl
- **Viz:** matplotlib, seaborn
- **ML:** scikit-learn, xgboost, lightgbm, feature-engine, imbalanced-learn
- **Drift:** scipy (KS, JS, chi2), PSI manual
- **API:** FastAPI, uvicorn, pydantic, requests
- **Dashboard:** Streamlit
- **Despliegue:** Docker
- **Testing/Calidad:** pytest, pytest-cov, SonarCloud

---

## Autora

**Anastasia** - aganderatsi@gmail.com  
Henry - Modulo 5 - Cohorte 2026
