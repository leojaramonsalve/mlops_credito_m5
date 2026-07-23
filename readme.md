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

### Modelado (Avance 2)
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

### Monitoreo de drift (Avance 3)
El monitoreo simula el flujo productivo con un **split temporal por `fecha_prestamo`**: el 70% mas antiguo actua como **historico** (referencia) y el 30% mas reciente como **actual** (produccion simulada). Se comparan ambos periodos.

**Metricas calculadas por variable:**
- **KS test** (numericas): compara distribuciones; p-value < 0.05 indica cambio significativo.
- **PSI** (estandar de la banca): < 0.10 sin drift, 0.10-0.25 moderado, > 0.25 severo.
- **Jensen-Shannon**: distancia simetrica entre distribuciones (0 a 1); > 0.10 notable.
- **Chi-cuadrado** (categoricas): p-value < 0.05 indica cambio en las proporciones.

**Resultados (historico n=7.534 vs actual n=3.229):**
- **24 variables analizadas**: 6 con drift severo, 10 moderado, 8 sin drift.
- Variables con drift severo: `total_otros_prestamos`, `promedio_ingresos_datacredito`, `ratio_otros_salario`, `endeudamiento_total`, `plazo_meses`, `mes_prestamo`.
- **Model drift: OK.** El ROC-AUC del modelo pasa de 0.9383 (historico) a 0.9172 (actual): caida de 0.0211, dentro del umbral de 0.05. El modelo se mantiene estable.
- **Recomendacion automatica:** MONITOREO_REFORZADO + reentrenamiento preventivo (hay drift en los datos de entrada aunque la performance aun no cae).

El **dashboard de Streamlit** (`streamlit_app/app.py`) presenta todo esto de forma visual en 6 secciones: resumen ejecutivo, performance del modelo, drift por variable (tabla con semaforos), comparacion de distribuciones historico vs actual, evolucion temporal de predicciones y recomendaciones automaticas.

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
|       |-- modelamiento.ipynb           # Avance 2 - Analisis narrado del modelado
|       |-- model_monitoring.py          # Avance 3 - Drift detection
|       `-- model_deploy.py              # Avance 4 - FastAPI (pendiente)
|-- models/
|   |-- best_model.joblib                # Pipeline ganador (preproc + LGBM)
|   |-- preprocessor.joblib              # ColumnTransformer ajustado
|   |-- threshold_optimo.json            # Umbral 0.35
|   |-- model_metrics.json               # Todas las metricas
|   |-- comparacion_modelos.png          # Visualizaciones
|   |-- curvas_roc.png
|   |-- curvas_pr.png
|   |-- matrices_confusion.png
|   |-- feature_importance.png
|   `-- threshold_tuning.png
|-- data_processed/
|   |-- dataset_limpio.parquet           # Output Cargar_datos
|   |-- X_train.parquet, X_test.parquet  # Splits
|   |-- y_train.parquet, y_test.parquet
|   |-- X_train_transformed.parquet      # Post-preprocessor
|   |-- X_test_transformed.parquet
|   |-- feature_cols.json                # Catalogo de features
|   |-- reglas_validacion.json           # Contrato de datos
|   |-- prediction_log.csv               # Log para drift (Avance 3)
|   |-- drift_metrics.csv                # Metricas por variable (Avance 3)
|   |-- drift_summary.json               # Resumen ejecutivo (Avance 3)
|   `-- model_performance_drift.json     # Model drift (Avance 3)
|-- streamlit_app/
|   `-- app.py                           # Dashboard de monitoreo (Avance 3)
|-- tests/                               # Tests unitarios (Extra Credit)
|-- Base_de_datos.csv                    # Dataset fuente
|-- requirements.txt
|-- .gitignore
|-- readme.md
`-- set_up.bat                           # Setup venv (alternativo)
```

*Los elementos del Avance 4 (model_deploy.py con contenido, Dockerfile, .github/workflows) se incorporan en la siguiente entrega.*

---

## Setup local (Windows)

> **Importante - version de Python:** usar **Python 3.11 o 3.12**. Las versiones fijadas en `requirements.txt` (numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.0, scipy 1.13.1) aun no tienen instaladores para Python 3.13, por lo que la instalacion fallaria.

### Paso 1 - Clonar el repositorio
```cmd
git clone <URL_DEL_REPO>
cd PI
```

### Paso 2 - Crear el entorno virtual con Python 3.12
```cmd
py -3.12 -m venv mlops-venv
mlops-venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
> Si al instalar `jupyter`/`jupyterlab` aparece un error de rutas largas de Windows, activar el soporte de rutas largas (registro `LongPathsEnabled=1`, requiere administrador) o instalar solo lo necesario. Para trabajar en VS Code basta con `ipykernel` y no hace falta `jupyterlab`.

### Paso 3 - Ejecutar el pipeline en orden
```cmd
cd mlops_pipeline\src
python ft_engineering.py                      # Avance 2 - features y splits
python model_training_evaluation.py           # Avance 2 - modelado (o correr modelamiento.ipynb)
python model_monitoring.py                    # Avance 3 - calcula drift
cd ..\..
streamlit run streamlit_app\app.py            # Avance 3 - dashboard (http://localhost:8501)
```
En VS Code, los notebooks (`Cargar_datos.ipynb`, `comprension_eda.ipynb`, `modelamiento.ipynb`) se ejecutan seleccionando el kernel `mlops-venv`.

---

## Branches y versionado

```
                 V1.0.0       V1.0.1       V1.1.0       V1.1.1
master        ---o------------o------------o------------o
                              ^ merge via Pull Request
certification ---o------------o------------o------------o

developer     ---o------------o------------o------------o
                 estructura   carga+EDA    FE+modelado   drift+dashboard
```

| Version | Avance | Contenido |
|---------|--------|-----------|
| V1.0.0  | - | Estructura inicial del proyecto |
| V1.0.1  | 1 | Cargar_datos + comprension_eda |
| V1.1.0  | 2 | ft_engineering + model_training_evaluation + modelamiento |
| V1.1.1  | 3 | model_monitoring + dashboard Streamlit + README |
| V1.2.0  | 4 | model_deploy (FastAPI) + Dockerfile *(pendiente)* |

Flujo de PR: developer -> certification (validacion) -> master (produccion estable).

*Nota: el grafico de la consigna (image2) rotula la rama principal como `main`; este repositorio usa `master`, siguiendo el texto de las instrucciones. La version V1.2.0 (Avance 4) aun no se ha generado.*

---

## Tech stack

- **Lenguaje:** Python 3.11 / 3.12
- **Datos:** pandas, numpy, pyarrow, openpyxl
- **Viz:** matplotlib, seaborn
- **ML:** scikit-learn, xgboost, lightgbm, feature-engine, imbalanced-learn
- **Drift:** scipy (KS, JS, chi2), PSI manual
- **API:** FastAPI, uvicorn, pydantic, requests *(Avance 4)*
- **Dashboard:** Streamlit
- **Despliegue:** Docker *(Avance 4)*
- **Testing/Calidad:** pytest, pytest-cov, SonarCloud *(Extra Credit)*

---

## Autor

**Leonardo Jaramonsalve** - github.com/leojaramonsalve  
Henry - Modulo 5 - Cohorte 2026
