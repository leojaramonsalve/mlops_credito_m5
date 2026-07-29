"""
model_deploy.py
================
Despliegue del modelo de prediccion de pago a tiempo como API REST con FastAPI.

Endpoints:
  - GET  /                : informacion del servicio
  - GET  /health          : healthcheck (verifica que el modelo este cargado)
  - GET  /model_info      : metadata del modelo desplegado
  - POST /predict         : prediccion para un cliente individual (JSON)
  - POST /predict_batch   : prediccion por lote (lista de clientes JSON)
  - POST /predict_csv     : prediccion por lote desde upload de CSV

Schema de entrada validado con Pydantic.

Decisiones de diseno:
  - Threshold de decision configurable (default = optimo guardado por el training).
  - Validacion estricta del input: tipos, rangos, categorias permitidas.
  - Responde tanto la prediccion binaria como la probabilidad.
  - Modo batch optimizado: convierte a DataFrame y predice de una.

Estado: V1.1.1 - Avance 4.

Uso local:
    uvicorn model_deploy:app --reload --host 0.0.0.0 --port 8000

Uso con Docker:
    docker build -t mlops_credito_m5:latest .
    docker run -p 8000:8000 mlops_credito_m5:latest

Documentacion automatica:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import io
import json
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_engineering import (
    build_derived_features,
    get_feature_columns,
    MODELS_DIR,
    COLUMNAS_LEAKAGE,
    ORDEN_TENDENCIA,
)

# ============================================================
# Configuracion
# ============================================================

APP_NAME = "Pipeline MLOps - Prediccion de Pago a Tiempo"
APP_VERSION = "1.1.1"
PROJECT_CODE = "mlops_credito_m5"


# ============================================================
# Cargar modelo al iniciar
# ============================================================

MODELO = None
THRESHOLD = 0.5
METRICAS_MODELO = {}


def cargar_modelo():
    """Carga el modelo entrenado y el threshold optimo."""
    global MODELO, THRESHOLD, METRICAS_MODELO
    try:
        MODELO = joblib.load(MODELS_DIR / "best_model.joblib")
        # Cargar threshold optimo (si existe)
        thr_path = MODELS_DIR / "threshold_optimo.json"
        if thr_path.exists():
            with open(thr_path, "r") as f:
                thr_data = json.load(f)
                THRESHOLD = float(thr_data.get("threshold", 0.5))
        # Cargar metricas del modelo
        metrics_path = MODELS_DIR / "model_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                METRICAS_MODELO = json.load(f)
        return True
    except Exception as e:
        print(f"ERROR cargando modelo: {e}")
        return False


# ============================================================
# Schemas Pydantic
# ============================================================

class ClienteInput(BaseModel):
    """
    Schema de un cliente para prediccion. Las validaciones siguen las reglas
    de validacion documentadas en data_processed/reglas_validacion.json.
    """
    tipo_credito: int = Field(..., ge=1, le=20, description="Tipo de credito (codificado)")
    fecha_prestamo: str = Field(..., description="Fecha del prestamo (ISO 8601)")
    capital_prestado: float = Field(..., gt=0, description="Capital solicitado")
    plazo_meses: int = Field(..., gt=0, le=240, description="Plazo en meses")
    edad_cliente: int = Field(..., ge=18, le=100, description="Edad del cliente")
    tipo_laboral: Literal["Empleado", "Independiente"] = Field(..., description="Tipo de relacion laboral")
    salario_cliente: float = Field(..., ge=0, description="Salario mensual")
    total_otros_prestamos: float = Field(..., ge=0, description="Otros prestamos vigentes")
    cuota_pactada: float = Field(..., ge=0, description="Cuota mensual pactada")
    puntaje_datacredito: Optional[float] = Field(None, description="Score Datacredito (puede ser null)")
    cant_creditosvigentes: int = Field(..., ge=0, description="Cantidad de creditos vigentes")
    huella_consulta: int = Field(..., ge=0, description="Numero de consultas al buro")
    creditos_sectorFinanciero: int = Field(..., ge=0)
    creditos_sectorCooperativo: int = Field(..., ge=0)
    creditos_sectorReal: int = Field(..., ge=0)
    promedio_ingresos_datacredito: Optional[float] = Field(None, description="Promedio ingresos del buro (puede ser null)")
    tendencia_ingresos: Optional[Literal["Decreciente", "Estable", "Creciente"]] = Field(None)

    @field_validator("fecha_prestamo")
    @classmethod
    def validar_fecha(cls, v):
        try:
            pd.to_datetime(v)
        except Exception:
            raise ValueError(f"fecha_prestamo invalida: {v}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "tipo_credito": 4,
                "fecha_prestamo": "2026-05-15",
                "capital_prestado": 5000000.0,
                "plazo_meses": 24,
                "edad_cliente": 35,
                "tipo_laboral": "Empleado",
                "salario_cliente": 3500000.0,
                "total_otros_prestamos": 1500000.0,
                "cuota_pactada": 250000.0,
                "puntaje_datacredito": 780.0,
                "cant_creditosvigentes": 3,
                "huella_consulta": 2,
                "creditos_sectorFinanciero": 2,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 1,
                "promedio_ingresos_datacredito": 3200000.0,
                "tendencia_ingresos": "Estable",
            }
        }


class PredictBatchInput(BaseModel):
    """Lista de clientes para prediccion por lote."""
    clientes: List[ClienteInput] = Field(..., min_length=1, max_length=1000)


class PrediccionOutput(BaseModel):
    """Resultado de una prediccion individual."""
    prediccion: int = Field(..., description="0 = impago, 1 = paga a tiempo")
    probabilidad_pago: float = Field(..., ge=0, le=1, description="Probabilidad de pagar a tiempo")
    probabilidad_impago: float = Field(..., ge=0, le=1, description="Probabilidad de impago")
    threshold_aplicado: float = Field(..., description="Threshold de decision usado")
    interpretacion: str = Field(..., description="Mensaje en lenguaje natural")


class BatchOutput(BaseModel):
    """Resultado de prediccion por lote."""
    n_predicciones: int
    resumen: dict
    predicciones: List[PrediccionOutput]


# ============================================================
# App FastAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description="API REST para clasificar la probabilidad de pago a tiempo de un cliente "
                "de credito. Pipeline MLOps del Modulo 5 de Henry.",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
async def startup_event():
    if not cargar_modelo():
        print("WARNING: El modelo no se pudo cargar. /predict fallara hasta que esto se resuelva.")


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
async def root():
    """Informacion general del servicio."""
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "project_code": PROJECT_CODE,
        "estado": "online" if MODELO else "modelo_no_cargado",
        "endpoints": {
            "GET /health": "Healthcheck",
            "GET /model_info": "Metadata del modelo desplegado",
            "POST /predict": "Prediccion individual",
            "POST /predict_batch": "Prediccion por lote (max 1000)",
            "POST /predict_csv": "Prediccion por lote desde CSV upload",
            "GET /docs": "Documentacion Swagger",
            "GET /redoc": "Documentacion ReDoc",
        },
    }


@app.get("/health")
async def health():
    """Healthcheck - confirma que el servicio esta corriendo y el modelo cargado."""
    if MODELO is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {
        "status": "healthy",
        "modelo_cargado": True,
        "threshold": THRESHOLD,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model_info")
async def model_info():
    """Metadata del modelo desplegado: tipo, features esperadas, metricas, threshold."""
    if MODELO is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    try:
        clf = MODELO.named_steps['classifier']
        prep = MODELO.named_steps['preprocessor']
        feature_names = list(prep.get_feature_names_out())
    except Exception:
        clf = None
        feature_names = []

    return {
        "modelo_tipo": type(clf).__name__ if clf else "desconocido",
        "n_features_esperadas": len(feature_names),
        "feature_names": feature_names,
        "threshold_decision": THRESHOLD,
        "variables_excluidas_leakage": COLUMNAS_LEAKAGE,
        "metricas_holdout": METRICAS_MODELO.get("modelos_baseline", []),
        "mejor_modelo": METRICAS_MODELO.get("mejor_modelo", "desconocido"),
        "score_compuesto": METRICAS_MODELO.get("score_compuesto_mejor"),
    }


def _preparar_dataframe(clientes_json):
    """Convierte lista de dicts en DataFrame con tipos correctos + features derivadas."""
    df = pd.DataFrame(clientes_json)
    df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'])
    # Aplicar features derivadas (las mismas que en entrenamiento)
    df = build_derived_features(df)
    return df


def _predecir(df):
    """Ejecuta el modelo sobre un DataFrame y retorna predicciones + probabilidades."""
    feature_cols = get_feature_columns(df)
    todas = (feature_cols['numericas_continuas'] + feature_cols['numericas_discretas']
             + feature_cols['categoricas_nominales'] + feature_cols['categoricas_ordinales'])
    X = df[todas]
    probas = MODELO.predict_proba(X)[:, 1]  # proba de clase 1 (paga)
    preds = (probas >= THRESHOLD).astype(int)
    return preds, probas


@app.post("/predict", response_model=PrediccionOutput)
async def predict(cliente: ClienteInput):
    """Prediccion individual para un cliente."""
    if MODELO is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    try:
        df = _preparar_dataframe([cliente.model_dump()])
        preds, probas = _predecir(df)
        pred = int(preds[0])
        proba_pago = float(probas[0])
        proba_impago = 1 - proba_pago

        if pred == 1:
            interpretacion = f"APROBAR: probabilidad de pago a tiempo = {proba_pago:.1%}"
        else:
            interpretacion = f"RECHAZAR / REVISAR: probabilidad de impago = {proba_impago:.1%}"

        return PrediccionOutput(
            prediccion=pred,
            probabilidad_pago=round(proba_pago, 4),
            probabilidad_impago=round(proba_impago, 4),
            threshold_aplicado=THRESHOLD,
            interpretacion=interpretacion,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en prediccion: {e}")


@app.post("/predict_batch", response_model=BatchOutput)
async def predict_batch(payload: PredictBatchInput):
    """Prediccion por lote para hasta 1000 clientes."""
    if MODELO is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    try:
        clientes_json = [c.model_dump() for c in payload.clientes]
        df = _preparar_dataframe(clientes_json)
        preds, probas = _predecir(df)

        resultados = []
        for pred, proba in zip(preds, probas):
            pred_int = int(pred)
            proba_pago = float(proba)
            proba_impago = 1 - proba_pago
            interp = (f"APROBAR: prob pago = {proba_pago:.1%}" if pred_int == 1
                      else f"RECHAZAR/REVISAR: prob impago = {proba_impago:.1%}")
            resultados.append(PrediccionOutput(
                prediccion=pred_int,
                probabilidad_pago=round(proba_pago, 4),
                probabilidad_impago=round(proba_impago, 4),
                threshold_aplicado=THRESHOLD,
                interpretacion=interp,
            ))

        n_aprobados = int((preds == 1).sum())
        n_rechazados = int((preds == 0).sum())
        resumen = {
            "total": len(resultados),
            "aprobados": n_aprobados,
            "rechazados": n_rechazados,
            "tasa_aprobacion": round(n_aprobados / len(resultados), 4),
            "probabilidad_pago_promedio": round(float(probas.mean()), 4),
        }

        return BatchOutput(n_predicciones=len(resultados), resumen=resumen, predicciones=resultados)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en prediccion batch: {e}")


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(..., description="CSV con columnas iguales al schema de ClienteInput")):
    """
    Prediccion por lote desde un archivo CSV. Mas eficiente para volumenes grandes.
    El CSV debe tener las mismas columnas que ClienteInput.
    """
    if MODELO is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se acepta CSV")

    try:
        contenido = await file.read()
        df = pd.read_csv(io.BytesIO(contenido))
        df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'])
        df = build_derived_features(df)
        preds, probas = _predecir(df)

        return JSONResponse({
            "n_predicciones": int(len(preds)),
            "resumen": {
                "aprobados": int((preds == 1).sum()),
                "rechazados": int((preds == 0).sum()),
                "tasa_aprobacion": round(float((preds == 1).mean()), 4),
                "prob_pago_promedio": round(float(probas.mean()), 4),
            },
            "predicciones": [
                {"prediccion": int(p), "probabilidad_pago": round(float(pr), 4),
                 "probabilidad_impago": round(1 - float(pr), 4)}
                for p, pr in zip(preds, probas)
            ],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando CSV: {e}")


# Ejecutar como script directo (sin uvicorn separado) para pruebas
if __name__ == "__main__":
    import uvicorn
    cargar_modelo()
    uvicorn.run(app, host="0.0.0.0", port=8000)
