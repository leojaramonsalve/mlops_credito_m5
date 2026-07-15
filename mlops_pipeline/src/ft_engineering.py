"""
ft_engineering.py
==================
Ingenieria de caracteristicas para el pipeline MLOps de prediccion de pago a tiempo.

Este modulo construye el preprocesador (ColumnTransformer) que aplica:
  - Pipeline numerico:       SimpleImputer(mediana) -> StandardScaler
  - Pipeline categorico nom: SimpleImputer(most_frequent) -> OneHotEncoder
  - Pipeline categorico ord: SimpleImputer(most_frequent) -> OrdinalEncoder

Adicionalmente genera features derivadas (ratios financieros, banderas booleanas,
variables temporales) y excluye variables identificadas como leakage en el EDA.

Funciones principales:
  - load_clean_dataset(): carga el parquet generado por Cargar_datos.ipynb
  - build_derived_features(): genera features calculadas
  - get_feature_columns(): retorna las listas de columnas por tipo
  - build_preprocessor(): construye el ColumnTransformer
  - prepare_dataset(): orquesta carga + features + split estratificado
  - main(): entrypoint que persiste train/test/transformer en data_processed/

Salidas (en data_processed/):
  - X_train.parquet, X_test.parquet
  - y_train.parquet, y_test.parquet
  - X_train_transformed.parquet, X_test_transformed.parquet
  - preprocessor.joblib (en models/)
  - feature_cols.json (catalogo de features por familia)

Uso como script:
    python ft_engineering.py

Estado: V1.1.0 - Avance 2.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# Configuracion del modulo
# ============================================================

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROC_DIR = PROJECT_ROOT / "data_processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Variables identificadas como LEAKAGE.
#
# Grupo 1 - Detectado en EDA (Avance 1):
#   Saldos con AUC univariada > 0.90. Estado post-otorgamiento del prestamo,
#   no disponible al momento de evaluar un nuevo cliente.
#
# Grupo 2 - Detectado en validacion del Avance 2 (post-EDA):
#   'puntaje' tiene AUC univariada = 1.0 (prediccion perfecta como variable
#   unica), captura 74% de la importancia del Random Forest, y separa
#   trivialmente el target. Hipotesis: es un score post-hoc calculado a
#   partir del comportamiento observado, o el output de un modelo previo
#   entrenado sobre los mismos datos. En ambos casos su uso violaria el
#   principio de "disponibilidad al momento de la prediccion": un nuevo
#   cliente sin historial no tendria este puntaje. Se excluye y se
#   documenta como hallazgo.
#
# Nota: 'puntaje_datacredito' (AUC 0.62) SI se mantiene - es el score del
# buro externo Datacredito, disponible al momento de la solicitud.
COLUMNAS_LEAKAGE = [
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "puntaje",
]

# Orden ordinal para tendencia_ingresos (debe ser explicito).
ORDEN_TENDENCIA = ["Decreciente", "Estable", "Creciente"]


# ============================================================
# Carga del dataset limpio
# ============================================================

def load_config() -> Dict:
    """Carga config.json con los metadatos del proyecto."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_clean_dataset(parquet_path: Path = None) -> pd.DataFrame:
    """
    Carga el dataset limpio producido por Cargar_datos.ipynb.
    Re-aplica el orden ordinal de tendencia_ingresos (parquet no lo preserva).
    """
    if parquet_path is None:
        parquet_path = DATA_PROC_DIR / "dataset_limpio.parquet"

    df = pd.read_parquet(parquet_path)

    # Re-aplicar tipos categoricos que parquet no preserva
    df["tendencia_ingresos"] = pd.Categorical(
        df["tendencia_ingresos"],
        categories=ORDEN_TENDENCIA,
        ordered=True,
    )
    df["tipo_credito"] = df["tipo_credito"].astype("category")
    df["tipo_laboral"] = df["tipo_laboral"].astype("category")

    return df


# ============================================================
# Features derivadas
# ============================================================

def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye atributos derivados utiles identificados en el EDA.

    Categorias:
      - Ratios financieros: relativizan magnitudes al salario del cliente.
      - Banderas booleanas: capturan condiciones discretas relevantes.
      - Variables temporales: extraen periodicidad de fecha_prestamo.

    No usa informacion post-otorgamiento. Todas las features estan disponibles
    al momento de la solicitud de credito.
    """
    df = df.copy()
    salario_safe = df["salario_cliente"].replace(0, np.nan)

    # 1. Ratios financieros
    df["ratio_cuota_salario"] = df["cuota_pactada"] / salario_safe
    df["ratio_capital_salario"] = df["capital_prestado"] / salario_safe
    df["ratio_otros_salario"] = df["total_otros_prestamos"] / salario_safe
    df["endeudamiento_total"] = (
        df["cuota_pactada"] + df["total_otros_prestamos"]
    ) / salario_safe

    # 2. Banderas booleanas
    df["tiene_historial_datacredito"] = df["puntaje_datacredito"].notna().astype(int)
    df["multiples_sectores"] = (
        (df["creditos_sectorFinanciero"] > 0).astype(int)
        + (df["creditos_sectorCooperativo"] > 0).astype(int)
        + (df["creditos_sectorReal"] > 0).astype(int)
    )

    # 3. Variables temporales (de fecha_prestamo)
    df["anio_prestamo"] = df["fecha_prestamo"].dt.year
    df["mes_prestamo"] = df["fecha_prestamo"].dt.month

    return df


# ============================================================
# Definicion de columnas
# ============================================================

def get_feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Retorna las listas de columnas a usar para el modelo, agrupadas por tipo.
    Excluye:
      - El target.
      - Las columnas identificadas como leakage.
      - fecha_prestamo cruda (ya extraida en anio_prestamo y mes_prestamo).

    Las features derivadas se incluyen en su grupo correspondiente.
    """
    numericas_continuas = [
        "capital_prestado",
        "salario_cliente",
        "total_otros_prestamos",
        "cuota_pactada",
        # 'puntaje' EXCLUIDO por leakage (AUC=1.0); ver COLUMNAS_LEAKAGE
        "puntaje_datacredito",
        "promedio_ingresos_datacredito",
        # Derivadas continuas
        "ratio_cuota_salario",
        "ratio_capital_salario",
        "ratio_otros_salario",
        "endeudamiento_total",
    ]

    numericas_discretas = [
        "plazo_meses",
        "edad_cliente",
        "cant_creditosvigentes",
        "huella_consulta",
        "creditos_sectorFinanciero",
        "creditos_sectorCooperativo",
        "creditos_sectorReal",
        # Derivadas discretas
        "tiene_historial_datacredito",
        "multiples_sectores",
        "anio_prestamo",
        "mes_prestamo",
    ]

    categoricas_nominales = [
        "tipo_credito",
        "tipo_laboral",
    ]

    categoricas_ordinales = [
        "tendencia_ingresos",
    ]

    # Filtrar columnas que efectivamente existen en el df
    available = set(df.columns)
    numericas_continuas = [c for c in numericas_continuas if c in available]
    numericas_discretas = [c for c in numericas_discretas if c in available]
    categoricas_nominales = [c for c in categoricas_nominales if c in available]
    categoricas_ordinales = [c for c in categoricas_ordinales if c in available]

    return {
        "numericas_continuas": numericas_continuas,
        "numericas_discretas": numericas_discretas,
        "categoricas_nominales": categoricas_nominales,
        "categoricas_ordinales": categoricas_ordinales,
    }


# ============================================================
# Construccion del preprocesador
# ============================================================

def build_preprocessor(feature_cols: Dict[str, List[str]]) -> ColumnTransformer:
    """
    Construye el ColumnTransformer con los tres pipelines:
      - Numerico (continuas + discretas):   SimpleImputer(mediana) -> StandardScaler
      - Categorico nominal:                  SimpleImputer(most_frequent) -> OneHotEncoder
      - Categorico ordinal:                  SimpleImputer(most_frequent) -> OrdinalEncoder

    Para las categoricas nominales se usa handle_unknown='ignore' (si en produccion
    aparece una categoria nunca vista, se codifica como vector de ceros).
    """
    pipe_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pipe_nominal = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pipe_ordinal = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[ORDEN_TENDENCIA],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    columnas_numericas = (
        feature_cols["numericas_continuas"] + feature_cols["numericas_discretas"]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", pipe_numerico, columnas_numericas),
            ("nom", pipe_nominal, feature_cols["categoricas_nominales"]),
            ("ord", pipe_ordinal, feature_cols["categoricas_ordinales"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ============================================================
# Orquestador
# ============================================================

def prepare_dataset(
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, Dict]:
    """
    Carga, transforma y particiona el dataset listo para entrenar.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor (no ajustado), feature_cols (dict)
    """
    config = load_config()
    target = config["target_variable"]

    # 1. Cargar limpio
    df = load_clean_dataset()

    # 2. Features derivadas
    df = build_derived_features(df)

    # 3. Definir columnas
    feature_cols = get_feature_columns(df)
    todas_las_features = (
        feature_cols["numericas_continuas"]
        + feature_cols["numericas_discretas"]
        + feature_cols["categoricas_nominales"]
        + feature_cols["categoricas_ordinales"]
    )

    # 4. Asegurar que NO incluimos leakage por accidente
    for col_leak in COLUMNAS_LEAKAGE:
        assert col_leak not in todas_las_features, (
            f"Error: columna de leakage '{col_leak}' se colo en las features"
        )

    X = df[todas_las_features].copy()
    y = df[target].copy()

    # 5. Split estratificado (preserva desbalance 95/5 en ambos conjuntos)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 6. Construir preprocesador (NO se ajusta aqui; se ajusta cuando se entrena el modelo)
    preprocessor = build_preprocessor(feature_cols)

    return X_train, X_test, y_train, y_test, preprocessor, feature_cols


# ============================================================
# Entrypoint
# ============================================================

def main() -> None:
    """
    Entrypoint del script. Prepara los conjuntos de train/test, ajusta el
    preprocesador con el train, transforma ambos y persiste los artefactos.
    """
    print("=" * 70)
    print("FEATURE ENGINEERING - V1.1.0 (Avance 2)")
    print("=" * 70)

    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Preparar dataset
    print("\n[1/4] Preparando dataset...")
    X_train, X_test, y_train, y_test, preprocessor, feature_cols = prepare_dataset()

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  Distribucion target en train: {dict(y_train.value_counts(normalize=True).round(4))}")
    print(f"  Distribucion target en test:  {dict(y_test.value_counts(normalize=True).round(4))}")

    # 2. Ajustar preprocesador con train
    print("\n[2/4] Ajustando preprocesador en train...")
    preprocessor.fit(X_train)

    # 3. Transformar y persistir
    print("\n[3/4] Transformando y persistiendo...")
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]

    X_train_t_df = pd.DataFrame(X_train_t, columns=feature_names, index=X_train.index)
    X_test_t_df = pd.DataFrame(X_test_t, columns=feature_names, index=X_test.index)

    X_train.to_parquet(DATA_PROC_DIR / "X_train.parquet")
    X_test.to_parquet(DATA_PROC_DIR / "X_test.parquet")
    y_train.to_frame().to_parquet(DATA_PROC_DIR / "y_train.parquet")
    y_test.to_frame().to_parquet(DATA_PROC_DIR / "y_test.parquet")
    X_train_t_df.to_parquet(DATA_PROC_DIR / "X_train_transformed.parquet")
    X_test_t_df.to_parquet(DATA_PROC_DIR / "X_test_transformed.parquet")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    with open(DATA_PROC_DIR / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    # 4. Reporte final
    print("\n[4/4] Resumen:")
    print(f"  Features originales:      {X_train.shape[1]} columnas")
    print(f"  Features transformadas:   {X_train_t.shape[1]} columnas (post OneHotEncoder)")
    print(f"  Numericas:                {len(feature_cols['numericas_continuas']) + len(feature_cols['numericas_discretas'])}")
    print(f"  Categoricas nominales:    {len(feature_cols['categoricas_nominales'])} (-> OHE)")
    print(f"  Categoricas ordinales:    {len(feature_cols['categoricas_ordinales'])}")
    print(f"  Variables excluidas por leakage: {COLUMNAS_LEAKAGE}")
    print()
    print(f"  Artefactos persistidos en:")
    print(f"    {DATA_PROC_DIR}/")
    print(f"      - X_train.parquet, X_test.parquet")
    print(f"      - X_train_transformed.parquet, X_test_transformed.parquet")
    print(f"      - y_train.parquet, y_test.parquet")
    print(f"      - feature_cols.json")
    print(f"    {MODELS_DIR}/")
    print(f"      - preprocessor.joblib")
    print()
    print("Listo para model_training_evaluation.py")


if __name__ == "__main__":
    main()
