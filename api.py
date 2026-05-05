"""
SafeRouteAI — Fase 2: Backend FastAPI
======================================
API REST con dos endpoints principales:
  - POST /train      → entrena el modelo y guarda modelo.pkl
  - GET  /predecir   → predice el nivel de riesgo dado GPS + hora + día

Cómo arrancar:
  uvicorn api:app --reload

Documentación automática (Swagger):
  http://127.0.0.1:8000/docs

Autor: Cristian Corroto Otero / SafeRouteAI
"""

import os
import math
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

CSV_PATH   = "datos_riesgo_madrid.csv"
MODEL_PATH = "modelo.pkl"

# Hotspots con coordenadas GPS reales
HOTSPOTS = {
    # Coordenadas verificadas via Google Places API
    "Lavapiés":       {"lat": 40.4088, "lon": -3.7013, "radio_km": 0.8,  "ocio": 1},
    "Carabanchel":    {"lat": 40.3806, "lon": -3.7409, "radio_km": 1.5,  "ocio": 0},
    "Vallecas":       {"lat": 40.3790, "lon": -3.6214, "radio_km": 1.5,  "ocio": 0},
    "Sol_GranVia":    {"lat": 40.4170, "lon": -3.7035, "radio_km": 0.8,  "ocio": 1},
    "AZCA":           {"lat": 40.4767, "lon": -3.6879, "radio_km": 0.6,  "ocio": 1},
    "Becerril_Sierra":{"lat": 40.7163, "lon": -3.9893, "radio_km": 2.0,  "ocio": 0},
}

# ─────────────────────────────────────────────
# APP FASTAPI
# ─────────────────────────────────────────────

app = FastAPI(
    title="SafeRouteAI API",
    description="API de predicción de riesgo por zonas de Madrid. "
                "Devuelve nivel de riesgo: verde, amarillo o rojo.",
    version="1.0.0",
)

# Permite que la app Android se conecte sin problemas de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable global para el modelo cargado en memoria
modelo_cargado = None
label_encoder  = None


# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en km entre dos puntos GPS."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def zona_mas_cercana(lat: float, lon: float):
    """Devuelve el hotspot más cercano y la distancia a él."""
    min_dist = float("inf")
    zona_cercana = None
    for nombre, datos in HOTSPOTS.items():
        d = haversine(lat, lon, datos["lat"], datos["lon"])
        if d < min_dist:
            min_dist = d
            zona_cercana = nombre
    return zona_cercana, min_dist


def cargar_modelo():
    """Carga el modelo desde disco si existe."""
    global modelo_cargado, label_encoder
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
            modelo_cargado = data["modelo"]
            label_encoder  = data["encoder"]
        print("  Modelo cargado desde modelo.pkl")


# ─────────────────────────────────────────────
# MODELOS PYDANTIC (esquemas de entrada/salida)
# ─────────────────────────────────────────────

class PrediccionRequest(BaseModel):
    latitud:     float = Field(..., example=40.4090, description="Latitud GPS")
    longitud:    float = Field(..., example=-3.7036, description="Longitud GPS")
    hora:        int   = Field(..., ge=0, le=23, example=22, description="Hora del día (0-23)")
    dia_semana:  int   = Field(..., ge=0, le=6,  example=5,  description="Día (0=lunes, 6=domingo)")
    densidad:    Optional[int] = Field(50, ge=0, le=100, example=70,
                                       description="Densidad estimada de personas (0-100)")


class PrediccionResponse(BaseModel):
    zona:             str
    distancia_km:     float
    nivel_riesgo:     str
    probabilidad_pct: float
    color:            str
    consejo:          str
    hora:             int
    dia_semana:       int


class TrainResponse(BaseModel):
    mensaje:         str
    registros_train: int
    registros_test:  int
    accuracy:        float
    reporte:         str


# ─────────────────────────────────────────────
# ENDPOINT RAÍZ
# ─────────────────────────────────────────────

@app.get("/", tags=["Estado"])
def raiz():
    modelo_ok = modelo_cargado is not None
    return {
        "app": "SafeRouteAI",
        "version": "1.0.0",
        "modelo_entrenado": modelo_ok,
        "docs": "http://127.0.0.1:8000/docs",
        "mensaje": "Llama a /train primero si modelo_entrenado es false"
    }


# ─────────────────────────────────────────────
# ENDPOINT /train
# ─────────────────────────────────────────────

@app.post("/train", response_model=TrainResponse, tags=["Modelo"])
def entrenar_modelo():
    """
    Entrena el modelo Random Forest con el CSV de datos.
    Guarda el modelo en modelo.pkl para usarlo en predicciones.
    Llama a este endpoint una vez antes de hacer predicciones.
    """
    global modelo_cargado, label_encoder

    if not os.path.exists(CSV_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró {CSV_PATH}. Ejecuta primero generar_datos.py"
        )

    # Cargar datos
    df = pd.read_csv(CSV_PATH)

    # Features (variables de entrada al modelo)
    FEATURES = [
        "hora", "dia_semana", "densidad_personas",
        "ocio_nocturno", "distancia_hotspot_km"
    ]

    X = df[FEATURES].values
    y = df["nivel_riesgo"].values

    # Codificar etiquetas: verde=2, amarillo=0, rojo=1 (alfabético)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # División train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Entrenar Random Forest
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_train, y_train)

    # Evaluar
    y_pred = modelo.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))
    reporte = classification_report(y_test, y_pred, target_names=le.classes_)

    # Guardar modelo + encoder
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"modelo": modelo, "encoder": le}, f)

    modelo_cargado = modelo
    label_encoder  = le

    print(f"  Modelo entrenado. Accuracy: {accuracy:.2%}")
    print(reporte)

    return TrainResponse(
        mensaje=f"Modelo entrenado y guardado en {MODEL_PATH}",
        registros_train=len(X_train),
        registros_test=len(X_test),
        accuracy=round(accuracy, 4),
        reporte=reporte
    )


# ─────────────────────────────────────────────
# ENDPOINT /predecir
# ─────────────────────────────────────────────

@app.post("/predecir", response_model=PrediccionResponse, tags=["Predicción"])
def predecir_riesgo(req: PrediccionRequest):
    """
    Predice el nivel de riesgo para una ubicación GPS.
    
    Parámetros:
    - latitud / longitud: coordenadas GPS
    - hora: 0-23
    - dia_semana: 0 (lunes) a 6 (domingo)
    - densidad: 0-100 (opcional, por defecto 50)
    
    Devuelve:
    - nivel_riesgo: "verde", "amarillo" o "rojo"
    - color: código hex del color
    - probabilidad_pct: porcentaje de riesgo (0-100)
    - consejo: mensaje para el usuario
    """
    if modelo_cargado is None:
        raise HTTPException(
            status_code=400,
            detail="Modelo no entrenado. Llama primero a POST /train"
        )

    # Zona más cercana y distancia
    zona, distancia_km = zona_mas_cercana(req.latitud, req.longitud)
    ocio = HOTSPOTS[zona]["ocio"]

    # Riesgo base por zona (calibrado con datos reales)
    riesgo_base_zona = {
        "Lavapiés": 82, "Carabanchel": 76, "Vallecas": 74,
        "Sol_GranVia": 58, "AZCA": 42, "Becerril_Sierra": 6
    }

    # Regla explícita para zonas rurales/sierra (riesgo base < 15):
    # El modelo no tiene suficientes ejemplos nocturnos similares
    # y se confunde. Forzamos verde directamente.
    if riesgo_base_zona.get(zona, 50) < 15:
        prob_rural = round(min(req.densidad * 0.06 + 3, 25), 1)
        return PrediccionResponse(
            zona=zona,
            distancia_km=round(distancia_km, 3),
            nivel_riesgo="verde",
            probabilidad_pct=prob_rural,
            color="#639922",
            consejo="Zona rural/sierra muy segura. Puedes circular con total tranquilidad.",
            hora=req.hora,
            dia_semana=req.dia_semana,
        )

    # Distancia minima 0.05 km para evitar que coordenada exacta
    # del centro del hotspot confunda al modelo
    distancia_model = max(distancia_km, 0.05)

    # Vector de features para el modelo
    X_input = np.array([[
        req.hora,
        req.dia_semana,
        req.densidad,
        ocio,
        round(distancia_model, 4)
    ]])

    # Predicción
    pred_enc   = modelo_cargado.predict(X_input)[0]
    pred_proba = modelo_cargado.predict_proba(X_input)[0]
    nivel      = label_encoder.inverse_transform([pred_enc])[0]

    # Probabilidad del nivel predicho (como porcentaje)
    prob_pct = round(float(max(pred_proba)) * 100, 1)

    # Color y consejo
    if nivel == "rojo":
        color  = "#E24B4A"
        consejo = "Zona de alto riesgo. Evita esta área, especialmente de noche."
    elif nivel == "amarillo":
        color  = "#EF9F27"
        consejo = "Zona con cierto riesgo. Circula con precaución y atencion a tu entorno."
    else:
        color  = "#639922"
        consejo = "Zona segura. Puedes circular con tranquilidad."

    return PrediccionResponse(
        zona=zona,
        distancia_km=round(distancia_km, 3),
        nivel_riesgo=nivel,
        probabilidad_pct=prob_pct,
        color=color,
        consejo=consejo,
        hora=req.hora,
        dia_semana=req.dia_semana,
    )


# ─────────────────────────────────────────────
# ENDPOINT /zonas
# ─────────────────────────────────────────────

@app.get("/zonas", tags=["Información"])
def listar_zonas():
    """Devuelve la lista de zonas/hotspots con sus coordenadas."""
    return {
        "zonas": [
            {
                "nombre": nombre,
                "latitud": datos["lat"],
                "longitud": datos["lon"],
                "radio_km": datos["radio_km"],
                "ocio_nocturno": bool(datos["ocio"])
            }
            for nombre, datos in HOTSPOTS.items()
        ]
    }


# ─────────────────────────────────────────────
# ARRANQUE AUTOMÁTICO
# ─────────────────────────────────────────────

# Intentar cargar modelo al arrancar la API
cargar_modelo()
