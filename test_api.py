"""
SafeRouteAI — test_api.py
==========================
Script para probar la API desde la terminal sin necesidad de Swagger.
Ejecutar con la API ya arrancada: uvicorn api:app --reload

Uso:
    python test_api.py
"""

import requests
import json

BASE = "http://127.0.0.1:8000"

def separador(titulo):
    print(f"\n{'='*50}")
    print(f"  {titulo}")
    print('='*50)

def ok(label, valor):
    print(f"  OK  {label}: {valor}")

def fallo(label, error):
    print(f"  ERR {label}: {error}")


# ─────────────────────────────────────────────
# 1. Estado de la API
# ─────────────────────────────────────────────
separador("1. Estado de la API")
try:
    r = requests.get(f"{BASE}/")
    d = r.json()
    ok("App", d["app"])
    ok("Versión", d["version"])
    ok("Modelo entrenado", d["modelo_entrenado"])
except Exception as e:
    fallo("Conexión", e)
    print("\n  IMPORTANTE: Asegúrate de que la API está corriendo:")
    print("  uvicorn api:app --reload")
    exit(1)

# ─────────────────────────────────────────────
# 2. Entrenar el modelo
# ─────────────────────────────────────────────
separador("2. Entrenando el modelo (/train)")
try:
    r = requests.post(f"{BASE}/train")
    d = r.json()
    ok("Registros entrenamiento", d["registros_train"])
    ok("Registros test", d["registros_test"])
    ok("Accuracy", f"{d['accuracy']*100:.1f}%")
    print(f"\n  Reporte de clasificación:\n")
    for linea in d["reporte"].split("\n"):
        print(f"    {linea}")
except Exception as e:
    fallo("Train", e)

# ─────────────────────────────────────────────
# 3. Predicciones de prueba por zona
# ─────────────────────────────────────────────
separador("3. Predicciones por zona (/predecir)")

casos = [
    {
        "nombre": "Lavapiés — sábado noche",
        "payload": {"latitud": 40.4090, "longitud": -3.7036,
                    "hora": 23, "dia_semana": 5, "densidad": 80}
    },
    {
        "nombre": "Lavapiés — lunes mediodía",
        "payload": {"latitud": 40.4090, "longitud": -3.7036,
                    "hora": 13, "dia_semana": 0, "densidad": 40}
    },
    {
        "nombre": "Carabanchel — viernes noche",
        "payload": {"latitud": 40.3847, "longitud": -3.7312,
                    "hora": 22, "dia_semana": 4, "densidad": 75}
    },
    {
        "nombre": "Sol/Gran Vía — domingo tarde",
        "payload": {"latitud": 40.4168, "longitud": -3.7038,
                    "hora": 17, "dia_semana": 6, "densidad": 90}
    },
    {
        "nombre": "AZCA — martes mañana",
        "payload": {"latitud": 40.4530, "longitud": -3.6913,
                    "hora": 9,  "dia_semana": 1, "densidad": 60}
    },
    {
        "nombre": "Becerril de la Sierra — cualquier momento",
        "payload": {"latitud": 40.7283, "longitud": -3.8523,
                    "hora": 22, "dia_semana": 5, "densidad": 10}
    },
    {
        "nombre": "Vallecas — madrugada",
        "payload": {"latitud": 40.3868, "longitud": -3.6513,
                    "hora": 2,  "dia_semana": 5, "densidad": 65}
    },
]

colores = {"rojo": "ROJO", "amarillo": "AMRLL", "verde": "VERDE"}

for caso in casos:
    try:
        r = requests.post(f"{BASE}/predecir", json=caso["payload"])
        d = r.json()
        nivel = d["nivel_riesgo"].upper()
        prob  = d["probabilidad_pct"]
        zona  = d["zona"]
        dist  = d["distancia_km"]
        print(f"\n  [{nivel:7s}] {caso['nombre']}")
        print(f"           Zona detectada: {zona} ({dist} km)")
        print(f"           Probabilidad:   {prob}%")
        print(f"           Consejo:        {d['consejo'][:60]}...")
    except Exception as e:
        fallo(caso["nombre"], e)

# ─────────────────────────────────────────────
# 4. Listar zonas
# ─────────────────────────────────────────────
separador("4. Zonas disponibles (/zonas)")
try:
    r = requests.get(f"{BASE}/zonas")
    zonas = r.json()["zonas"]
    for z in zonas:
        ocio = "con ocio nocturno" if z["ocio_nocturno"] else "sin ocio nocturno"
        print(f"  - {z['nombre']:20s} lat={z['latitud']}  lon={z['longitud']}  ({ocio})")
except Exception as e:
    fallo("Zonas", e)

separador("Pruebas completadas")
print("  Si ves predicciones correctas, la API esta lista.")
print("  Siguiente paso: conectar la app Android con Retrofit.\n")
