# SafeRouteAI · Backend

API en Python (FastAPI) que predice el nivel de riesgo de una zona de Madrid (verde, amarillo o rojo) según la hora y el día. La consume la app Android de SafeRouteAI, mi proyecto final del ciclo de Desarrollo de Aplicaciones Multiplataforma (DAM, junio de 2026).

> **Sobre los datos:** el modelo se entrena con 10.000 registros **simulados** que genera `generar_datos.py` a partir de 6 zonas y reglas escritas por mí (hora, día, densidad y distancia). **No son datos delictivos reales** y las predicciones no deben usarse para decisiones de seguridad. Es un proyecto de aprendizaje.
>
> **Sobre el desarrollo:** lo he hecho con apoyo intensivo de IA, sobre todo esta parte en Python, que no formaba parte de mi ciclo.

## Qué hace

- Recibe unas coordenadas GPS, una hora y un día de la semana.
- Busca la zona conocida más cercana (fórmula de Haversine).
- Un modelo Random Forest (scikit-learn) devuelve el nivel de riesgo, un porcentaje y un consejo.
- Zonas cubiertas: Lavapiés, Carabanchel, Vallecas, Sol/Gran Vía, AZCA y Becerril de la Sierra. Cualquier punto se asigna a la más cercana. Para Becerril, una regla fija devuelve siempre verde.

## Tecnologías

Python · FastAPI · scikit-learn · pandas · NumPy

## Cómo ejecutarlo

```bash
python -m venv .venv
.venv\Scripts\activate          # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload
```

La documentación interactiva queda en <http://127.0.0.1:8000/docs>. Si `modelo.pkl` no existe, llama una vez a `POST /train`. Para regenerar los datos: `python generar_datos.py`. Con la API en marcha, `python test_api.py` la prueba desde la terminal (necesita `pip install requests`).

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/` | Estado de la API y del modelo |
| POST | `/train` | Entrena el modelo con el CSV y lo guarda en `modelo.pkl` |
| POST | `/predecir` | Predice el riesgo para unas coordenadas, hora y día |
| GET | `/zonas` | Lista las zonas con sus coordenadas |

Ejemplo de petición a `/predecir`:

```json
{ "latitud": 40.4090, "longitud": -3.7036, "hora": 22, "dia_semana": 5, "densidad": 70 }
```

## Limitaciones

- Datos simulados y solo 6 zonas (ver arriba).
- El modelo aprende las reglas con las que se generaron los datos, no patrones reales.
- La demo está en un servidor gratuito: la primera respuesta puede tardar al despertar el servicio.

## Autor

Cristian Corroto · Web de descarga de la app: <https://saferouteai-app.netlify.app/>
