from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = FastAPI()

# Configurar CORS - Modificado para aceptar cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir cualquier origen para que funcione con tu WordPress
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Normalizador
def normalizar(texto: str) -> str:
    texto = texto.lower().strip()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    return texto

# Base de FAQs
faq_raw = {
    "¿cuánto cuesta el envío?|¿cuál es el valor del envío?|precio del envío": "El envío cuesta $10.000 a todo el país.",
    "¿tienen devoluciones?|¿puedo devolver un producto?|¿cómo funciona la devolución?": "Sí, puedes devolver productos dentro de los 7 días.",
    "¿cuáles son los métodos de pago?|¿cómo puedo pagar?|formas de pago disponibles": "Aceptamos tarjetas, transferencias y pagos contraentrega.",
    "¿tienen cuadernos o libretas kawaii?": "- Cuaderno kawaii de tapa dura: $9.000\n- Libreta mini con forma de gatito: $6.000\n- Planificador semanal kawaii: $7.000",
    "¿qué bolígrafos o lápices kawaii tienen?": "- Bolígrafo gel con diseño de helado: $3.500\n- Lápiz con pompón o diseño cute: $2.500\n- Marcador doble punta pastel (unidad): $2.800",
    "¿qué accesorios de oficina kawaii ofrecen?": "- Estuche con diseño de animalitos: $15.000\n- Cartuchera transparente con glitter: $13.000\n- Sacapuntas con diseño de oso: $2.000\n- Tijeras pequeñas con orejitas: $5.000\n- Porta notas acrílico con diseño: $6.500",
    "¿tienen stickers o washi tape kawaii?": "- Pegatinas decorativas (hoja): $2.000\n- Stickers 3D kawaii (set): $4.500\n- Washi tape decorativo (rollo): $3.000",
    "¿cuáles son los productos más vendidos?": "- Cuaderno kawaii de tapa dura: $9.000\n- Estuche con diseño de animalitos: $15.000\n- Bolígrafo gel con diseño de helado: $3.500\n- Stickers 3D kawaii (set): $4.500\n- Planificador semanal kawaii: $7.000",
    "¿tienen agendas kawaii?": "- Planificador mensual kawaii: $8.000\n- Agenda 2025 con diseño de gatitos: $10.000\n- Agenda con separadores y stickers: $9.500",
    "¿tienen planificadores kawaii?": "- Planificador semanal kawaii: $7.000\n- Planificador diario con diseño de unicornio: $6.500\n- Planificador de bolsillo con ilustraciones: $5.500"
}

# Variables globales
modelo = None
index = None
preguntas_originales = []
respuestas = []

# Inicializar modelo y FAISS
def inicializar_modelo():
    global modelo, index, preguntas_originales, respuestas

    if modelo is not None and index is not None:
        return

    preguntas = []
    respuestas.clear()
    for grupo, respuesta in faq_raw.items():
        for pregunta in grupo.split("|"):
            preguntas.append(normalizar(pregunta))
            respuestas.append(respuesta)

    preguntas_originales[:] = preguntas

    modelo = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)
    vectores = modelo.fit_transform(preguntas).toarray().astype(np.float32)
    dimension = vectores.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectores)

# Modelo de entrada
class Pregunta(BaseModel):
    texto: str

@app.get("/")
def root():
    return {"mensaje": "API de FAQs funcionando correctamente. Usa /chat para hacer preguntas."}

@app.post("/chat")
def responder_pregunta(pregunta: Pregunta):
    inicializar_modelo()
    
    # Verifica si el texto es demasiado corto o un simple saludo
    texto = pregunta.texto.strip().lower()
    if len(texto) < 4 or texto in ["hola", "hi", "saludos", "hey"]:
        return {
            "respuesta": "¡Hola! ¿En qué puedo ayudarte hoy? Puedes preguntarme sobre nuestros productos kawaii, envíos, devoluciones y más."
        }

    pregunta_norm = normalizar(pregunta.texto)
    vector_pregunta = modelo.transform([pregunta_norm]).toarray().astype(np.float32)
    distancias, indices = index.search(vector_pregunta, k=3)

    # Umbral más estricto (0.7 en lugar de 1.2)
    mejor_distancia = distancias[0][0]
    if mejor_distancia < 0.7:
        return {"respuesta": respuestas[indices[0][0]]}
    else:
        # Verificar si hay algunas palabras clave para ofrecer sugerencias más relevantes
        palabras_clave = ["envío", "envio", "producto", "precio", "kawaii", "pago", "devolver", "devolución", "compra"]
        if any(palabra in pregunta_norm for palabra in palabras_clave):
            sugerencias = [preguntas_originales[i] for i in indices[0]]
            return {
                "respuesta": "No estoy seguro exactamente de lo que preguntas. ¿Quizás te refieres a alguno de estos temas?",
                "sugerencias": sugerencias
            }
        else:
            return {
                "respuesta": "Lo siento, no entiendo tu pregunta. ¿Podrías reformularla o preguntar sobre nuestros productos kawaii, envíos, métodos de pago o devoluciones?"
            }

# Inicializar si se configura por entorno
if os.environ.get("INICIAR_MODELO_AL_ARRANQUE", "false").lower() == "true":
    try:
        inicializar_modelo()
    except Exception as e:
        print(f"No se pudo inicializar el modelo al arranque: {e}")
