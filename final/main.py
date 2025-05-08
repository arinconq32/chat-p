from fastapi import FastAPI, Request
from pydantic import BaseModel
import faiss
import unicodedata
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Normalizador
def normalizar(texto):
    texto = texto.lower().strip()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    return texto

# FAQ
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

# Vectorización
preguntas = []
respuestas = []

for grupo, respuesta in faq_raw.items():
    for pregunta in grupo.split("|"):
        preguntas.append(normalizar(pregunta))
        respuestas.append(respuesta)

preguntas_originales = preguntas.copy()

modelo = SentenceTransformer("all-MiniLM-L6-v2")
vectores = modelo.encode(preguntas, convert_to_numpy=True)

dimension = vectores.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectores)

# Input del usuario
class Pregunta(BaseModel):
    texto: str

@app.post("/chat")
def responder_pregunta(pregunta: Pregunta):
    pregunta_norm = normalizar(pregunta.texto)
    vector_pregunta = modelo.encode([pregunta_norm], convert_to_numpy=True)

    distancias, indices = index.search(vector_pregunta, k=3)
    mejor_distancia = distancias[0][0]

    if mejor_distancia < 0.5:
        return {"respuesta": respuestas[indices[0][0]]}
    else:
        sugerencias = [preguntas_originales[i] for i in indices[0]]
        return {
            "respuesta": "Lo siento, no tengo una respuesta clara para eso.",
            "sugerencias": sugerencias
        }
