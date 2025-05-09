from fastapi import FastAPI, Request
from pydantic import BaseModel
import faiss
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

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

# Inicialización retrasada para optimizar recursos
modelo = None
index = None
preguntas_originales = []
respuestas = []

def inicializar_modelo():
    global modelo, index, preguntas_originales, respuestas
    
    # Si ya está inicializado, no hacer nada
    if modelo is not None:
        return
    
    # Vectorización
    preguntas = []
    respuestas.clear()
    for grupo, respuesta in faq_raw.items():
        for pregunta in grupo.split("|"):
            preguntas.append(normalizar(pregunta))
            respuestas.append(respuesta)
    
    preguntas_originales = preguntas.copy()
    
    print("Inicializando vectorizador TF-IDF...")
    # Usamos TF-IDF que es mucho más ligero que los modelos de transformers
    modelo = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)
    
    print("Codificando preguntas...")
    # Convertimos la matriz dispersa a densa para FAISS
    vectores = modelo.fit_transform(preguntas).toarray().astype(np.float32)
    dimension = vectores.shape[1]
    
    print("Creando índice FAISS...")
    global index
    index = faiss.IndexFlatL2(dimension)
    index.add(vectores)
    print("Sistema inicializado correctamente")

# Input del usuario
class Pregunta(BaseModel):
    texto: str

@app.get("/")
def root():
    return {"mensaje": "API de FAQs funcionando correctamente. Usa /chat para hacer preguntas."}

@app.post("/chat")
def responder_pregunta(pregunta: Pregunta):
    # Inicializar el modelo bajo demanda
    inicializar_modelo()
    
    pregunta_norm = normalizar(pregunta.texto)
    # Transformamos con el vectorizador TF-IDF
    vector_pregunta = modelo.transform([pregunta_norm]).toarray().astype(np.float32)
    distancias, indices = index.search(vector_pregunta, k=3)
    
    mejor_distancia = distancias[0][0]
    # Ajustamos el umbral para TF-IDF que tiene una escala diferente
    if mejor_distancia < 1.2:  
        return {"respuesta": respuestas[indices[0][0]]}
    else:
        sugerencias = [preguntas_originales[i] for i in indices[0]]
        return {
            "respuesta": "Lo siento, no tengo una respuesta clara para eso.",
            "sugerencias": sugerencias
        }

# Inicializar el modelo al arranque si hay suficiente memoria
if os.environ.get("INICIAR_MODELO_AL_ARRANQUE", "false").lower() == "true":
    try:
        inicializar_modelo()
    except Exception as e:
        print(f"No se pudo inicializar el modelo al arranque: {e}")
