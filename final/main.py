from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from collections import defaultdict

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    "¿tienen devoluciones?|¿puedo devolver un producto?|¿cómo funciona la devolución?": "Sí, puedes devolver productos dentro de los 5 días.",
    "¿cuáles son los métodos de pago?|¿cómo puedo pagar?|formas de pago disponibles": "Aceptamos tarjetas, transferencias y pagos contraentrega.",
    "¿tienen cuadernos o libretas kawaii?": "- Cuaderno kawaii de tapa dura: $9.000\n- Libreta mini con forma de gatito: $6.000\n- Planificador semanal kawaii: $7.000",
    "¿qué bolígrafos o lápices kawaii tienen?": "- Bolígrafo gel con diseño de helado: $3.500\n- Lápiz con pompón o diseño cute: $2.500\n- Marcador doble punta pastel (unidad): $2.800",
    "¿qué accesorios de oficina kawaii ofrecen?": "- Estuche con diseño de animalitos: $15.000\n- Cartuchera transparente con glitter: $13.000\n- Sacapuntas con diseño de oso: $2.000\n- Tijeras pequeñas con orejitas: $5.000\n- Porta notas acrílico con diseño: $6.500",
    "¿tienen stickers o washi tape kawaii?": "- Pegatinas decorativas (hoja): $2.000\n- Stickers 3D kawaii (set): $4.500\n- Washi tape decorativo (rollo): $3.000",
    "¿cuáles son los productos más vendidos?": "- Cuaderno kawaii de tapa dura: $9.000\n- Estuche con diseño de animalitos: $15.000\n- Bolígrafo gel con diseño de helado: $3.500\n- Stickers 3D kawaii (set): $4.500\n- Planificador semanal kawaii: $7.000",
    "¿tienen agendas kawaii?": "- Planificador mensual kawaii: $8.000\n- Agenda 2025 con diseño de gatitos: $10.000\n- Agenda con separadores y stickers: $9.500",
    "¿tienen planificadores kawaii?": "- Planificador semanal kawaii: $7.000\n- Planificador diario con diseño de unicornio: $6.500\n- Planificador de bolsillo con ilustraciones: $5.500",
    "gracias": "Un placer atenderte, no dudes en escribirnos para resolver tus dudas. También contamos con whatsapp y formulario de contácto.",
    "¿Deseas terminar la conversación?": "Un placer atenderte, no dudes en escribirnos para resolver tus dudas. También contamos con whatsapp y formulario de contácto."
}

# Variables globales
modelo = None
index = None
preguntas_originales = []
respuestas = []
grupos_faq = []
palabras_clave_faq = defaultdict(list)

# Inicializar modelo y FAISS
def inicializar_modelo():
    global modelo, index, preguntas_originales, respuestas, grupos_faq, palabras_clave_faq

    if modelo is not None and index is not None:
        return

    preguntas = []
    respuestas.clear()
    grupos_faq.clear()
    palabras_clave_faq.clear()

    # Palabras clave mejoradas
    palabras_clave_categorias = {
        "envío": ["envío", "envio", "precio", "cuesta", "valor", "costo", "envios"],
        "devoluciones": ["devolver", "devolución", "cambio", "retorno", "devolverlo"],
        "pagos": ["pago", "métodos", "tarjeta", "transferencia", "contraentrega", "pagar"],
        "kawaii": ["kawaii", "kawali", "cawai", "lindo", "cute", "tierno", "adorable"],
        "productos": ["cuaderno", "libreta", "bolígrafo", "lápiz", "accesorio", 
                     "sticker", "washi tape", "agenda", "planificador", "producto"],
        "terminar": ["gracias", "chao", "adiós", "adios", "bye"]
    }

    for grupo, respuesta in faq_raw.items():
        grupo_splitted = grupo.split("|")
        for pregunta in grupo_splitted:
            preguntas.append(normalizar(pregunta))
            respuestas.append(respuesta)
            grupos_faq.append(grupo_splitted[0])
            
            for categoria, palabras in palabras_clave_categorias.items():
                if any(palabra in normalizar(pregunta) for palabra in palabras):
                    palabras_clave_faq[categoria].append(grupo_splitted[0])

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
    
    texto = pregunta.texto.strip().lower()
    if len(texto) < 4 or texto in ["hola", "hi", "saludos", "hey"]:
        return {
            "respuesta": "¡Hola! ¿En qué puedo ayudarte hoy? Puedes preguntarme sobre nuestros productos kawaii, envíos, métodos de pago o devoluciones."
        }

    pregunta_norm = normalizar(pregunta.texto)
    vector_pregunta = modelo.transform([pregunta_norm]).toarray().astype(np.float32)
    distancias, indices = index.search(vector_pregunta, k=3)

    mejor_distancia = distancias[0][0]
    mejor_indice = indices[0][0]

    if mejor_distancia < 0.7:
        return {"respuesta": respuestas[mejor_indice]}
    else:
        # Palabras clave mejoradas
        palabras_clave = {
            "envío": ["envío", "envio", "precio", "cuesta", "valor", "costo", "envios"],
            "devoluciones": ["devolver", "devolución", "cambio", "retorno", "devolverlo"],
            "pagos": ["pago", "métodos", "tarjeta", "transferencia", "contraentrega", "pagar"],
            "kawaii": ["kawaii", "kawali", "cawai", "lindo", "cute", "tierno", "adorable"],
            "productos": ["cuaderno", "libreta", "bolígrafo", "lápiz", "accesorio", 
                         "sticker", "washi tape", "agenda", "planificador", "producto"],
             "terminar": ["gracias", "chao","adiós", "adios", "bye"]
        }

        sugerencias = []
        for categoria, palabras in palabras_clave.items():
            if any(palabra in pregunta_norm for palabra in palabras):
                for i, pregunta_bd in enumerate(preguntas_originales):
                    if any(palabra in pregunta_bd for palabra in palabras):
                        tema = grupos_faq[i]
                        if tema not in sugerencias:
                            sugerencias.append(tema)
                            if len(sugerencias) >= 3:
                                break
                if len(sugerencias) >= 3:
                    break

        if sugerencias:
            return {
                "respuesta": "No encontré una coincidencia exacta, pero quizás te interese:",
                "sugerencias": sugerencias[:3]
            }
        else:
            sugerencias_generales = [
                "¿Quieres saber sobre el precio de envío?",
                "¿Necesitas información sobre cómo devolver un producto?",
                "¿Buscas nuestros productos kawaii como cuadernos o accesorios?",
                "¿Deseas terminar la conversacion?"
            ]
            return {
                "respuesta": "No pude entender tu pregunta. Aquí tienes algunas opciones:",
                "sugerencias": sugerencias_generales
            }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
