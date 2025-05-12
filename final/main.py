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
    "¿cuánto cuesta el envío?|¿cuál es el valor del envío?|precio del envío": "En principio, el valor del envío depende de la distancia de tu ubicación. No obstante, en general el envío cuesta $10.000 a todo Bogotá.",
    "¿tienen devoluciones?|¿puedo devolver un producto?|¿cómo funciona la devolución?": "Sí, puedes devolver productos dentro de los 5 días.",
    "¿cuáles son los métodos de pago?|¿cómo puedo pagar?|formas de pago disponibles": "Aceptamos tarjetas, transferencias, Nequi, Daviplata y pagos contraentrega.",
    "¿tienen agendas kawaii?|¿qué tipos de agendas tienen?": "- Agenda Hologramada Kuromi: $22.000\n- Agenda Stitch imantada: $20.000\n- Agenda Hello Kitty imantada: $20.000\n- Agenda argollada Cinnamoroll: $13.000",
    "¿qué mugs kawaii tienen?|¿tienen tazas personalizadas?": "- Mug Kuromi con cuchara: $30.000\n- Mug Stitch: $26.000\n- Mug Capybara: $30.000\n- Mug Totoro con cuchara: $30.000\n- Mug Mickey: $30.000",
    "¿tienen productos de maquillaje kawaii?|¿qué productos de belleza ofrecen?": "- Paleta Kevin y Coco 18 tonos: $21.000\n- Set de brochas profesionales: $20.000\n- Serum Bioaqua Centella: $8.000\n- Paleta Barbie 82 tonos: $52.000",
    "¿tienen bolsos kawaii?|¿qué mochilas tienen?": "- Bolso Siliconado Zorro: $20.000\n- Mochila Peluche Ositos Cariñositos: $40.000\n- Cartera Cinnamoroll: $36.000\n- Bolso Stitch: $39.000",
    "¿tienen juguetes kawaii?|¿qué peluches tienen?": "- Peluche Angela - Stitch: $25.000\n- Tiburón Robotizado: $44.000\n- Dragón Robotizado: $24.000\n- Panda Robotizado: $25.000",
    "¿tienen productos de papelería kawaii?|¿qué cuadernos tienen?": "- Set 80 marcadores Offiesco: $130.000\n- Set 12 marcadores doble punta: $22.000\n- Colores Prismacolor x18: $20.000",
    "¿tienen monederos kawaii?|¿qué billeteras tienen?": "- Monedero Lentejuelas: $10.000\n- Billetera Kuromi: $26.000\n- Billetera Death Note: $26.000\n- Monedero redondo Capybara: $14.000",
    "¿tienen lámparas kawaii?|¿qué lámparas decorativas tienen?": "- Lámpara astronauta sobre luna: $30.000\n- Lámpara conejo: $25.000\n- Lámpara capybara alien: $35.000",
    "¿tienen cartucheras kawaii?|¿qué estuches tienen?": "- Cartuchera Capybara: $25.000\n- Cartuchera Gato: $16.000\n- Cartuchera Peluche Sanrio: $15.000",
    "¿tienen termos kawaii?|¿qué termos tienen?": "- Termos Kawaii (Rosa, Morado, Negro, Verde): $18.000 cada uno",
    "¿cuáles son los productos más vendidos?": "- Mug Kuromi con cuchara: $30.000\n- Agenda Hologramada Kuromi: $22.000\n- Billetera Kuromi: $26.000\n- Set de brochas profesionales: $20.000",
   
       # Productos específicos
    "¿tienen el Mug Kuromi con cuchara?|¿el Mug Kuromi tiene tapa?": "Sí, el Mug Kuromi con cuchara ($30.000) incluye tapa de madera y cuchara metálica. ¡Es perfecto para bebidas calientes! 🍵✨",
    "¿qué diseños tienen de la Agenda Hologramada Kuromi?|¿la Agenda Kuromi es argollada?": "La Agenda Hologramada Kuromi ($22.000) tiene efecto brillante, es argollada e incluye 4 separadores. Disponible en diseños: Lenticular 1 y 2.",
    "¿el Bolso Stitch es de peluche?|¿qué tamaño tiene el Bolso Stitch?": "¡Sí! El Bolso Stitch ($39.000) es de peluche, mide 30x32 cm y tiene correa para cargarlo de medio lado. 🎒💙",
    "¿el Dragón Robotizado hace sonidos?|¿qué colores tiene el Dragón Robotizado?": "Sí, el Dragón Robotizado ($24.000) emite sonidos, se desplaza y luce en verde o naranja. ¡Incluye baterías! 🐉🔊",

    # Inventario y disponibilidad
    "¿tienen en stock el Mug Stitch?|¿el Mug Stitch está disponible?": "Actualmente tenemos 3 unidades del Mug Stitch ($26.000) en inventario. ¡Pide el tuyo antes de que se agote! ☕",
    "¿cuándo llegan más Agendas Cinnamoroll?|¿la Agenda Cinnamoroll está agotada?": "La Agenda argollada Cinnamoroll ($13.000) está en reposición. ¡Puedes preordenarla! Escribe tu email para avisarte. ✨",

    # Personalización y opciones
    "¿los Termos Kawaii son para bebidas frías?|¿qué capacidad tienen los Termos Kawaii?": "Los Termos Kawaii ($18.000) son ideales para bebidas frías. Vienen en 4 diseños: rosa, morado, negro y verde. 🧊",
    "¿las Pestañas Engol son reutilizables?|¿qué medidas tienen las pestañas 3D?": "Las Pestañas Efecto 3D Engol ($10.000) son reutilizables. Medidas disponibles: 3D-02, 3D-04, 3D-17 y 3D-24. 👁️✨",

    # Envíos y logística
    "¿hacen envíos el mismo día?|¿cuánto tarda el envío?": "Envíos express en 24 hrs (solo Bogotá). Para otras ciudades: 2-3 días hábiles. 📦⏳",
    "¿puedo recoger en tienda?|¿tienen local físico?": "Sí, puedes recoger en nuestro local en Bogotá (Calle Kawaii #123). Horario: L-V de 9 AM a 6 PM. 🏪📍",

    # Promociones y descuentos
    "¿tienen descuento por compra mayorista?|¿hacen precios por cantidad?": "¡Sí! Descuentos del 10% en compras mayores a $200.000. Contáctanos por WhatsApp para pedidos especiales. 💰📲",

    # Garantías y cuidados
    "¿el Serum Bioaqua es para piel sensible?|¿el Serum Centella sirve para acne?": "El Serum Bioaqua Centella ($8.000) es hipoalergénico, ideal para pieles sensibles y ayuda a reducir el acné. 🌿💧",
    "¿las Brochas profesionales son sintéticas?|¿qué incluye el set de brochas?": "El Set de Brochas ($20.000) incluye 9 piezas sintéticas para maquillaje profesional. ¡Incluye estuche! 🖌️🎨",

    # Preguntas técnicas
    "¿la Licuadora Portátil es recargable?|¿qué incluye la Licuadora Portátil?": "La Licuadora Portátil ($26.000) es recargable e incluye vaso de vidrio y cuchillas de acero inoxidable. 🥤🔋",
    "¿el Peluche Angela es lavable?|¿qué material es el Peluche Stitch?": "El Peluche Angela ($25.000) es de felpa suave y se puede lavar a mano. ¡No usar secadora! 🧼🧸",

    # Nuevas preguntas y respuestas
    "¿el Mug Capybara trae tapa incluida?|¿qué incluye el Mug Capybara?": "¡Así es! El Mug Capybara ($30.000) incluye tapa de porcelana y cuchara de acero. Perfecto para mantener tus bebidas calientes. 🦝☕",
    
    "¿la Lámpara Astronauta tiene pilas?|¿cómo funciona la Lámpara Astronauta?": "La Lámpara Astronauta ($30.000) funciona con baterías recargables (incluidas) y tiene luz LED ajustable. ¡Ideal para noches de lectura! 👨‍🚀🌙",
    
    "¿qué diferencia hay entre las agendas imantadas y argolladas?|¿cuál agenda recomiendan?": """
    Diferencias clave:
    - Agendas Imantadas ($20.000): Cierre seguro con imán, hojas amarillas
    - Agendas Argolladas ($13.000): Más hojas (80), diseño cuadriculado
    ¡Recomendamos la imantada para llevar siempre contigo! 📓💫
    """,
    
    "¿puedo personalizar un producto?|¿hacen productos a pedido?": "Actualmente no ofrecemos personalización, pero tenemos más de 50 diseños kawaii para elegir. ¡Seguro encuentras tu favorito! ✨🎨",
    
    "¿el Peluche Tiburón canta Baby Shark?|¿qué canciones tiene el Tiburón Robotizado?": "¡Exacto! El Tiburón Robotizado ($44.000) canta Baby Shark y se mueve al ritmo. ¡Diversión garantizada para los pequeños! 🦈🎶",
    
    "¿las Brochas profesionales son para maquillaje base?|¿qué tipos de brochas incluye el set?": "El Set Profesional ($20.000) incluye:\n- 2 brochas para base\n- 3 para sombras\n- 1 para rubor\n- 1 para contorno\n- 2 para difuminar. ¡Kit completo! 💄👩‍🎨",
    
    "¿el Monedero de Lentejuelas es resistente?|¿de qué material está hecho?": "El Monedero Lentejuelas ($10.000) tiene base de tela resistente con lentejuelas cosidas. ¡Brillante y duradero! ✨👛",
    
    "¿qué tan grande es la Mochila Ositos Cariñositos?|¿caben libros en la mochila?": "Mide 30x32 cm. ¡Sí! Tiene capacidad para:\n- 2-3 libros medianos\n- Estuche\n- Lonchera. Perfecta para el cole. 🎒📚",
    
    "¿el Termo Kawaii mantiene el frío?|por cuántas horas?": "¡Claro! Nuestros Termos ($18.000) mantienen bebidas frías por 12 horas y calientes por 6 horas. ¡Compañero ideal! ❄️⏱️",
    
    "¿las Pestañas punto a punto son cómodas?|¿se sienten pesadas?": "Las Pestañas punto a punto ($10.000) son ultra ligeras y flexibles. ¡No sentirás que las llevas puestas! 👁️💕",
    
    "¿la Cartuchera Sanrio tiene varios compartimentos?|cuántos bolsillos tiene": "La Cartuchera Peluche Sanrio ($15.000) tiene:\n- 1 compartimento principal\n- 2 bolsillos laterales\n- 1 red para lápices. ¡Super práctica! ✏️🦊",
    
    "¿el Serum de Vitamina C huele bien?|¿tiene fragancia?": "El Serum Vitamina C ($8.000) tiene un suave aroma a cítricos naturales (sin perfumes artificiales). ¡Refrescante! 🍊🌿",
    
    "¿qué incluye el Set de Marcadores Offiesco?|vienen con punta fina y gruesa": "El Set de 80 marcadores ($130.000) incluye:\n- 40 colores doble punta (fina/gruesa)\n- 20 neón\n- 20 pasteles. ¡Para artistas! 🎨🖍️",
    
    "¿la Billetera Kuromi tiene espacio para tarjetas?|cuántas tarjetas caben": "La Billetera Kuromi ($26.000) tiene:\n- 8 ranuras para tarjetas\n- Compartimento para billetes\n- Monedero. ¡Todo en uno! 💳👛",
    
    "¿el Peluche Robotizado Elefante camina solo?|¿necesita control remoto": "El Elefante Robotizado ($25.000) se mueve automáticamente al encenderlo (no necesita control). ¡Solo ponle pilas! 🐘🔋",
    
    "¿las Lámparas Conejo tienen diferentes intensidades de luz?|¿se puede regular": "¡Sí! La Lámpara Conejo ($25.000) tiene 3 niveles de intensidad (suave/medio/fuerte). ¡Ambiente perfecto! 🐰💡",
    
    "¿el Bolso Siliconado Mario Bros es para niños?|¿qué edad recomiendan": "Es ideal para:\n- Niños desde 6 años\n- Jóvenes\n- Adultos fanáticos. Tamaño universal (25x20cm). 🎮👦",
    
    "¿la Agenda Harry Potter trae stickers?|qué incluye adicional": "La Agenda HP ($20.000) incluye:\n- 5 stickers temáticos\n- Marcador de páginas\n- Hoja de contactos. ¡Magia organizada! ⚡📖",
    
    "¿el Set de Brochas trae instructivo?|¿cómo saber cuál es cuál": "Incluye:\n- Guía impresa con usos de cada brocha\n- Numeración en los mangos\n- Estuche organizador. ¡Aprende fácil! 🖌️📝",
    
    "¿puedo lavar la Cartuchera Capybara en lavadora?|cómo limpiarla": "Recomendamos:\n- Limpieza manual con paño húmedo\n- Secar al aire\n- No usar lavadora (para mantener el peluche suave). 🦦🧼",
    
    # Despedidas
    "gracias|muchas gracias": "¡Gracias a ti! 💖 Si necesitas algo más, aquí estaremos. Un placer atenderte, no dudes en escribirnos para resolver tus dudas. También contamos con whatsapp y formulario de contacto.",
    "adiós|hasta luego|chao": "¡Que tengas un día adorable! 🌸 No olvides visitar nuestra web para más novedades. ¡Hasta pronto! 🛍️",
    "¿Deseas terminar la conversación?": "Un placer atenderte, no dudes en escribirnos para resolver tus dudas. También contamos con whatsapp y formulario de contacto.",
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
