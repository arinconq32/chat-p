from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from collections import defaultdict
import re

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

# Base de FAQs (igual que antes)
faq_raw = {
    # Envíos
    "¿cuánto cuesta el envío?|¿cuál es el valor del envío?|precio del envío|envíos|costos de envío|información de envíos": "En principio, el valor del envío depende de la distancia de tu ubicación. No obstante, en general el envío cuesta $10.000 a todo Bogotá.",

    # Devoluciones
    "¿tienen devoluciones?|¿puedo devolver un producto?|¿cómo funciona la devolución?|devoluciones|política de devoluciones|información de devoluciones": "Sí, puedes devolver productos dentro de los 5 días.",

    # Pagos
    "¿cuáles son los métodos de pago?|¿cómo puedo pagar?|formas de pago disponibles|métodos de pago|información de pagos": "Aceptamos tarjetas, transferencias, Nequi, Daviplata y pagos contraentrega.",

    # Agendas
    "¿tienen agendas kawaii?|¿qué tipos de agendas tienen?|agendas|planificadores|organizadores|información de agendas": "- Agenda Hologramada Kuromi: $22.000\n- Agenda Stitch imantada: $20.000\n- Agenda Hello Kitty imantada: $20.000\n- Agenda argollada Cinnamoroll: $13.000",

    # Mugs
    "¿qué mugs kawaii tienen?|¿tienen tazas personalizadas?|mugs|tazas|vasos|información de mugs": "- Mug Kuromi con cuchara: $30.000\n- Mug Stitch: $26.000\n- Mug Capybara: $30.000\n- Mug Totoro con cuchara: $30.000\n- Mug Mickey: $30.000",

    # Maquillaje
    "¿tienen productos de maquillaje kawaii?|¿qué productos de belleza ofrecen?|maquillaje|productos de belleza|cosméticos|información de maquillaje": "- Paleta Kevin y Coco 18 tonos: $21.000\n- Set de brochas profesionales: $20.000\n- Serum Bioaqua Centella: $8.000\n- Paleta Barbie 82 tonos: $52.000",

    # Bolsos
    "¿tienen bolsos kawaii?|¿qué mochilas tienen?|bolsos|mochilas|carteras|información de bolsos": "- Bolso Siliconado Zorro: $20.000\n- Mochila Peluche Ositos Cariñositos: $40.000\n- Cartera Cinnamoroll: $36.000\n- Bolso Stitch: $39.000",

    # Juguetes
    "¿tienen juguetes kawaii?|¿qué peluches tienen?|peluches|juguetes|muñecos|información de peluches": "- Peluche Angela - Stitch: $25.000\n- Tiburón Robotizado: $44.000\n- Dragón Robotizado: $24.000\n- Panda Robotizado: $25.000",

    # Papelería
    "¿tienen productos de papelería kawaii?|¿qué cuadernos tienen?|papelería|cuadernos|información de papelería": "- Set 80 marcadores Offiesco: $130.000\n- Set 12 marcadores doble punta: $22.000\n- Colores Prismacolor x18: $20.000",

    # Monederos
    "¿tienen monederos kawaii?|¿qué billeteras tienen?|monederos|billeteras|información de monederos": "- Monedero Lentejuelas: $10.000\n- Billetera Kuromi: $26.000\n- Billetera Death Note: $26.000\n- Monedero redondo Capybara: $14.000",

    # Lámparas
    "¿tienen lámparas kawaii?|¿qué lámparas decorativas tienen?|lámparas|información de lámparas": "- Lámpara astronauta sobre luna: $30.000\n- Lámpara conejo: $25.000\n- Lámpara capybara alien: $35.000",

    # Cartucheras
    "¿tienen cartucheras kawaii?|¿qué estuches tienen?|cartucheras|estuches|información de cartucheras": "- Cartuchera Capybara: $25.000\n- Cartuchera Gato: $16.000\n- Cartuchera Peluche Sanrio: $15.000",

    # Termos
    "¿tienen termos kawaii?|¿qué termos tienen?|termos|botellas|portabebidas|información de termos": "- Termos Kawaii (Rosa, Morado, Negro, Verde): $18.000 cada uno",

    # Más vendidos
    "¿cuáles son los productos más vendidos?|productos más vendidos|top ventas": "- Mug Kuromi con cuchara: $30.000\n- Agenda Hologramada Kuromi: $22.000\n- Billetera Kuromi: $26.000\n- Set de brochas profesionales: $20.000",

    # Productos específicos
    "¿tienen el Mug Kuromi con cuchara?|¿el Mug Kuromi tiene tapa?|mug kuromi|detalles mug kuromi": "Sí, el Mug Kuromi con cuchara ($30.000) incluye tapa de madera y cuchara metálica. ¡Es perfecto para bebidas calientes! 🍵✨",
    
    "¿qué diseños tienen de la Agenda Hologramada Kuromi?|¿la Agenda Kuromi es argollada?|agenda kuromi|detalles agenda kuromi": "La Agenda Hologramada Kuromi ($22.000) tiene efecto brillante, es argollada e incluye 4 separadores. Disponible en diseños: Lenticular 1 y 2.",
    
    "¿el Bolso Stitch es de peluche?|¿qué tamaño tiene el Bolso Stitch?|bolso stitch|detalles bolso stitch": "¡Sí! El Bolso Stitch ($39.000) es de peluche, mide 30x32 cm y tiene correa para cargarlo de medio lado. 🎒💙",
    
    "¿el Dragón Robotizado hace sonidos?|¿qué colores tiene el Dragón Robotizado?|dragón robotizado|detalles dragón robotizado": "Sí, el Dragón Robotizado ($24.000) emite sonidos, se desplaza y luce en verde o naranja. ¡Incluye baterías! 🐉🔊",

    # Inventario y disponibilidad
    "¿tienen en stock el Mug Stitch?|¿el Mug Stitch está disponible?|disponibilidad mug stitch|stock mug stitch": "Actualmente tenemos 3 unidades del Mug Stitch ($26.000) en inventario. ¡Pide el tuyo antes de que se agote! ☕",
    
    "¿cuándo llegan más Agendas Cinnamoroll?|¿la Agenda Cinnamoroll está agotada?|disponibilidad agenda cinnamoroll|stock agenda cinnamoroll": "La Agenda argollada Cinnamoroll ($13.000) está en reposición. ¡Puedes preordenarla! Escribe tu email para avisarte. ✨",

    # Personalización y opciones
    "¿los Termos Kawaii son para bebidas frías?|¿qué capacidad tienen los Termos Kawaii?|detalles termos kawaii|características termos": "Los Termos Kawaii ($18.000) son ideales para bebidas frías. Vienen en 4 diseños: rosa, morado, negro y verde. 🧊",
    
    "¿las Pestañas Engol son reutilizables?|¿qué medidas tienen las pestañas 3D?|pestañas engol|detalles pestañas": "Las Pestañas Efecto 3D Engol ($10.000) son reutilizables. Medidas disponibles: 3D-02, 3D-04, 3D-17 y 3D-24. 👁️✨",

    # Envíos y logística
    "¿hacen envíos el mismo día?|¿cuánto tarda el envío?|envíos express|tiempo de envío": "Envíos express en 24 hrs (solo Bogotá). Para otras ciudades: 2-3 días hábiles. 📦⏳",
    
    "¿puedo recoger en tienda?|¿tienen local físico?|recoger en tienda|local físico": "Sí, puedes recoger en nuestro local en Bogotá (Calle Kawaii #123). Horario: L-V de 9 AM a 6 PM. 🏪📍",

    # Promociones y descuentos
    "¿tienen descuento por compra mayorista?|¿hacen precios por cantidad?|descuentos|compras mayoristas": "¡Sí! Descuentos del 10% en compras mayores a $200.000. Contáctanos por WhatsApp para pedidos especiales. 💰📲",

    # Garantías y cuidados
    "¿el Serum Bioaqua es para piel sensible?|¿el Serum Centella sirve para acne?|serum bioaqua|detalles serum": "El Serum Bioaqua Centella ($8.000) es hipoalergénico, ideal para pieles sensibles y ayuda a reducir el acné. 🌿💧",
    
    "¿las Brochas profesionales son sintéticas?|¿qué incluye el set de brochas?|brochas profesionales|detalles brochas": "El Set de Brochas ($20.000) incluye 9 piezas sintéticas para maquillaje profesional. ¡Incluye estuche! 🖌️🎨",

    # Preguntas técnicas
    "¿la Licuadora Portátil es recargable?|¿qué incluye la Licuadora Portátil?|licuadora portátil|detalles licuadora": "La Licuadora Portátil ($26.000) es recargable e incluye vaso de vidrio y cuchillas de acero inoxidable. 🥤🔋",
    
    "¿el Peluche Angela es lavable?|¿qué material es el Peluche Stitch?|peluche angela|cuidado peluches": "El Peluche Angela ($25.000) es de felpa suave y se puede lavar a mano. ¡No usar secadora! 🧼🧸",

    # Productos específicos adicionales
    "¿el Mug Capybara trae tapa incluida?|¿qué incluye el Mug Capybara?|mug capybara|detalles mug capybara": "¡Así es! El Mug Capybara ($30.000) incluye tapa de porcelana y cuchara de acero. Perfecto para mantener tus bebidas calientes. 🦝☕",
    
    "¿la Lámpara Astronauta tiene pilas?|¿cómo funciona la Lámpara Astronauta?|lámpara astronauta|detalles lámpara": "La Lámpara Astronauta ($30.000) funciona con baterías recargables (incluidas) y tiene luz LED ajustable. ¡Ideal para noches de lectura! 👨‍🚀🌙",
    
    "¿qué diferencia hay entre las agendas imantadas y argolladas?|¿cuál agenda recomiendan?|comparación agendas|tipos de agendas": """
    Diferencias clave:
    - Agendas Imantadas ($20.000): Cierre seguro con imán, hojas amarillas
    - Agendas Argolladas ($13.000): Más hojas (80), diseño cuadriculado
    ¡Recomendamos la imantada para llevar siempre contigo! 📓💫
    """,
    
    "¿puedo personalizar un producto?|¿hacen productos a pedido?|personalización|productos a pedido": "Actualmente no ofrecemos personalización, pero tenemos más de 50 diseños kawaii para elegir. ¡Seguro encuentras tu favorito! ✨🎨",
    
    "¿el Peluche Tiburón canta Baby Shark?|¿qué canciones tiene el Tiburón Robotizado?|tiburón robotizado|detalles tiburón": "¡Exacto! El Tiburón Robotizado ($44.000) canta Baby Shark y se mueve al ritmo. ¡Diversión garantizada para los pequeños! 🦈🎶",
    
    "¿las Brochas profesionales son para maquillaje base?|¿qué tipos de brochas incluye el set?|tipos de brochas|uso brochas": "El Set Profesional ($20.000) incluye:\n- 2 brochas para base\n- 3 para sombras\n- 1 para rubor\n- 1 para contorno\n- 2 para difuminar. ¡Kit completo! 💄👩‍🎨",
    
    "¿el Monedero de Lentejuelas es resistente?|¿de qué material está hecho?|monedero lentejuelas|detalles monedero": "El Monedero Lentejuelas ($10.000) tiene base de tela resistente con lentejuelas cosidas. ¡Brillante y duradero! ✨👛",
    
    "¿qué tan grande es la Mochila Ositos Cariñositos?|¿caben libros en la mochila?|mochila ositos|tamaño mochila": "Mide 30x32 cm. ¡Sí! Tiene capacidad para:\n- 2-3 libros medianos\n- Estuche\n- Lonchera. Perfecta para el cole. 🎒📚",
    
    "¿el Termo Kawaii mantiene el frío?|por cuántas horas?|termo kawaii|características termo": "¡Claro! Nuestros Termos ($18.000) mantienen bebidas frías por 12 horas y calientes por 6 horas. ¡Compañero ideal! ❄️⏱️",
    
    "¿las Pestañas punto a punto son cómodas?|¿se sienten pesadas?|pestañas punto a punto|comodidad pestañas": "Las Pestañas punto a punto ($10.000) son ultra ligeras y flexibles. ¡No sentirás que las llevas puestas! 👁️💕",
    
    "¿la Cartuchera Sanrio tiene varios compartimentos?|cuántos bolsillos tiene|cartuchera sanrio|detalles cartuchera": "La Cartuchera Peluche Sanrio ($15.000) tiene:\n- 1 compartimento principal\n- 2 bolsillos laterales\n- 1 red para lápices. ¡Super práctica! ✏️🦊",
    
    "¿el Serum de Vitamina C huele bien?|¿tiene fragancia?|serum vitamina c|detalles serum": "El Serum Vitamina C ($8.000) tiene un suave aroma a cítricos naturales (sin perfumes artificiales). ¡Refrescante! 🍊🌿",
    
    "¿qué incluye el Set de Marcadores Offiesco?|vienen con punta fina y gruesa|set marcadores|detalles marcadores": "El Set de 80 marcadores ($130.000) incluye:\n- 40 colores doble punta (fina/gruesa)\n- 20 neón\n- 20 pasteles. ¡Para artistas! 🎨🖍️",
    
    "¿la Billetera Kuromi tiene espacio para tarjetas?|cuántas tarjetas caben|billetera kuromi|detalles billetera": "La Billetera Kuromi ($26.000) tiene:\n- 8 ranuras para tarjetas\n- Compartimento para billetes\n- Monedero. ¡Todo en uno! 💳👛",
    
    "¿el Peluche Robotizado Elefante camina solo?|¿necesita control remoto|elefante robotizado|detalles elefante": "El Elefante Robotizado ($25.000) se mueve automáticamente al encenderlo (no necesita control). ¡Solo ponle pilas! 🐘🔋",
    
    "¿las Lámparas Conejo tienen diferentes intensidades de luz?|¿se puede regular|lámpara conejo|detalles lámpara": "¡Sí! La Lámpara Conejo ($25.000) tiene 3 niveles de intensidad (suave/medio/fuerte). ¡Ambiente perfecto! 🐰💡",
    
    "¿el Bolso Siliconado Mario Bros es para niños?|¿qué edad recomiendan|bolso mario bros|detalles bolso": "Es ideal para:\n- Niños desde 6 años\n- Jóvenes\n- Adultos fanáticos. Tamaño universal (25x20cm). 🎮👦",
    
    "¿la Agenda Harry Potter trae stickers?|qué incluye adicional|agenda harry potter|detalles agenda": "La Agenda HP ($20.000) incluye:\n- 5 stickers temáticos\n- Marcador de páginas\n- Hoja de contactos. ¡Magia organizada! ⚡📖",
    
    "¿el Set de Brochas trae instructivo?|¿cómo saber cuál es cuál|set brochas|instrucciones brochas": "Incluye:\n- Guía impresa con usos de cada brocha\n- Numeración en los mangos\n- Estuche organizador. ¡Aprende fácil! 🖌️📝",
    
    "¿puedo lavar la Cartuchera Capybara en lavadora?|cómo limpiarla|cartuchera capybara|cuidado cartuchera": "Recomendamos:\n- Limpieza manual con paño húmedo\n- Secar al aire\n- No usar lavadora (para mantener el peluche suave). 🦦🧼",

    # Categorías generales
    "maquillaje|productos de belleza|cosméticos|información maquillaje": """
    💄 *Línea completa de maquillaje kawaii*:
    - Paletas de sombras ($10.000 a $52.000)
    - Sets de brochas profesionales ($20.000)
    - Serums faciales ($8.000)
    - Labiales y delineadores
    ¡Todos hipoalergénicos y testeados! ✨
    """,
    
    "agendas|planificadores|organizadores|información agendas": """
    📅 *Agendas para todos los gustos*:
    - Imantadas ($20.000)
    - Argolladas ($13.000-$22.000)
    - Holográficas ($22.000)
    - Con diseños: Stitch, Kuromi, Hello Kitty
    ¡Incluyen separadores y stickers! 🎀
    """,
    
    "bolsos|mochilas|carteras|información bolsos": """
    👜 *Bolsos kawaii en variedad de estilos*:
    - Siliconados ($20.000-$30.000)
    - De peluche ($35.000-$40.000)
    - Tamaños: Mediolado, bandolera, mochila
    Materiales: Peluche, silicona, cuerina
    """,
    
    "peluches|juguetes|muñecos|información peluches": """
    🧸 *Peluches interactivos y robotizados*:
    - Robotizados ($25.000-$44.000): caminan y hacen sonidos
    - Peluches clásicos ($20.000-$25.000)
    - Personajes: Stitch, Capybara, Pokémon
    ¡Incluyen baterías o son lavables! 🔋
    """,
    
    "mugs|tazas|vasos|información mugs": """
    ☕ *Mugs kawaii con accesorios*:
    - Con cuchara y tapa ($26.000-$32.000)
    - Diseños: Stitch, Kuromi, animales
    - Material: Porcelana de alta calidad
    ¡Perfectos para regalo! 🎁
    """,
    
    # Información por categoría
    "¿qué productos tienen de papelería?|cuadernos|papelería|información papelería": """
    ✏️ *Papelería kawaii*:
    - Sets de marcadores ($7.500 a $130.000)
    - Libretas ($15.000-$22.000)
    - Resaltadores pasteles
    - Stickers y washi tape (próximamente)
    """,
    
    "¿qué hay en accesorios?|productos varios|accesorios|información accesorios": """
    🎀 *Accesorios variados*:
    - Lámparas decorativas ($25.000-$35.000)
    - Cartucheras ($15.000-$25.000)
    - Joyeros ($20.000)
    - Monederos ($5.000-$14.000)
    """,
    
    "termos|botellas|portabebidas|información termos": """
    🧋 *Termos y portabebidas*:
    - Para bebidas frías ($18.000)
    - 4 diseños kawaii
    - Incluye pajilla reutilizable
    ¡Mantiene temperatura por 12 horas! ⏱️
    """,
    
    # Detalles por categoría
    "¿qué paletas de sombras tienen?|sombras|paletas sombras|información paletas": """
    🌈 *Paletas de sombras disponibles*:
    1. Kevin & Coco 18 tonos - $21.000
    2. Barbie 82 tonos - $52.000
    3. Lacalterra nude - $23.000
    4. Mini paleta x9 - $14.000
    (Todas incluyen espejo) 💝
    """,
    
    "¿qué tipos de billeteras hay?|monederos|billeteras|información billeteras": """
    💳 *Billeteras y monederos*:
    - Siliconadas ($26.000): Kuromi, Stitch
    - De peluche ($10.000-$14.000)
    - Con diseño de personajes
    ¡Algunas incluyen tarjetero! 👛
    """,
    
    "dime sobre los sets de brochas|herramientas maquillaje|sets brochas|información brochas": """
    🖌️ *Sets de brochas profesionales*:
    - Básico (9 piezas) - $20.000
    - Viajero (5 piezas) - $15.000
    - Incluyen:
      • Brochas para base/sombra
      • Difuminadores
      • Estuche protector
    """,
    
    "¿qué productos tienen para skincare?|cuidado facial|skincare|información skincare": """
    🧴 *Skincare kawaii*:
    - Serums ($8.000): Vitamina C, Ácido Hialurónico
    - Limpiadores faciales ($10.000-$20.000)
    - Contorno de ojos ($10.000)
    ¡Hipoalergénicos y sin parabenos! 🌿
    """,
    
    # Materiales y características
    "¿de qué material son los bolsos?|materiales|materiales bolsos": """
    🧵 *Materiales principales*:
    - Peluche: Suave y lavable
    - Silicona: Resistente al agua
    - Cuerina: Liviana y duradera
    - Tela: Con diseños estampados
    """,
    
    "¿los peluches son lavables?|cuidado de peluches|lavado peluches": """
    🧼 *Instrucciones de lavado*:
    - Peluches clásicos: Lavado a mano con agua fría
    - Robotizados: Solo limpieza superficial
    - Secar al aire libre
    ¡No usar secadora o blanqueadores! ⚠️
    """,
    
    "¿las agendas traen hojas cuadriculadas o rayadas?|hojas|tipo de hojas": """
    📝 *Tipos de hojas*:
    - Cuadriculadas: Agendas Stitch/Cinnamoroll
    - Rayadas: Hello Kitty/Harry Potter
    - Mixtas: Holográficas
    ¡Todas incluyen separadores temáticos! ✨
    """,
    
    # Promociones
    "¿tienen promociones en maquillaje?|ofertas belleza|promociones maquillaje": """
    💖 *Ofertas en maquillaje*:
    - 2 paletas por $35.000 (antes $42.000)
    - Serum + contorno de ojos: $15.000
    - ¡Solo hasta agotar stock!
    """,
    
    "¿hacen descuentos en bolsos?|promos accesorios|descuentos bolsos": """
    🛍️ *Promociones activas*:
    - 2 bolsos siliconados por $45.000
    - Mochila + cartuchera: $50.000
    Consulta nuestro Instagram @KawaiiShopCO 🎀
    """,
    
    # Recomendaciones
    "recomiéndame algo para regalar|ideas de regalo|recomendaciones regalos": """
    🎁 *Top regalos kawaii*:
    1. Mug + agenda ($50.000)
    2. Peluche + bolso ($60.000)
    3. Set maquillaje completo ($80.000)
    ¡Te ayudamos a elegir! 💌
    """,
    
    "¿qué me recomiendan para empezar en maquillaje?|básicos|kit básico maquillaje": """
    💫 *Kit básico ideal*:
    - Paleta nude ($21.000)
    - Set brochas ($20.000)
    - Serum hidratante ($8.000)
    Total: $49.000 (¡Ahorras $5.000!)
    """,
    
    # Marcadores
    "¿qué sets de marcadores tienen?|marcadores|sets marcadores|información marcadores": """
    ✏️ *Sets profesionales disponibles*:
    - Set 80 marcadores Offiesco doble punta: $130.000
    - Set 12 marcadores Offiesco doble punta biselados: $22.000
    - Set 6 marcadores Jumbo Offiesco: $13.000
    - Set 6 resaltadores pasteles delgados Offiesco: $7.500
    """,

    "¿los marcadores son doble punta?|tipo de punta|puntas marcadores": """
    🖍️ *Características*:
    - Los sets de 80 y 12 unidades tienen doble punta (fina/biselada)
    - Los Jumbo son de punta gruesa única
    - Resaltadores tienen punta delgada
    """,

    "¿los marcadores son permanentes?|tinta|tipo tinta marcadores": """
    🔍 *Sobre la tinta*:
    - Todos los marcadores Offiesco son a base de agua
    - No permanentes (ideales para resaltar sin dañar libros)
    - Los resaltadores pasteles son borrables
    """,

    # Colores Prismacolor
    "¿venden colores Prismacolor?|lápices profesionales|prismacolor|información prismacolor": """
    🎨 *Opción profesional*:
    - Colores Prismacolor Unipunta x18: $20.000
    - Colores doble punta Kiut: $22.000
    (Ambos sets son de alta pigmentación)
    """,

    # Resaltadores
    "¿tienen resaltadores pasteles?|resaltadores|resaltadores pasteles": """
    🌸 *Resaltadores suaves*:
    - Set x6 resaltadores pasteles delgados Offiesco: $7.500
    - Colores: rosa, azul, amarillo, verde, lila y melocotón
    """,

    # Agendas específicas
    "¿qué tipos de agendas tienen?|agendas escolares|tipos agendas": """
    📚 *Agendas disponibles*:
    - Agenda Stitch imantada: $20.000
    - Agenda Hello Kitty imantada: $20.000
    - Agenda argollada Stitch: $13.000
    - Agenda argollada Cinnamoroll: $13.000
    - Agenda Hologramada Kuromi: $22.000
    """,

    "¿las agendas traen separadores?|organización|separadores agendas": """
    🔖 *Incluyen*:
    - Todas las argolladas: 4-6 separadores
    - Hologramada Kuromi: 4 separadores a color
    - Imantadas: 1 separador básico
    """,

    # Planeadores
    "¿venden planeadores?|organizadores semanales|planeadores|información planeadores": """
    📅 *Planeadores tipo kawaii*:
    - Planeadores Kuromi (Diseño Fucsia/Lila/Rosado): $12.000 c/u
    - Planeador Kawaii-1: $20.000
    - Planeador Kawaii-2: $20.000
    (Todos con hojas semanales imantadas)
    """,

    # Cartucheras específicas
    "¿qué cartucheras tienen?|estuches|cartucheras|información cartucheras": """
    🏷️ *Cartucheras disponibles*:
    - Cartuchera Capybara (peluche): $25.000
    - Cartuchera Gato (4 diseños): $16.000
    - Cartuchera Sanrio (My Melody): $15.000
    - Cosmetiquera vinilo + peluche: $17.000
    """,

    "¿de qué material son las cartucheras?|materiales|materiales cartucheras": """
    🧵 *Según modelo*:
    - Peluche: Capybara/Sanrio (lavables)
    - Sintético: Cartuchera Gato (resistente)
    - Vinilo: Cosmetiquera (transparente)
    """,

    "¿qué tamaño tienen las cartucheras?|medidas|tamaño cartucheras": """
    📏 *Medidas aproximadas*:
    - Capybara: 21x12 cm
    - Gato: 17x10 cm
    - Sanrio: 10x16 cm
    - Cosmetiquera: 20x10 cm
    """,

    # Regalos
    "¿qué set de papelería recomiendan para regalo?|kits|sets regalo|recomendaciones regalo": """
    🎁 *Combos perfectos*:
    1. Agenda Hologramada + Set 12 marcadores: $44.000
    2. Planeador Kuromi + 6 resaltadores: $19.500
    3. Cartuchera Gato + Colores Prismacolor: $36.000
    """,

    # Stock
    "¿tienen el Set de 80 marcadores en stock?|disponibilidad|stock marcadores": """
    📦 *Inventario actual*:
    - Set 80 marcadores: Disponible (últimas 2 unidades)
    - Tiempo de reposición: 15 días hábiles
    """,

    # Novedades
    "¿tienen productos nuevos en papelería?|lanzamientos|novedades papelería": """
    🆕 *Recientemente agregados*:
    - Set 6 resaltadores pasteles Offiesco ($7.500)
    - Colores doble punta Kiut ($22.000)
    - Agenda Hologramada Kuromi versión 2 ($22.000)
    """,

    # Comparativos
    "¿qué diferencia hay entre los colores Prismacolor y Kiut?|comparación|comparativa prismacolor kiut": """
    🔍 *Diferencias clave*:
    | Característica | Prismacolor | Kiut |
    |---------------|------------|------|
    | Tipo | Unipunta | Doble punta |
    | Cantidad | 18 | 12 |
    | Precio | $20.000 | $22.000 |
    | Ideal para | Detalles | Bocetos rápidos |
    """,
    
    # Despedidas
    "gracias|muchas gracias|agradecimiento": "¡Gracias a ti! 💖 Si necesitas algo más, aquí estaremos. Un placer atenderte, no dudes en escribirnos para resolver tus dudas. También contamos con whatsapp y formulario de contacto.",
    "adiós|hasta luego|chao|despedida": "¡Que tengas un día adorable! 🌸 No olvides visitar nuestra web para más novedades. ¡Hasta pronto! 🛍️",
    "¿Deseas terminar la conversación?|terminar conversación": "Un placer atenderte, no dudes en escribirnos para resolver tus dudas. También contamos con whatsapp y formulario de contacto.",
}


# Variables globales
modelo = None
index = None
preguntas_originales = []
respuestas = []
grupos_faq = []
indice_invertido = defaultdict(list)  # Índice invertido para búsqueda rápida
preguntas_completas = []  # Almacena todas las variantes de preguntas

# Función para extraer palabras clave principales
def extraer_palabras_clave(texto):
    texto = re.sub(r'[^\w\s]', '', texto)
    palabras = texto.lower().split()
    
    stopwords = {'que', 'como', 'cual', 'cuanto', 'tienen', 'tiene', 'hay', 
                'para', 'por', 'con', 'los', 'las', 'del', 'de', 'la', 'el', 
                'en', 'un', 'una', 'y', 'o', 'es', 'son', 'me', 'mi', 'tu', 
                'su', 'nos', 'se', 'te', 'le', 'les', 'nosotros', 'ustedes'}
    
    palabras_clave = [palabra for palabra in palabras if palabra not in stopwords and len(palabra) > 3]
    
    return palabras_clave[:3]

# Inicializar modelo y FAISS
def inicializar_modelo():
    global modelo, index, preguntas_originales, respuestas, grupos_faq, indice_invertido, preguntas_completas

    if modelo is not None and index is not None:
        return

    preguntas = []
    respuestas.clear()
    grupos_faq.clear()
    indice_invertido.clear()
    preguntas_completas.clear()

    for grupo, respuesta in faq_raw.items():
        grupo_splitted = grupo.split("|")
        primera_pregunta = grupo_splitted[0]
        
        for pregunta in grupo_splitted:
            pregunta_norm = normalizar(pregunta)
            preguntas.append(pregunta_norm)
            preguntas_completas.append(pregunta)  # Guardamos la pregunta original
            respuestas.append(respuesta)
            grupos_faq.append(primera_pregunta)
            
            palabras_clave = extraer_palabras_clave(pregunta)
            for palabra in palabras_clave:
                indice_invertido[palabra].append(len(preguntas)-1)

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
            "respuesta": "¡Hola! ¿En qué puedo ayudarte hoy? Puedes preguntarme sobre nuestros productos kawaii, envíos, métodos de pago o devoluciones. Estoy para ayudarte con preguntas frecuentes, para dudas más específicas por favor contáctanos por whatsapp o el formulario de contacto."
        }

    pregunta_norm = normalizar(pregunta.texto)
    vector_pregunta = modelo.transform([pregunta_norm]).toarray().astype(np.float32)
    distancias, indices = index.search(vector_pregunta, k=3)

    mejor_distancia = distancias[0][0]
    mejor_indice = indices[0][0]

    if mejor_distancia < 0.7:
        return {"respuesta": respuestas[mejor_indice]}
    else:
        palabras_clave_usuario = extraer_palabras_clave(pregunta.texto)
        
        # Buscar en el índice invertido
        indices_coincidentes = set()
        for palabra in palabras_clave_usuario:
            if palabra in indice_invertido:
                for idx in indice_invertido[palabra]:
                    indices_coincidentes.add(idx)
        
        # Ordenar por relevancia
        preguntas_coincidentes = []
        for idx in indices_coincidentes:
            count = sum(1 for palabra in palabras_clave_usuario if palabra in preguntas_originales[idx])
            preguntas_coincidentes.append((count, idx))
        
        preguntas_coincidentes.sort(reverse=True, key=lambda x: x[0])
        
        # Obtener las preguntas completas (no normalizadas)
        sugerencias_preguntas = []
        temas_vistos = set()
        for count, idx in preguntas_coincidentes:
            tema = grupos_faq[idx]
            if tema not in temas_vistos:
                temas_vistos.add(tema)
                sugerencias_preguntas.append(preguntas_completas[idx])
                if len(sugerencias_preguntas) >= 3:
                    break
        
        # Si no hay suficientes, agregar sugerencias generales
        if len(sugerencias_preguntas) < 3:
            categorias_generales = {
                "envío": ["envío", "envio", "precio", "cuesta", "valor", "costo", "envios"],
                "devoluciones": ["devolver", "devolución", "cambio", "retorno", "devolverlo"],
                "pagos": ["pago", "métodos", "tarjeta", "transferencia", "contraentrega", "pagar"],
                "kawaii": ["kawaii", "kawali", "cawai", "lindo", "cute", "tierno", "adorable"],
                "productos": ["cuaderno", "libreta", "bolígrafo", "lápiz", "accesorio", 
                             "sticker", "washi tape", "agenda", "planificador", "producto"]
            }
            
            for categoria, palabras in categorias_generales.items():
                if any(palabra in pregunta_norm for palabra in palabras):
                    for i, pregunta_bd in enumerate(preguntas_originales):
                        if any(palabra in pregunta_bd for palabra in palabras):
                            tema = grupos_faq[i]
                            if tema not in temas_vistos:
                                temas_vistos.add(tema)
                                sugerencias_preguntas.append(preguntas_completas[i])
                                if len(sugerencias_preguntas) >= 3:
                                    break
                    if len(sugerencias_preguntas) >= 3:
                        break

        if sugerencias_preguntas:
            return {
                "respuesta": "No encontré una coincidencia exacta, pero quizás te interese:",
                "sugerencias": sugerencias_preguntas[:3]
            }
        else:
            sugerencias_generales = [
                "¿Cuánto cuesta el envío?",
                "¿Puedo devolver un producto?",
                "¿Qué productos kawaii tienen?"
            ]
            return {
                "respuesta": "No pude entender tu pregunta. Aquí tienes algunas opciones:",
                "sugerencias": sugerencias_generales
            }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
