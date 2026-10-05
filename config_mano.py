"""
Configuracion de la OrcaHand para la teleoperacion del TRON 2.

Sustituye al antiguo config.py (que era del proyecto ARCTOS y traia ~400 lineas de
cosas que aqui no se usan: IK del brazo ARCTOS, taladro, fisica en hilo, hardware...).
Aqui solo esta lo que usan retargeting.py, biomecanica.py y mano_orca.py.
"""

# ── ACTUADORES Y LIMITES ANATOMICOS DE SEGURIDAD (MANO) ───────────────────────
# Orden de los 17 actuadores de la OrcaHand v2 (el mismo que su .mjcf):
#   0 muneca | 1-3 menique (abd, mcp, pip) | 4-6 anular | 7-9 corazon | 10-12 indice
#   13 pulgar cmc (oposicion) | 14 pulgar abd | 15 pulgar mcp | 16 pulgar ip
N_ACT = 17
OFFSET_MANO = 0       # (antes 6: el brazo ARCTOS ocupaba los primeros ctrl; aqui la mano va sola)

LIMITES_CTRL = [
    (-1.134,  0.610),  #  0: muneca
    (-0.20,   0.50),   #  1: menique abd
    ( 0.0,    1.80),   #  2: menique mcp
    ( 0.0,    1.80),   #  3: menique pip
    (-0.20,   0.50),   #  4: anular abd
    ( 0.0,    1.80),   #  5: anular mcp
    ( 0.0,    1.80),   #  6: anular pip
    ( 0.0,    0.0),    #  7: corazon abd (fijo)
    ( 0.0,    1.80),   #  8: corazon mcp
    ( 0.0,    1.80),   #  9: corazon pip
    (-0.50,   0.50),   # 10: indice abd
    ( 0.0,    1.80),   # 11: indice mcp
    ( 0.0,    1.80),   # 12: indice pip
    (-0.785,  1.85),   # 13: pulgar oposicion/CMC
    (-0.35,   1.35),   # 14: pulgar abd
    (-0.15,   1.80),   # 15: pulgar mcp  (permite extension activa)
    ( 0.0,    1.80),   # 16: pulgar ip
]

POSICION_STANDBY = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    -0.0979, 0.641,  0.00,  0.175
]

# Postura de "mano cerrada" para el modo sencillo (sin landmarks de los dedos), por
# indice de ctrl. Lo no indicado se queda como en POSICION_STANDBY.
MANO_CERRADA_CTRL = {0: 0.0,
                     2: 1.25, 3: 1.10,     # menique
                     5: 1.25, 6: 1.10,     # anular
                     8: 1.25, 9: 1.10,     # corazon
                     11: 1.00, 12: 0.90,   # indice
                     13: 0.35, 14: 0.45, 15: 0.70, 16: 0.90}   # pulgar rodeando

# ── MODO DE LA MANO ──────────────────────────────────────────────────────────
# True  = IMITACION: los dedos del robot copian los tuyos, sin posturas predefinidas.
# False = modo "pinzas" del proyecto ARCTOS: al juntar el pulgar con un dedo la mano
#         salta a una postura de pinza calibrada (PINZA_CTRL_CONTACTO en retargeting.py).
#         Se alterna en vivo con la tecla [g] en la ventana de la camara.
IMITACION_PURA = True

# "Puno colectivo": al cerrar el indice/anular mas de UMBRAL_PUNO, arrastra al resto de
# dedos (compensa que MediaPipe pierda las yemas tapadas dentro del puno). Con el recorte
# HD de la mano las yemas se ven mejor; si notas que los dedos se cierran "solos", ponlo a False.
PUNO_COLECTIVO = True
UMBRAL_PUNO = 0.60

# ── PARAMETROS BIOMECANICOS Y FILTROS ─────────────────────────────────────────
UMBRAL_BLOQUEO    = 0.50
UMBRAL_LIBERACION = 0.10

# Ganancias globales (los dedos de la Orca recorren mas que los tuyos en el mapeo)
FUERZA_NUDILLOS   = 1.35
FUERZA_PUNTAS     = 1.45
FUERZA_ABD        = 1.50

# Ganancias INDIVIDUALES por dedo (multiplican sobre la global)
GANANCIA_DEDOS = {
    "menique_mcp":  1.00,   # ctrl[2]
    "menique_pip":  1.00,   # ctrl[3]
    "anular_mcp":   1.10,   # ctrl[5]
    "anular_pip":   1.10,   # ctrl[6]
    "corazon_mcp":  1.05,   # ctrl[8]
    "corazon_pip":  1.05,   # ctrl[9]
    "indice_mcp":   1.00,   # ctrl[11] - referencia
    "indice_pip":   1.00,   # ctrl[12]
}

# Filtro One Euro de los 16 ctrl de los dedos (sustituye a la media movil de 4 fotogramas,
# que a ~12 fps anadia ~0.3 s de retraso): poco temblor quieto, poco retraso al moverse.
DEDOS_FILTRO_MIN_CUTOFF = 1.5   # Hz: mas bajo = menos temblor en reposo
DEDOS_FILTRO_BETA       = 0.8   # mas alto = menos retraso al abrir/cerrar deprisa

# ── UMBRALES Y GANANCIAS DEL PULGAR ───────────────────────────────────────────
UMBRAL_EXT_PULGAR_MCP = 0.15   # por debajo -> zona de extension activa
UMBRAL_EXT_PULGAR_IP  = 0.25   # por debajo -> filtra temblor de punta
PULGAR_GANANCIA_MCP   = 1.70   # amplifica flexion del nudillo
PULGAR_GANANCIA_IP    = 1.90   # amplifica flexion de la punta

# ── UMBRALES DE PINZA PREDEFINIDA (solo con IMITACION_PURA = False) ───────────
PINZA_UMBRAL_ON = {
    "indice": 0.030, "corazon": 0.040, "anular": 0.040, "menique": 0.035,
    "corazon_anular": 0.040, "indice_corazon": 0.040, "anular_menique": 0.040
}
PINZA_UMBRAL_OFF = {
    "indice": 0.040, "corazon": 0.050, "anular": 0.050, "menique": 0.045,
    "corazon_anular": 0.050, "indice_corazon": 0.050, "anular_menique": 0.050
}
UMBRAL_COMBO_ON   = {"corazon": 0.035, "anular": 0.040}
UMBRAL_COMBO_OFF  = {"corazon": 0.045, "anular": 0.050}
UMBRAL_COMBO2_ON  = {"indice": 0.040, "corazon": 0.045}
UMBRAL_COMBO2_OFF = {"indice": 0.050, "corazon": 0.055}
UMBRAL_COMBO3_ON  = {"anular": 0.040, "menique": 0.040}
UMBRAL_COMBO3_OFF = {"anular": 0.050, "menique": 0.050}
PINZA_DIST_MAX = 0.120

# ── LANDMARKS MEDIAPIPE HANDS ─────────────────────────────────────────────────
LM_MUNECA      = 0
LM_INDICE_MCP  = 5
LM_CORAZON_MCP = 9
LM_ANULAR_MCP  = 13
LM_MENIQUE_MCP = 17
