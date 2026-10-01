#!/usr/bin/env python3
"""
Skill Shield Nivel 6 - Panel de Configuración Global
===================================================
Modifica los parámetros de este archivo para alterar la simulación o el hardware
sin necesidad de tocar el código lógico principal.
"""
import os
import math
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── CONFIGURACIÓN DE ENTORNO Y RUTAS ──────────────────────────────────────────
MODO_EJECUCION = "SIMULACION"  # Opciones: "SIMULACION" | "HARDWARE_REAL"

# Puerto mano
PUERTO_HARDWARE = "/dev/ttyUSB0"

# Ruta al MJCF del brazo ARCTOS + mano OrcaHand ya fusionados
RUTA_XML  = os.path.join(_BASE_DIR, "arctos_description-main",
                          "arctos_urdf_description", "urdf", "arctos_mjcf.xml")

# Ruta al logo IDI / Grupo Elecnor (Apuntando a la nueva carpeta Nivel6)
RUTA_LOGO = os.path.join(_BASE_DIR, "orcahand_description-main", "v2", "I+D+I_DENOMIN_GRUPO_RGB.png")

# ── PARÁMETROS DE LA CÁMARA OAK-D Y PROYECCIÓN ────────────────────────────────
RES            = 300    # (LEGADO) resolución cuadrada antigua. Ya no se usa en el teleop;
                        # se deja por si algún script auxiliar (p.ej. test_ia_mano.py) la importa.

# Formato 4:3 (SENSOR COMPLETO). El sensor de color de la OAK-D (IMX378) es 4:3.
# Al pedir 1080p (16:9) el firmware RECORTA franjas arriba y abajo: se pierde ~24 %
# de altura de visión. Pidiendo 12 MP (4056x3040, 4:3) y escalando en el ISP se
# conserva TODO el sensor → VFOV pasa de ~42° a ~55° (≈ +30 % de altura).
# Esto es lo que te permite levantar el brazo sin que la mano se salga del cuadro.
ANCHO_CAM      = 640    # píxeles de ancho del fotograma (color y profundidad)
ALTO_CAM       = 480    # píxeles de alto  (4:3 → sensor completo, sin recorte vertical)
ESCALA_VENTANA = 1.35   # Zoom de la ventana OpenCV (640x480 * 1.35 = 864x648)
F_PX           = 465.0  # Focal en px a 640 de ancho. SOLO es respaldo: al arrancar se lee
                        # la focal real de la calibración de la cámara.
                        # (El HFOV no cambia al pasar a 4:3, así que esta focal sigue valiendo.)

# Modo de sensor del color:
#   True  → 12 MP + setIspScale: FOV vertical COMPLETO (recomendado, brazo levantado)
#   False → 1080p clásico (16:9, recortado arriba/abajo). Solo si el ISP te da problemas.
CAM_FOV_COMPLETO = True
CAM_FPS          = 30     # fija los FPS del sensor; si tu bucle va lento, bájalo a 20

# ── DETECCIÓN DE CUERPO (MediaPipe Pose) ──────────────────────────────────────
MOSTRAR_ESQUELETO_CUERPO = True   # True = dibuja el esqueleto del cuerpo en la ventana
POSE_COMPLEJIDAD         = 1      # 0 = rápido | 1 = equilibrado | 2 = más preciso y lento
POSE_CONF_DETECCION      = 0.6
POSE_CONF_SEGUIMIENTO    = 0.45   # BAJADO: con el bucle a ~12 FPS un 0.6 hace que Pose
                                  # pierda el "lock" y reinicie la detección constantemente
POSE_MEDIR_FPS           = True   # imprime los FPS reales del bucle de teleoperación

# ── DETECCIÓN DE PELOTA POR COLOR HSV ─────────────────────────────────────────
COLORES_HSV = {
    "pelota_amarilla": ((22,  80, 100), (38, 255, 255)),
    "pelota_roja":     ((0,   95,  95), (10, 105, 105)),
    "verde":           ((36,  50,  70), (89, 255, 255)),
    "pelota_azul":     ((94,  80,  20), (126, 255, 255)),
}
COLOR_PELOTA       = "pelota_amarilla"
MIN_AREA_PELOTA    = 800
N_FRAMES_DEBOUNCE  = 10

# ── PUNTOS DE APARICIÓN (SPAWN) DINÁMICOS POR POSE ────────────────────────────
PUNTOS_SPAWN = {
    "arriba": {
        "pelota":         {"pos": ( 0.0, 0.08, 0.30), "euler": (0.0, 0.0, 0.0)},
        "destornillador": {"pos": ( 0.0, 0.08, 0.30), "euler": (1.5708, 0.0, -1.70)}, 
        "soporte":        {"pos": ( 0.0, 0.09, 0.17), "euler": (0.0, 0.0, -0.3)}
    },
    "izquierda": {
        "pelota":         {"pos": ( 0.0, -10, 0.20),  "euler": (0.0, 0.0, 0.0)},
        "destornillador": {"pos": ( 0.185, 0.035, 0.0), "euler": (0.0, 0.0, 0.0)},    
        "soporte":        {"pos": ( 0.0, 0.0, 0.10),  "euler": (0.0, 0.0, 0.0)}
    },
    "abajo": {
        "pelota":         {"pos": ( 0.0, 0.2, -0.05), "euler": (0.0, 0.0, 0.0)},
        "destornillador": {"pos": ( -0.005, 0.2, -0.05), "euler": (1.5708, 0.0, -1.70)}, 
        "soporte":        {"pos": ( 0.0, 0.0, 0.10),  "euler": (0.0, 0.0, 0.0)}
    },
    "derecha": {
        "pelota":         {"pos": ( 0.0, -0.20, 0.25), "euler": (0.0, 0.0, 0.0)},
        "destornillador": {"pos": ( 0.02, -0.20, 0.0), "euler": (1.5708, 0.0, -1.70)},    
        "soporte":        {"pos": ( 0.0, 10.0, -10.10), "euler": (0.0, 0.0, 0.0)}
    }
}

# ── BRAZO ARCTOS: JOINTS, LÍMITES Y OFFSET DE ACTUADORES ──────────────────────
# En arctos_mjcf.xml el <actuator> define PRIMERO los 6 del brazo y DESPUÉS
# los 17 de la mano. OFFSET_MANO es cuánto hay que desplazar cualquier índice
# "de mano" (0..16, los mismos de siempre) para caer en el actuador correcto
# de datos.ctrl[] una vez cargado el modelo fusionado.
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

LIMITES_CTRL_BRAZO = [
    (-3.14, 3.14),   # joint1 (base)
    (-2.5,  2.5),    # joint2 (hombro)
    (-2.5,  2.5),    # joint3 (codo)
    (-3.14, 3.14),   # joint4 (muñeca 1)
    (-3.14, 3.14),   # joint5 (muñeca 2)
    (-3.14, 3.14),   # joint6 (muñeca 3)
]

# Postura de reposo del brazo mientras no haya IK conectada (paso 1) o cuando
# no se esté siguiendo la cámara. AJUSTA ESTOS VALORES en el visor MuJoCo a
# una postura cómoda, sin autocolisión con la mano ni con la mesa.
POSICION_STANDBY_BRAZO = [0.0, -0.3, 0.6, 0.0, 0.0, 0.0]

OFFSET_MANO = len(ARM_JOINTS)  # = 6

# ── ACTUADORES Y LÍMITES ANATÓMICOS DE SEGURIDAD (MANO) ───────────────────────
N_ACT = 17

LIMITES_CTRL = [
    (-1.134,  0.610),  #  0: muñeca
    (-0.20,   0.50),   #  1: meñique abd
    ( 0.0,    1.80),   #  2: meñique mcp
    ( 0.0,    1.80),   #  3: meñique pip
    (-0.20,   0.50),   #  4: anular abd
    ( 0.0,    1.80),   #  5: anular mcp
    ( 0.0,    1.80),   #  6: anular pip
    ( 0.0,    0.0),    #  7: corazón abd (fijo)
    ( 0.0,    1.80),   #  8: corazón mcp
    ( 0.0,    1.80),   #  9: corazón pip
    (-0.50,   0.50),   # 10: índice abd
    ( 0.0,    1.80),   # 11: índice mcp
    ( 0.0,    1.80),   # 12: índice pip
    (-0.785,  1.85),   # 13: pulgar oposición/CMC
    (-0.35,   1.35),   # 14: pulgar abd
    (-0.15,   1.80),   # 15: pulgar mcp  ← permite extensión activa
    ( 0.0,    1.80),   # 16: pulgar ip
]

POSICION_STANDBY = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    -0.0979, 0.641,  0.00,  0.175
]

# ── PARÁMETROS BIOMECÁNICOS Y FILTROS ─────────────────────────────────────────
UMBRAL_BLOQUEO    = 0.50
UMBRAL_LIBERACION = 0.10

# Ganancias globales
FUERZA_NUDILLOS   = 1.35
FUERZA_PUNTAS     = 1.45
FUERZA_ABD        = 1.50

# Ganancias INDIVIDUALES por dedo (multiplican sobre la global)
# 1.0 = sin cambio respecto a la ganancia global
# Sube un dedo concreto sin tocar el resto
GANANCIA_DEDOS = {
    "menique_mcp":  1.00,   # ctrl[2]
    "menique_pip":  1.00,   # ctrl[3]
    "anular_mcp":   1.10,   # ctrl[5]
    "anular_pip":   1.10,   # ctrl[6]
    "corazon_mcp":  1.05,   # ctrl[8]
    "corazon_pip":  1.05,   # ctrl[9]
    "indice_mcp":   1.00,   # ctrl[11] — referencia
    "indice_pip":   1.00,   # ctrl[12]
}

# ── UMBRALES Y GANANCIAS DEL PULGAR ───────────────────────────────────────────
# CRÍTICO: estas 4 constantes son usadas directamente en retarget_a_ctrl()
UMBRAL_EXT_PULGAR_MCP = 0.15   # por debajo → zona de extensión activa
UMBRAL_EXT_PULGAR_IP  = 0.25   # por debajo → filtra temblor de punta
PULGAR_GANANCIA_MCP   = 1.70   # amplifica flexión del nudillo
PULGAR_GANANCIA_IP    = 1.90   # amplifica flexión de la punta

# ── UMBRALES DE PINZA PREDEFINIDA (CON HISTÉRESIS) ────────────────────────────
PINZA_UMBRAL_ON = {
    "indice": 0.030, "corazon": 0.040, "anular": 0.040, "menique": 0.035,
    "corazon_anular": 0.040, "indice_corazon": 0.040, "anular_menique": 0.040
}

PINZA_UMBRAL_OFF = {
    "indice": 0.040, "corazon": 0.050, "anular": 0.050, "menique": 0.045,
    "corazon_anular": 0.050, "indice_corazon": 0.050, "anular_menique": 0.050
}

_UMBRAL_COMBO_ON   = {"corazon": 0.035, "anular": 0.040}   
_UMBRAL_COMBO_OFF  = {"corazon": 0.045, "anular": 0.050}
_UMBRAL_COMBO2_ON  = {"indice": 0.040, "corazon": 0.045}  
_UMBRAL_COMBO2_OFF = {"indice": 0.050, "corazon": 0.055}
_UMBRAL_COMBO3_ON  = {"anular": 0.040, "menique": 0.040}  
_UMBRAL_COMBO3_OFF = {"anular": 0.050, "menique": 0.050}

PINZA_DIST_MAX = 0.120

# ── DETECCIÓN DE LA MANO CON RECORTE (arregla las pinzas con el cuerpo en cuadro) ─
# Con la cámara abierta a todo el cuerpo pasan dos cosas malas para MediaPipe Hands:
#   1) Se ven LAS DOS manos y, con max_num_hands=1, puede engancharse a la que NO es
#      (la que cuelga quieta) → los dedos del robot no siguen a los tuyos y la pinza
#      nunca se activa.
#   2) A ~2 m la mano ocupa ~40 px: las yemas se estiman con errores de varios cm y la
#      distancia pulgar–dedo casi nunca baja del umbral de pinza (3-4 cm).
# Solución: MediaPipe Pose dice DÓNDE está tu mano (la del BRAZO_HUMANO) y Hands se
# ejecuta solo sobre un recorte cuadrado alrededor de ella, ampliado a MANO_RECORTE_PX.
MANO_RECORTE        = True    # False = comportamiento anterior (Hands en toda la imagen)
MANO_RECORTE_HD     = True    # True = el recorte sale de la imagen del sensor a 1352x1014
                              # (2,1x más píxeles). Necesita CAM_FOV_COMPLETO = True y USB3;
                              # si la cámara va lenta o da errores de enlace, pon False.
MANO_RECORTE_PX     = 256     # lado (px) al que se amplía el recorte antes de pasarlo a Hands
MANO_RECORTE_FACTOR = 3.0     # tamaño del recorte = FACTOR × (muñeca → nudillos) en la imagen
MANO_RECORTE_MIN    = 0.12    # lado mínimo del recorte, como fracción del alto de la imagen

# Diagnóstico de pinzas: imprime la distancia del pulgar a cada yema (cm), el umbral,
# si la pinza está activa y de dónde sale la mano (recorte / imagen completa).
DEBUG_PINZA        = True
DEBUG_PINZA_CADA_S = 0.5

# ── LANDMARKS MEDIAPIPE INDICES ──────────────────────────────────────────────
LM_MUNECA      = 0
LM_INDICE_MCP  = 5
LM_CORAZON_MCP = 9
LM_ANULAR_MCP  = 13
LM_MENIQUE_MCP = 17

# ── LANDMARKS MEDIAPIPE POSE (cuerpo) ─────────────────────────────────────────
# "IZQ"/"DER" son el lado del PROPIO SUJETO (no de la imagen).
PS_HOMBRO_IZQ, PS_HOMBRO_DER = 11, 12
PS_CODO_IZQ,   PS_CODO_DER   = 13, 14
PS_MUNECA_IZQ, PS_MUNECA_DER = 15, 16
PS_CADERA_IZQ, PS_CADERA_DER = 23, 24

# ── ÁNGULOS DEL BRAZO HUMANO (paso 3: de momento solo se calculan y se muestran) ─
BRAZO_HUMANO         = "derecho"   # "derecho" | "izquierdo" — brazo que se va a imitar
POSE_VISIBILIDAD_MIN = 0.35        # por debajo, MediaPipe "adivina" el punto → se ignora
                                   # (0.5 era demasiado estricto: al girar el torso o
                                   #  levantar el brazo, el hombro lejano baja de 0.5 y
                                   #  el brazo entero se descartaba)
DEBUG_ANGULOS_BRAZO  = True        # imprime los ángulos en la terminal
DEBUG_ANGULOS_CADA_S = 0.5         # cada cuántos segundos imprimir

# De dónde sale la PROFUNDIDAD (eje que se aleja de la cámara) de los puntos del brazo:
#   "estereo"   → profundidad REAL medida por las dos cámaras de la OAK-D (recomendado)
#   "mediapipe" → profundidad ESTIMADA por la red neuronal a partir de una sola imagen
# En la terminal se imprimen SIEMPRE las dos, para poder compararlas.
BRAZO_FUENTE_3D = "estereo"
PROF_RADIO_PX   = 4      # radio (px) del parche donde se mide la profundidad de cada articulación
                         # (subido de 5: el antebrazo es fino y un parche 11x11 se queda
                         #  sin píxeles válidos en cuanto hay un hueco del estéreo)
PROF_PERCENTIL  = 15     # percentil bajo = "el punto más cercano" → evita coger el fondo tras el brazo
PROF_MIN_PIXELES = 4     # píxeles con profundidad válida mínimos en el parche (antes 6 fijo)
PROF_Z_MIN      = 0.25   # m — por debajo, el estéreo no puede medir (línea base de la OAK-D)
PROF_Z_MAX      = 5.0    # m
PROF_DISPERSION_Z_MAX = 1.4   # m — cuánto puede alejarse una articulación de la mediana
                              # del resto antes de descartarla. Con 0.8 m, al ESTIRAR el
                              # brazo hacia la cámara la muñeca quedaba fuera y el brazo
                              # se perdía justo cuando más lo necesitas.

# Estéreo con precisión sub-píxel: sin ello la profundidad va "a escalones" de varios cm.
STEREO_SUBPIXEL = True

# Post-procesado del mapa de profundidad EN LA PROPIA CÁMARA (no cuesta CPU).
# Rellena los agujeros del brazo, que son la causa nº1 de "se pierde a ratos".
STEREO_FILTROS      = True
STEREO_HOLE_FILLING = 2       # radio de relleno de huecos (0 = desactivado, 2-4 típico)
STEREO_TEMPORAL     = True   # filtro temporal: rellena más, pero deja ESTELA en
                              # movimientos rápidos. Actívalo solo si te mueves despacio.
STEREO_EXTENDED     = False   # disparidad extendida: baja la distancia mínima a ~15 cm.
                              # En algunas versiones de depthai choca con subpixel; si al
                              # activarlo peta, pon STEREO_SUBPIXEL = False.

# ── ROBUSTEZ CUANDO EL BRAZO APUNTA A LA CÁMARA ───────────────────────────────
# Con el brazo estirado hacia la cámara, el codo queda tapado por la mano/antebrazo
# y MediaPipe lo coloca en un sitio falso (p.ej. a la altura del ombligo).
# Se detecta porque las LONGITUDES de los huesos no cuadran, y entonces el codo
# se reconstruye con geometría a partir de hombro y muñeca.
#   · LONG_BRAZO_MANUAL = None            → las longitudes se aprenden solas (mediana)
#   · LONG_BRAZO_MANUAL = (0.30, 0.26)    → tus medidas reales en metros (brazo, antebrazo)
LONG_BRAZO_MANUAL = None
LONG_BRAZO_TOL    = 0.25   # un hueso puede desviarse ±25 % de su longitud habitual

# Los huesos se aprenden AL PULSAR [A]: durante ~1 s hay que mantener el brazo
# estirado en T-pose (así se ve entero y sin acortarse por la perspectiva).
CALIB_HUESOS_FRAMES = 30   # fotogramas válidos que se promedian

# Cuando el frame actual NO puede calcular ángulos del brazo (profundidad ruidosa
# en un solo punto, oclusión de un instante...), en vez de congelar el brazo en
# su última posición comandada se sigue usando el último ángulo VÁLIDO durante
# hasta este tiempo (segundos). Pasado ese tiempo, se asume que el brazo se ha
# perdido de verdad y se deja de actualizar. Sube este valor si ves "paradas"
# breves pero molestas; bájalo si el brazo tarda en reaccionar a una pérdida real.
ANG_BRAZO_HOLD_S = 0.9   # SUBIDO de 0.4: a ~12 FPS reales, 0.4 s son solo 5 fotogramas,
                         # así que cualquier bache del estéreo congelaba el brazo.

# Los hombros se miden en el PECHO (a esta fracción del camino hacia el centro del torso)
# y no en el propio hombro: si el brazo apunta a la cámara, la mano tapa el hombro en el
# mapa de profundidad y se leería la profundidad de la mano, deformando el marco del torso.
HOMBRO_PROF_HACIA_TORSO = 0.6

# ── QUE EL BRAZO NO SE "PIERDA" POR UN HUECO DEL ESTÉREO ──────────────────────
# Antes, si UNA sola articulación (casi siempre la muñeca o el codo: son finos y se
# mueven) no tenía profundidad estéreo válida en ese fotograma, se descartaba el
# brazo entero → "sin datos" aunque la cámara lo viera perfectamente.
# Ahora cada articulación prueba, por orden:
#   E  → profundidad estéreo en su parche normal
#   E+ → estéreo en un parche el doble de grande (sale de un hueco pequeño)
#   M  → profundidad RELATIVA de MediaPipe anclada a un punto que sí tiene estéreo
#        (hombro/cadera): z = z_ancla_estéreo + (z_mp_punto − z_mp_ancla)
# Así solo se cae a "sin datos" si no hay estéreo ni en hombros ni en caderas.
PROF_RELLENO_MEDIAPIPE = True

# ── DATASET E IA (ARCHIVOS) ───────────────────────────────────────────────────
ARCHIVO_DATASET = "movimientos_mano.csv"
ARCHIVO_PESOS   = "cerebro_orca.pth"

# ── MAPEO MUÑECA HUMANA → OBJETIVO DEL EFECTOR DEL BRAZO ARCTOS ──────────────
# Punto (en metros, en el frame BASE del ARCTOS) donde debe estar el efector
# final (Link_6_1) cuando la muñeca humana está en su posición NEUTRA
# (la que tenga en el instante en que actives la cámara con [A]).
# Ajusta esto a un punto cómodo y alcanzable del brazo (ni muy cerca de la
# base ni estirado al límite).
ARM_OBJETIVO_NEUTRO = (0.30, 0.0, 0.30)   # (X adelante, Y lateral, Z altura)

# Cuántos metros se mueve el efector del brazo por cada metro que se mueve
# la muñeca humana respecto a la cámara. 1.0 = movimiento 1:1.
ARM_ESCALA_MOVIMIENTO = (1.0, 1.0, 1.0)   # (X, Y, Z) del robot

# La cámara entrega (x_m, y_m, z_m) con: x = lateral (+derecha),
# y = vertical (+hacia ABAJO, porque el eje Y de imagen crece hacia abajo),
# z = profundidad (+alejándose de la cámara).
# Aquí decides qué eje de CÁMARA alimenta cada eje del ROBOT, y con qué signo.
# Son valores de partida — muy probablemente tocará invertir algún signo tras
# la primera prueba (ver más abajo cómo hacerlo).
#
# AJUSTE (tras la primera prueba con el brazo): el mapeo "intuitivo" de
# arriba (adelante←profundidad, lateral←x) resultó estar CRUZADO en la
# práctica — mover la muñeca hacia la cámara movía el brazo a la derecha, y
# moverla a la izquierda lo movía hacia delante. Así que aquí abajo la
# profundidad ("z") alimenta ahora el lateral del robot, y el lateral de
# cámara ("x") alimenta el "adelante" del robot.
#
# Los signos son una PRIMERA HIPÓTESIS para que "izquierda → delante" y
# "hacia la cámara → derecha" salgan en el sentido correcto — pruébalo y, si
# alguno sale invertido (p.ej. "hacia delante" te lleva hacia atrás), cambia
# solo el signo de esa fila (+1.0 ↔ -1.0), no el eje de cámara.
ARM_MAPEO_EJES = {
    "robot_x": ("x", 1.0),   # "adelante" del robot ← lateral de la cámara
    "robot_y": ("z", 1.0),   # lateral del robot ← profundidad de la cámara
    "robot_z": ("y", -1.0),   # altura del robot ← vertical de cámara (invertido)
}

# Seguridad: nunca apuntar el efector a más de este radio desde la base,
# ni por debajo de esta altura (evita objetivos de IK imposibles o que
# choquen contra la mesa).
ARM_RADIO_MAX  = 0.55   # metros
ARM_Z_MINIMA   = 0.05   # metros

# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — IMITACIÓN DEL BRAZO HUMANO POR ÁNGULOS ARTICULARES (ARCTOS)
# ══════════════════════════════════════════════════════════════════════════════
# "imitacion" → el brazo copia los ÁNGULOS del brazo humano (joint1..3), sin IK.
# "ik"        → el modo anterior: la punta del brazo persigue tu muñeca (IK).
MODO_BRAZO = "imitacion"

# Cómo se convierte cada ángulo humano (en RADIANES) en el ángulo de un joint:
#
#       q_joint = offset + signo * escala * angulo_humano
#
# Sale de la geometría del arctos_mjcf.xml (con todos los q = 0 el brazo apunta
# HACIA ARRIBA y el antebrazo horizontal hacia delante; "delante" del robot = -Y):
#   · joint1 (giro de base, eje +Z): q>0 gira el brazo hacia la IZQUIERDA del robot.
#       Tu azimut es + hacia FUERA (tu derecha) → signo -1  (tu derecha = su derecha).
#   · joint2 (hombro, eje +X): q=0 brazo vertical hacia arriba; q crece hacia delante y abajo.
#       Tu elevación es 0 con el brazo colgando y 180° con él arriba → q2 = π - elevación.
#   · joint3 (codo, eje -X): con q≈1.52 el antebrazo sigue la línea del brazo (codo
#       estirado); q baja al flexionar → q3 = 1.52 - flexión.
#
# Si al probar un joint se mueve al revés, cambia SOLO su signo (+1 ↔ -1).
# Si un joint sale "descentrado" (p.ej. el codo estirado no queda recto), retoca su offset.
ARM_IMITACION = {
    #  joint      (ángulo humano, signo, offset [rad],  escala)
    "joint1": ("azimut",    -1.0, 0.0,      1.0),
    "joint2": ("elevacion", -1.0, math.pi,  1.0),
    "joint3": ("codo",      -1.0, 1.52,     1.0),
}

# Límites de SEGURIDAD para la imitación (rad). Más estrechos que los del actuador:
#   · joint2 máx 2.3 evita que el brazo baje del todo y golpee la mesa
#   · joint3 evita plegar el codo contra el propio brazo
IMITAR_LIMITES = {
    "joint1": (-3.14, 3.14),
    "joint2": (-2.5, 2.5),
    "joint3": (-2.5, 2.5),
}

IMITAR_VEL_MAX   = 1.5   # (antes 2.5; con física en tiempo real era demasiado brusco) rad/s: velocidad máxima de cada joint (protege de saltos)

# ── T-POSE COMO REFERENCIA DE POSTURA (arregla "el brazo se queda abajo") ─────
# La calibración de huesos solo aprendía LONGITUDES. Ahora, en esos mismos ~30
# fotogramas de T-pose, también se mide el az/elev/codo que ve la cámara y se
# compensa la diferencia con la T-pose ideal, de modo que
#       tu T-pose  ==  T-pose del brazo simulado
# aunque la cámara esté inclinada, el hombro se mida algo alto, o dejes caer
# un poco el codo al calibrar.
IMITAR_CALIBRAR_TPOSE = True
T_POSE_HUMANA = (90.0, 90.0, 0.0)   # (azimut, elevación, codo) en GRADOS de una T-pose
                                     # perfecta: brazo horizontal, hacia fuera, estirado.
                                     # Solo se usa si la T-pose medida no es válida y en
                                     # el modo "absoluto".

# CÓMO SE RELACIONA TU T-POSE CON EL BRAZO SIMULADO:
#   "relativo" → la pose en la que está el brazo simulado al pulsar [A] (tu
#                POSICION_STANDBY_BRAZO, la "T-pose" de reposo del sim) SE CORRESPONDE
#                con tu T-pose medida. El brazo NO se mueve al calibrar; luego copia
#                tus movimientos como incrementos respecto a esa pose:
#                    q = q_reposo + signo·escala·(ángulo_humano − ángulo_T_medido)
#   "absoluto" → el brazo copia tus ángulos directamente (q = offset + signo·escala·ángulo),
#                así que tu T-pose real equivale a la T-pose GEOMÉTRICA del robot
#                (brazo horizontal a su derecha) y el sim salta a ella.
IMITAR_REFERENCIA = "absoluto"

# Solo modo "absoluto": al pulsar [A] llevar el brazo simulado a su T-pose geométrica.
IMITAR_TPOSE_SIM_AL_CALIBRAR = True

AZIMUT_ELEV_MIN = 35.0

# ── GRAVEDAD DE LA MANO ──────────────────────────────────────────────────────
# En arctos_mjcf.xml los links del brazo llevan gravcomp="1", pero los cuerpos de
# la OrcaHand (dentro del <include>) NO lo heredan: gravcomp es un atributo de
# cada <body>, no de la clase. Resultado: la mano pesa sobre joint2/joint3 y el
# brazo "cuelga" por debajo de lo ordenado (kp=500/300 no basta para masas de kg).
# True = se compensa la gravedad de TODOS los cuerpos que cuelgan de mano_mount.
HAND_GRAVCOMP = True
IMITAR_SUAVIZADO = 3     # fotogramas de media móvil sobre los ángulos objetivo

# ── PASO 5 — MUÑECA DEL ARCTOS: SOLO PRONOSUPINACIÓN ─────────────────────────
PRONO_JOINT    = "joint6"   # "joint6" (solo mueve la mano) | "joint4"
PRONO_SIGNO    = -1.0        # si gira al revés → -1.0
PRONO_ESCALA   = 1.2
PRONO_LIMITE   = 1.5        # rad (~86°) a cada lado del neutro
PRONO_VEL_MAX  = 2.0        # (antes 3.0) rad/s
PRONO_SUAVIZADO = 0.35      # 0..1 (filtro exponencial; más bajo = más suave)
JOINT5_FIJO    = 0.0        # muñeca del ARCTOS recta: la flexión la hace la OrcaHand

# Vertical para medir el brazo:
#   "caderas" → línea cadera→hombro (bien de pie; SENTADO la mesa tapa las caderas)
#   "camara"  → vertical de la cámara (la OAK-D está fija; la T-pose corrige su inclinación)
ARRIBA_REF = "camara"

# Si la muñeca se mide más lejos que brazo+antebrazo por encima de este margen,
# es una mala medida (mano delante del cuerpo): NO se fuerza el codo a recto.
CODO_ESTIRADO_MARGEN = 0.05     # 5 %

# ── PASO 2 — HÍBRIDO: LA MANO DEL ROBOT VA DONDE VA TU MANO (distancias OAK-D) ──
# 0.0 = solo ángulos (como hasta ahora) · 1.0 = la mano del robot sigue a tu mano
# Intermedio (p.ej. 0.6) = mezcla. Se alterna en vivo con [H].
HIBRIDO_PESO        = 1.0
HIBRIDO_ESCALA      = 1.0    # >1: el robot llega proporcionalmente más lejos que tú
HIBRIDO_ALCANCE_MAX = 0.95   # nunca estirar el ARCTOS del todo (singularidad)
HIBRIDO_SUAVIZADO   = 4      # media móvil (fotogramas) sobre el punto objetivo
HIBRIDO_IK_ITER     = 15     # iteraciones de IK por fotograma
HIBRIDO_IK_PASO     = 0.12   # rad máx por iteración

# ── FÍSICA EN TIEMPO REAL (mejora 1, idea del proyecto TRON 2) ───────────────
# True: la simulación avanza en un hilo propio al ritmo del reloj real y el visor
#       se refresca a FISICA_HZ, independientemente de los FPS de la cámara.
# False: comportamiento anterior (4 mj_step por fotograma → ~0.08x a 8 FPS).
FISICA_EN_HILO     = True

# PERFIL DEL ORDENADOR — cambia SOLO esta línea al pasar a producción.
#   "puesto"     → tu ordenador de trabajo (más modesto): menos carga para el visor
#   "produccion" → ordenador bueno: valores óptimos
PERFIL_EQUIPO = "puesto"

if PERFIL_EQUIPO == "produccion":
    FISICA_HZ          = 60     # refrescos por segundo del visor / avances de la física
    FISICA_RETRASO_MAX = 0.2    # s: retraso máximo que se recupera de golpe
else:  # "puesto"
    FISICA_HZ          = 30     # la mitad de refrescos del visor: libera CPU/GPU
    FISICA_RETRASO_MAX = 0.1    # si se atrasa, se descarta antes en vez de "acelerar"

# ── VALIDACIÓN DE PROFUNDIDAD CON MEDIAPIPE (mejora 2, idea del TRON 2) ──────
# Si el estéreo del codo o de la muñeca se aleja más de esto (m) de la profundidad
# que predice MediaPipe (anclada al hombro), se considera una mala lectura (fondo,
# torso...) y no se usa. 0 = desactivado (comportamiento anterior).
#   · más bajo (0.15) → más estricto: descarta más estéreo, confía más en MediaPipe
#   · más alto (0.30) → más permisivo
PROF_TOL_MEDIAPIPE = 0.12

# ── PASO 3 — DIRECCIONES + IK CON CODO Y MUÑECA (ideas del TRON 2) ───────────
# "codo_muneca": tus DIRECCIONES de brazo y antebrazo → dónde deben estar el codo
#                y la muñeca del ARCTOS (con SUS longitudes) → IK que persigue ambos.
# "muneca":      el híbrido anterior (solo la muñeca, escalada por el alcance).
# [H] sigue alternando entre el híbrido elegido aquí y "solo ángulos".
HIBRIDO_MODO       = "codo_muneca"
HIBRIDO_PESO_CODO  = 0.4            # (antes 0.6) importancia del codo frente a la muñeca (0 = solo muñeca)
HIBRIDO_CODO_RECTO = (10.0, 30.0)   # grados de flexión: <10 el codo no cuenta, >30 cuenta entero
ZONA_MUERTA_CODO   = (5.0, 25.0)    # grados: <5 se trata como brazo recto, hasta 25 se atenúa

# ── PASO 5 — FILTRO ONE EURO (idea del TRON 2) ──────────────────────────────
# En el modo "codo_muneca" filtra tus DIRECCIONES de brazo y antebrazo y sustituye
# a las medias móviles (HIBRIDO_SUAVIZADO e IMITAR_SUAVIZADO), que retrasaban siempre.
FILTRO_ONE_EURO   = True
FILTRO_MIN_CUTOFF = 0.4     # (antes 0.8) Hz: más bajo = menos temblor en reposo (pero más retraso)
FILTRO_BETA       = 0.3     # (antes 0.7) más alto = menos retraso en movimientos rápidos (pero más temblor)

# ── PASO 6 — RESCATE DE LA IK (idea del TRON 2) ─────────────────────────────
# Si la IK lleva IK_RESCATE_TIEMPO segundos con error > IK_RESCATE_ERROR, se prueba
# desde otras posturas de partida y se cambia a la mejor si mejora de verdad.
IK_RESCATE_ERROR  = 0.08    # m (error ponderado codo+muñeca)
IK_RESCATE_TIEMPO = 0.5     # s
IK_RESCATE_MEJORA = 0.7     # la alternativa debe dejar el error por debajo del 70 %
IK_RESCATE_ITER   = 30      # iteraciones por postura de partida
# Probar también la configuración "girada" (base +180°, hombro hacia atrás), que dobla
# el codo hacia el OTRO lado. Solo funciona si IMITAR_LIMITES["joint2"] empieza por
# debajo de -1.0 (comprueba antes en el visor que el hombro puede ir hacia atrás sin chocar).
IK_RESCATE_GIRADA = False   # (antes True) PASO 7: la solución "girada" gira la base
                            # ~180° de golpe → con herramienta es un salto enorme.
                            # Vuelve a True solo si ves que la IK se atasca a menudo.

# Correcciones del rescate (28-sep) — evitan las "vueltas" de la base:
IK_RESCATE_ESPERA  = 1.5    # s mínimos entre dos cambios de solución (sin ping-pong)
IK_PENALIZA_BASE   = 0.04   # m de "coste" por cada rad que haya que girar la base
IK_PENALIZA_GIRADA = 0.06   # m de "coste" extra por usar la configuración girada

# ── CORRECCIONES 28-sep ─────────────────────────────────────────────────────
# Fusión de direcciones: el brazo y el antebrazo usan la dirección del ESTÉREO
# solo si se desvía menos de esto (grados) de la de MediaPipe; si no, MediaPipe.
# Evita codos "doblados" falsos. 0 = desactivado (solo estéreo).
FUSION_DIR_MAX_DEG = 30.0

# Giro de muñeca:
#   "mundo"    → la mano del robot se orienta IGUAL que la tuya en el espacio
#                (funciona con cualquier postura que elija la IK). Recomendado.
#   "relativo" → como antes: giro relativo al neutro, con PRONO_ESCALA.
PRONO_MODO     = "mundo"
PRONO_Z_ESCALA = 1.6        # MediaPipe Hands "aplasta" la profundidad de la mano

# ── OBJETIVO DE LA IK: PALMA DE LA ORCAHAND (28-sep) ─────────────────────────
# "palma"  → la palma de la OrcaHand va donde va TU palma, teniendo en cuenta las
#            proporciones (brazo + antebrazo + mano) tuyas y del ARCTOS + OrcaHand.
# "muneca" → como antes: se persigue el muñón (mano_mount) y se ignora la mano.
HIBRIDO_OBJETIVO    = "palma"
HUMANO_MUNECA_PALMA = 0.08   # m: de tu muñeca al centro de tu palma (aprox. adulto)

# ── ESCENARIO TALADRO — TECLA [4] ───────────────────────────────────────────
# Antes, una vez: python taladro/preparar_taladro.py  (+ include en arctos_mjcf.xml)
TALADRO_POS_SOPORTE   = (-0.45, -0.20, 0.25)  # m: centro de la cara SUPERIOR de la peana
TALADRO_TECHO_XY      = (0.0, 0.0)            # m: centro del techo (encima del robot)
TALADRO_TECHO_Z       = None    # m: altura del techo; None = automática (ver holgura)
TALADRO_TECHO_HOLGURA = 0.16    # m por encima de la palma con el brazo vertical
TALADRO_ASISTENCIA_AGARRE = True   # el taladro "no pesa" mientras la mano lo agarra
TALADRO_CONTACTOS_AGARRE  = 3      # contactos mano↔taladro para considerarlo agarrado

# ── DEMO AUTOMÁTICA DEL TALADRO — TECLA [5] ─────────────────────────────────
# El brazo coge el taladro de la peana (TALADRO_POS_SOPORTE) con la OrcaHand
# DERECHA por el lado DERECHO de la empuñadura, lo lleva al techo, aprieta el
# gatillo más fuerte con el índice y hace un agujero. Necesita el bloque
# <equality> de arctos_mjcf.xml (welds taladro_en_mano / taladro_en_peana).
TALADRO_DEMO_AGUJERO_XY    = (-0.15, -0.10)  # m: dónde se hace el agujero (dentro del techo)
TALADRO_DEMO_SEP_AGUJEROS  = 0.05     # m: cada vez que repites la demo, el agujero se desplaza
TALADRO_DEMO_TECHO_Z       = None     # m; None = automática: lo más alto que llega la BROCA
                                      # (brazo + OrcaHand + taladro) menos el margen
TALADRO_DEMO_MARGEN_TECHO  = 0.05     # m de margen hasta el alcance máximo (brazo sin estirar del todo)
TALADRO_DEMO_PROFUNDIDAD   = 0.04    # m que entra la broca en el techo
TALADRO_DEMO_BAJO_TECHO    = 0.08     # m bajo el techo desde donde sube la broca en vertical
TALADRO_DEMO_APROX         = 0.12     # m: la mano llega por el lado derecho del mango desde aquí
TALADRO_DEMO_ALTURA_PASO   = 0.12     # m que se levanta el taladro al cogerlo / dejarlo
TALADRO_DEMO_ALTURA_SEGURA = 0.20     # m sobre el mango por los que la mano se mueve al acercarse

# Dónde queda el eje del mango respecto a los nudillos de la OrcaHand (m):
#   (hacia la punta de los dedos, hacia fuera de la palma, hacia el índice)
# Al segundo valor se le suma el radio del mango. Si el mango queda metido en la
# palma sube el 2º; si los dedos no lo rodean, baja el 1º; si el índice choca con
# la cabeza del taladro, baja el 3º (negativo).
TALADRO_DEMO_AGARRE_OFFSET = (-0.017, -0.01, 0.00)

# Posturas de la mano (índices de ctrl 0..16, los de LIMITES_CTRL). Lo no indicado = 0.
TALADRO_DEMO_MANO_ABIERTA = {0: 0.0, 13: -0.0979, 14: 0.641, 15: 0.0, 16: 0.175}
TALADRO_DEMO_MANO_AGARRE  = {0: 0.0,
                             2: 1.25, 3: 1.10,     # meñique
                             5: 1.25, 6: 1.10,     # anular
                             8: 1.25, 9: 1.10,     # corazón
                             11: 1.00, 12: 0.90,   # índice (sobre el gatillo, sin apretar)
                             13: 0.35, 14: 0.45, 15: 0.70, 16: 0.90}   # pulgar rodeando
# "Apretar más fuerte el botón": solo cambia el índice. El motor arranca al pasar
# el 70 % del camino entre el agarre y este valor.
TALADRO_DEMO_MANO_GATILLO = {11: 1.60, 12: 1.50}

TALADRO_DEMO_RPM        = 240     # giro de la broca (visual; más alto parece girar al revés en pantalla)
TALADRO_DEMO_VIBRACION  = 0.003   # rad de vibración de la muñeca mientras perfora (0 = sin)
TALADRO_DEMO_VELOCIDAD  = 1.0     # >1 demo más rápida, <1 más lenta
TALADRO_DEMO_TIEMPOS = {          # s de cada fase (a TALADRO_DEMO_VELOCIDAD = 1)
    "subir": 1.5, "acercar": 3.0, "bajar": 1.5, "entrar": 2.0, "cerrar": 1.5,
    "levantar": 1.5, "al_techo": 3.5, "acercar_broca": 2.0,
    "gatillo": 0.7, "arrancar": 0.8, "perforar": 4.0, "sacar": 2.0, "soltar_gatillo": 0.6,
    "volver": 3.5, "dejar": 1.5, "abrir": 1.0, "retirar": 1.5, "reposo": 3.0,
}
TALADRO_DEMO_VEL_MAX    = 1.0     # rad/s máx. de cualquier joint del brazo (alarga la fase si hace falta)
TALADRO_DEMO_TOL_AGARRE = 0.035   # m: si al cerrar la mano el mango está más lejos, NO se agarra
                                  # (la mano se retira y se imprime cuánto se ha desviado)
TALADRO_DEMO_CAMARA = {"azimut": 25, "elevacion": -12, "distancia": 2.0}   # vista durante [5]
TALADRO_DEMO_BRAZO_CINEMATICO = True   # True: el brazo sigue el plan EXACTO (animación); los dedos
                                       # y el taladro siguen siendo físicos. False: servos + integral
TALADRO_DEMO_CONTACTOS_MIN = 2      # contactos dedos↔taladro esperados al cerrar la mano (aviso)
TALADRO_DEMO_SOLDADURA_RIGIDEZ = 0.004   # s: solref de las soldaduras taladro↔mano/peana en la demo
                                       # (más bajo = más rígida; mínimo 2×timestep = 0.004)
TALADRO_DEMO_KI = 2.0             # 1/s: corrección integral si el brazo se queda corto (0 = sin)
TALADRO_DEMO_SIN_AUTOCOLISION = True   # en la demo, los eslabones del brazo no chocan con la
                                       # mano ni con el taladro (evita atascos por roces de mallas)

# ══════════════════════════════════════════════════════════════════════════════
# PASO 7 — BRAZO PRECISO Y SIN SALTOS (para maniobrar con herramientas)
# ══════════════════════════════════════════════════════════════════════════════
# 7.1 COMANDO CONTINUO. La cámara da un objetivo nuevo solo 10-30 veces por segundo
#     y antes se escribía tal cual en ctrl → tirón en cada fotograma y parada hasta el
#     siguiente ("escalera"). Ahora el hilo de física, ANTES DE CADA mj_step (500/s),
#     lleva ctrl hacia el objetivo con velocidad y aceleración limitadas (igual que la
#     demo [5] del taladro). False = comportamiento anterior (para comparar).
SUAVE_ACTIVO     = True
SUAVE_INTERPOLAR = True   # reparte cada objetivo entre dos fotogramas: movimiento continuo,
                          # a cambio de ~1 fotograma de retraso. False = solo los límites.
#                  joint1  joint2  joint3  giro muñeca (PRONO_JOINT)
SUAVE_VEL_MAX  = (1.2,    1.2,    1.5,    2.0)    # rad/s
SUAVE_ACEL_MAX = (12.0,   12.0,   15.0,   20.0)   # rad/s² — más bajo = arranques y frenadas más suaves

# 7.2 SIN SALTOS EN LA MEDIDA
#   · Filtro de saltos: si el objetivo de la palma/codo del robot salta más de
#     SALTO_VEL_MAX·Δt + SALTO_MARGEN, ese fotograma es un error de medida y se ignora.
#     Si el salto se mantiene, el umbral crece con el tiempo y se acepta. 0 = desactivado.
SALTO_VEL_MAX = 1.0       # m/s (espacio del robot; tu mano ≈ ×0.75)
SALTO_MARGEN  = 0.03      # m
#   · Zona muerta: la palma del robot no se mueve por temblores menores que esto.
ZONA_MUERTA_PALMA = 0.005 # m (0 = desactivada)
#   · Fusión estéreo/MediaPipe sin interruptor: en FUSION_DIR_MAX_DEG ± esto (grados)
#     las dos direcciones se mezclan poco a poco en vez de saltar de una a otra.
FUSION_DIR_TRANSICION = 10.0
#   · IK con memoria: girar mucho un joint para ganar poca precisión "cuesta" (m por rad).
#     Evita los giros bruscos de la base cuando la mano pasa cerca de la vertical.
IK_PESO_CONTINUIDAD = 0.03

# 7.3 MODO FINO [F] y CONGELAR [C] — para maniobrar con herramientas
#   [F] el robot se mueve FINO_ESCALA veces lo que te mueves tú, desde donde esté
#       (1 cm tuyo = 3.5 mm del robot): más precisión y el temblor igual de reducido.
#   [C] el brazo se queda quieto (p.ej. con el taladro apoyado en el techo). Al quitarlo,
#       en modo fino sigue desde ahí; en modo normal vuelve a imitarte en FINO_SALIDA_S.
FINO_ESCALA      = 0.35
FINO_ESCALA_GIRO = 0.5    # lo mismo para el giro de muñeca
FINO_FACTOR_VEL  = 0.5    # en modo fino el comando continuo va a la mitad de vel./acel.
FINO_SALIDA_S    = 1.5    # s MÍNIMOS de transición suave al volver a la imitación normal
FINO_SALIDA_VEL  = 0.15   # m/s máx. de esa transición: si el robot quedó lejos, tarda más
IK_RESCATE_EN_FINO = False  # el rescate de la IK (cambiar de postura) nunca en fino/congelado

# 7.4 PROPORCIONES DEL ROBOT (el ARCTOS no tiene tus medidas)
#   · PALMA_CENTRO_REAL: la IK persigue el centro REAL de la palma de la OrcaHand
#     (antes un punto ~3 cm más fuera, casi en los nudillos).
#   · CODO_COMPATIBLE: el ARCTOS mide 0.28 m de hombro a codo y ~0.53 m de codo a palma
#     (tú ~0.30 y ~0.35). El objetivo del codo se coloca donde PUEDE estar con esas
#     medidas (en el lado de tu codo), para que no tire de la palma y esta no se pase.
#   False en cualquiera de los dos = comportamiento anterior.
PALMA_CENTRO_REAL = True
CODO_COMPATIBLE   = True
# ══════════════════════════════════════════════════════════════════════════════
# FLUIDEZ (30-sep) — MANO FÍSICA SIN TROMPICONES Y CÁMARA MÁS FLUIDA
# ══════════════════════════════════════════════════════════════════════════════
# ── Mano física (orca_hardware.py) ──
# La cámara da un objetivo nuevo solo ~10-15 veces por segundo. Antes se mandaba tal
# cual a los servos DESDE el bucle de la cámara → tirón y parada en cada fotograma
# (la "escalera" del PASO 7.1, pero en la mano real). Ahora un hilo propio manda un
# comando CONTINUO a ORCA_HW_HZ: interpola entre fotogramas y limita velocidad y
# aceleración de cada articulación (mismo algoritmo que GeneradorSuave del brazo).
ORCA_HW_HZ            = 50     # envíos/s a los servos (17 motores a 1 Mbps ≈ 1-2 ms por envío)
ORCA_HW_INTERPOLAR    = True   # reparte cada objetivo entre dos fotogramas (≈1 fotograma de retraso)
ORCA_HW_VEL_MAX       = 6.0    # rad/s máx. por articulación (≈340°/s). Más bajo = más suave pero más retraso
ORCA_HW_ACEL_MAX      = 60.0   # rad/s² máx. — más bajo = arranques y frenadas más suaves
ORCA_HW_LIMITES_JOINT = {      # excepciones por articulación: "joint": (vel. máx, acel. máx)
    "wrist": (3.0, 30.0),      # la muñeca mueve toda la mano: más despacio
}
ORCA_HW_DEBUG_TIEMPOS = True   # imprime los envíos/s reales y lo que tarda el SDK en cada envío
ORCA_HW_DEBUG_CADA_S  = 5.0

# ── Cámara ──
# Hands en PARALELO: el recorte de la mano se calcula con la Pose del fotograma
# ANTERIOR (la del actual aún no existe) y MediaPipe Hands corre en otro hilo mientras
# el principal hace Pose y el brazo. Así el coste de Hands casi desaparece del bucle.
# El margen del recorte (MANO_RECORTE_FACTOR) cubre lo que se mueve la mano entre dos
# fotogramas; si en ese recorte no hay mano, se sigue como antes (recorte con la Pose
# actual → imagen completa). False = todo en serie, como antes.
MANO_EN_PARALELO = True

# Perfil del bucle: cada PERFIL_BUCLE_CADA_S segundos imprime cuántos ms se van en cada
# fase (cámara, pose, brazo, mano...). Sirve para ver QUÉ frena la cámara en cada
# ordenador. Coste despreciable; pon False cuando ya no lo necesites.
PERFIL_BUCLE        = True
PERFIL_BUCLE_CADA_S = 3.0