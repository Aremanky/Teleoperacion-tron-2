"""
Fusion de varias OAK-D para la teleoperacion del TRON 2.

Cada camara sigue con su propio HiloSeguimiento (camara + MediaPipe de cuerpo y
manos). Este modulo se sienta DETRAS de los hilos, junta lo que ven todas y
devuelve UN resultado con el mismo formato que hilo.ultimo(), asi teleop_tron2.py,
objetivos(), marco_mano(), retargeting... no se enteran de cuantas camaras hay.

MARCOS
  W        marco "mundo": orientado como el OPERADOR (x delante, y izquierda,
           z arriba) con el origen donde estaba la primera camara situada.
  R_op[i]  rotacion camara i -> W.      t[i]  posicion de la camara i en W (m).
           Un punto p de la camara i esta en W en  R_op[i] @ p + t[i].
  R_F      marco de SALIDA: el de la camara principal al calibrar con [c] (o el de
           por defecto). El teleop lo usa como R_cam. R[i] = R_F^T @ R_op[i].
           Con la camara A en su sitio, R[0] = identidad.

COMO SE COMBINAN LAS CAMARAS
  Por PUNTOS 3D (cuando todas las camaras en uso estan situadas, t conocida):
    para cada hombro, codo y muneca, cada camara que lo ve aporta una gaussiana
    3D: estrecha de lado (el pixel de MediaPipe es preciso: rayo desde la camara)
    y alargada a lo largo del rayo (la profundidad de la OAK-D lo es menos; la
    prevista con MediaPipe, menos aun). Con dos camaras, el cruce de las dos
    gaussianas es una TRIANGULACION; con una, su medida con profundidad. Asi se
    rescata cada punto por separado (el codo de B con el hombro y la muneca de A)
    y las direcciones salen de los puntos fusionados. Antes se adelanta cada
    camara al instante de la mas reciente (no estan sincronizadas). Si dos camaras
    no cuadran en un punto (izquierda/derecha cambiadas, deteccion mala) se queda
    la que sigue al punto anterior.
  Por DIRECCIONES (mientras alguna camara aun no esta situada, o si no hay puntos):
    media ponderada de las direcciones de los huesos que da cada camara (la de la
    fase 2). El paso de un modo al otro es gradual.

CALIBRACION
  [c]: de pie, quieto, brazos colgando. Orienta W con el operador. Si las camaras
      ya estaban calibradas entre si, se CONSERVA esa relacion (es mas precisa que
      la postura) y solo se gira el conjunto; si te ven varias, sus estimaciones
      se promedian y su discrepancia se muestra como comprobacion. Las camaras que
      aun no estaban calibradas toman la de la postura (si te ven) o se calibran
      solas despues.
  Automatica y continua (con el operador moviendose, tambien en pausa):
    rotacion: parejas de direcciones de un mismo hueso visto casi a la vez por una
      camara calibrada y otra (problema de Wahba, robusto). Con IMU la gravedad fija
      la inclinacion y solo queda el giro alrededor de la vertical.
    posicion: el mismo punto del cuerpo (hombros, caderas, codos, munecas) medido
      con profundidad por las dos camaras. Se corrige que cada camara ve la PIEL
      de su lado (el centro de la articulacion esta unos cm mas alla).
  Camaras que se mueven: la IMU lo detecta (giro o desplazamiento) y esa camara se
    descarta y se recalibra sola contra las demas; si la que se mueve es la
    principal, otra sostiene el marco y el robot no se entera. Sin IMU se detecta
    porque sus medidas dejan de cuadrar (suponiendo que se ha movido la secundaria).

MANOS (oct-2026): si dos camaras SITUADAS ven la misma mano casi a la vez, se TRIANGULAN
  sus 21 landmarks (manos3d.py): orientacion y puntos 3D metricos reales, sin la
  profundidad aplastada del modelo. Si no, la camara que mejor la ve (tamano x calidad,
  con histeresis): "2.5D" (pixeles + profundidad OAK-D de la palma) o el modelo. La
  apertura siempre sale de la mejor camara. Todo en el marco de salida (R_F).
  El codo que nadie ve se RECONSTRUYE (brazo_geometria.py) en vez de dar el brazo por recto.
  Cada punto 3D entra en el Kalman con su 'calidad' (seguimiento_brazos): un codo con una
  longitud de hueso imposible pesa poco.
"""
import json
import os
import time
from collections import deque
from types import SimpleNamespace

import numpy as np

import brazo_geometria
import manos3d

# Ejes del operador en el marco de la camara, con la camara horizontal y el
# operador de frente a ella (la tecla 'c' los recalcula).
R_CAM_DEFECTO = np.array([[0.0, 0.0, -1.0],   # delante    = -z camara (hacia la camara)
                          [1.0, 0.0, 0.0],    # izquierda  = +x camara
                          [0.0, -1.0, 0.0]])  # arriba     = -y camara

# ------------------------------------------------------------------ pesos (direcciones)
PESO_FUENTE = {"profundidad": 1.0, "mixta": 0.7, "mediapipe3d": 0.4}
FACTOR_CODO_TAPADO = 0.3      # con el codo tapado la direccion de antebrazo es inventada
W_REF_CODO = 0.3              # peso de las vistas que SI ven el codo a partir del cual la tapada ya no cuenta
PESO_PERSPECTIVA_MIN = 0.15   # una direccion clavada en el eje optico no pesa 0: pesa esto
VIS_MIN = 0.35                # = VIS_MIN de seguimiento_brazos: por debajo el brazo no llega aqui
VIS_PLENA = 0.85              # visibilidad a partir de la cual pesa del todo
PESO_VIS_MIN = 0.0            # peso justo en VIS_MIN: un brazo entra sin salto
T_RAMPA = 0.3                 # s que tarda una vista nueva en pesar del todo
HUECO_RAMPA = 0.25            # s sin ver un brazo tras los que, al volver, repite la rampa
DESACUERDO_SUAVE_DEG = 20.0   # hasta aqui dos vistas se promedian con su peso completo
DESACUERDO_MAX_DEG = 40.0     # a partir de aqui la vista discrepante ya no cuenta
# La EDAD de un dato se mide desde que esta DISPONIBLE (cuando MediaPipe termina), no
# desde la captura: con dos camaras en un PC, MediaPipe puede tardar bastante y no por
# eso el dato es inservible. El instante de captura solo se usa para alinear camaras.
EDAD_LIBRE = 0.10             # s desde que el dato esta disponible: peso pleno
TAU_EDAD = 0.08               # s: despues su peso cae como exp(-(edad - EDAD_LIBRE) / TAU_EDAD)
EDAD_MAX = 0.40               # s: mas viejo que esto no entra
T_AVISO_IMU = 3.0             # s excluida por su IMU antes de avisar en pantalla
EDAD_PREVIO = 0.5             # s: la fusion anterior sirve de referencia durante este tiempo
MARGEN_CRUCE_DEG = 20.0       # cruzados tienen que encajar esto mejor (media por hueso)
HISTERESIS_MANO = 1.25        # la mano solo cambia de camara si la otra la ve 25 % mas grande
DT_MANOS = 0.06               # s: diferencia maxima de captura entre camaras para triangular una mano
                              # (los pixeles se adelantan con su velocidad hasta el instante comun)
MARGEN_INVARIANTE = 25.0      # grados: para decidir un cruce sin calibracion (flexion + elevacion)
# Cruces izquierda/derecha: filtro de Markov sobre "esta camara tiene izq/der cambiados".
P_CAMBIO_CRUCE = 0.02         # por foto: MediaPipe cambia sus etiquetas (en un sentido u otro)
P_SIN_CONTINUIDAD = 0.3       # idem si hace rato que la camara no daba brazos
# MediaPipe acierta el lado mas veces de las que falla (mas aun si ve la cara). Es lo unico
# que fija el sentido ABSOLUTO (lo demas es relativo), asi que entra como evidencia debil en
# cada foto: deshace un cruce mal asumido en unos segundos sin pisar uno bien detectado.
KAPPA_CRUCE = 0.95            # por foto, para "esta camara esta cruzada"
KAPPA_CRUCE_CARA = 0.90
TAU_CONTINUIDAD = 3.0         # grados: entre dos fotos seguidas un brazo apenas se mueve
RATIO_MAX = 1e4               # una sola foto no puede decidir mas que esto (por si hay un error raro)
TAU_ENTRE_CAMARAS = 15.0      # grados: dos camaras calibradas ven el mismo brazo casi igual
T_CONTINUIDAD = 0.2           # s: mas separadas, dos fotos no dicen nada de continuidad
ETIQUETAS = "ABCD"
OTRO = {"L": "R", "R": "L"}
HUESOS = ("dir_brazo", "dir_antebrazo")

# ------------------------------------------------------------------ puntos 3D
PUNTOS_BRAZO = {"L": (11, 13, 15), "R": (12, 14, 16)}   # hombro, codo, muneca (MediaPipe)
LADO_PUNTO = {13: "L", 15: "L", 14: "R", 16: "R"}        # codos y munecas
L_BRAZO_TIPICO, L_ANTEBRAZO_TIPICO = 0.30, 0.27          # m, para estimar la velocidad de un punto
PUNTOS_3D = (11, 12, 13, 14, 15, 16, 23, 24)
CRUCE_PUNTO = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 23: 24, 24: 23}
RADIO = {11: 0.05, 12: 0.05, 13: 0.035, 14: 0.035, 15: 0.025, 16: 0.025, 23: 0.07, 24: 0.07}
                              # m: de la piel que ve la camara al centro de la articulacion
VIS_PUNTO = 0.5               # visibilidad minima para usar un punto
SIGMA_PX = 5.0                # px: error tipico de un punto de MediaPipe
SIGMA_OAK = (0.015, 0.01)     # m: error de profundidad de la OAK-D = a + b * z^2
SIGMA_PRED = 0.08             # m: error de la profundidad prevista con MediaPipe
SIGMA_RAYO = 50.0             # m: "sin profundidad" (solo el rayo del pixel)
SIGMA_MAX_PUNTO = 0.3         # m: un punto peor determinado que esto (en alguna direccion) no vale
CHI2_MAX = 16.0               # dos camaras que no cuadran a 4 sigmas en un punto: se queda una
MAX_EXTRAP = 0.06             # s: lo maximo que se adelanta una camara al instante de la otra
FACTOR_EXTRAP = 0.7           # se adelanta algo menos de lo que dice la velocidad (es ruidosa)
RANGO_BRAZO = (0.15, 0.50)    # m: longitudes plausibles
RANGO_ANTEBRAZO = (0.12, 0.45)
RANGO_HOMBRO_MUNECA = (0.15, 0.85)
T_MODO = 0.4                  # s del paso gradual de direcciones a puntos
# Filtro de Kalman (posicion + velocidad) por articulacion, en el modo puntos. Cada
# camara se incorpora como una medida con SU incertidumbre (precisa de lado, ruidosa en
# profundidad): el ruido de profundidad se promedia en el tiempo sin el retraso de un
# filtro normal. Ruido de proceso = cuanto puede acelerar cada articulacion (m^2/s^3).
Q_KALMAN = {11: 0.005, 12: 0.005, 13: 0.1, 14: 0.1, 15: 0.2, 16: 0.2}   # ajustado en simulacion
CHI2_KALMAN = 16.0            # una medida mas lejos que esto (4 sigmas) se descarta (cruce, error)
T_KALMAN_MAX = 0.3            # s sin medidas: se reinicia
FILTRO_MIN_CUTOFF = 0.8       # One Euro de la salida por puntos (= seguimiento_brazos)
FILTRO_BETA = 0.7

# ------------------------------------------------------------------ calibracion con [c]
DURACION_CALIB = 1.5          # s de frames que se promedian
MIN_FRAMES_CALIB = 8          # por camara
MAX_FLEXION_CALIB = 45.0      # grados: con los brazos colgando el codo esta casi recto (con ruido real)
MAX_ENTRE_BRAZOS_CALIB = 50.0 # grados: y los dos brazos casi paralelos
MAX_COS_HOMBROS_CALIB = 0.5   # la linea de hombros casi perpendicular a los brazos
MAX_DIF_IMU = 20.0            # grados: si la gravedad discrepa mas de los brazos, la IMU no es fiable
AVISO_DISCREPANCIA_C = 8.0    # grados entre lo que dicen dos camaras al pulsar c: avisar

# ------------------------------------------------------------------ calibracion automatica: rotacion
DT_PAR = 0.05                 # s: diferencia maxima de captura entre camaras para emparejar
VEL_MAX_PAR = 90.0            # deg/s: con el hueso moviendose mas rapido no se empareja
PESO_MIN_PAR = 0.12           # calidad minima de una pareja
VENTANA_PARES = 20.0          # s de parejas que se guardan
MAX_PARES = 1500
PERIODO_SOLVER = 0.5          # s entre resoluciones
MIN_PARES = 40                # parejas para intentar resolver
RES_ESCALA = 8.0              # deg: escala del peso robusto
RES_INLIER = 20.0             # deg: por debajo una pareja "cuadra"
RES_MAX_OK = 14.0             # deg: mediana de residuos para dar la solucion por buena (datos reales: 8-14)
MIN_INLIERS = 0.5             # fraccion (por peso) de parejas que cuadran
VARIEDAD_MIN = {"libre": 0.06, "gravedad": 0.12}   # direcciones suficientemente variadas
# Para ACEPTAR una calibracion automatica hacen falta pruebas solidas. Con IMU solo falta el
# giro alrededor de la vertical, y ese giro solo lo dicen los huesos que NO son verticales:
# con los brazos colgando (o balanceandose al andar, en un solo plano y en contrafase) hay
# soluciones falsas que encajan casi igual (p. ej. 180 grados). Asi que se exige:
MIN_PARES_ACEPTAR = 150       # parejas en total
MIN_HORIZONTALES = 80         # parejas de huesos claramente no verticales
RES_H_OK = 10.0               # deg: mediana del error de giro en el plano horizontal
INLIERS_H_OK = 0.65           # fraccion de esas parejas que cuadran (a menos de 20 deg)
DISPERSION_AZ_MAX = 0.8       # direcciones variadas (no todas en un mismo plano vertical)
AMBIGUEDAD_MAX = 0.3          # otra solucion distinta no puede tener ni un 30 % del apoyo
RES_LIBRE_OK = 10.0           # sin IMU: mas exigente que con ella
IMU_INCOHERENTE_DEG = 4.0     # la solucion libre (sin IMU) difiere de la de gravedad mas que esto...
MEJORA_LIBRE_DEG = 1.5        # ...y encaja mejor por este margen (mediana): la IMU de esa camara miente
INLIERS_LIBRE_OK = 0.7
MAX_DERIVA = 8.0              # deg: el afinado automatico no aleja una camara mas que esto de
                              # su calibracion de referencia (la de la c o la aceptada)
RES_DESALINEADA = 25.0        # deg: residuos recientes por encima -> la camara se aparta
RES_REALINEADA = 12.0         # deg: y vuelve si bajan de aqui
DISCREPANCIA_C_MAX = 10.0     # deg: al pulsar c, si dos camaras discrepan mas, no se promedian
FRONTALIDAD_C_MIN = 0.3       # al pulsar c, una camara que no te ve la cara no calibra (izq/der dudosos)
REFINO = 0.2                  # fraccion hacia la nueva solucion en cada afinado
REFINO_MIN_DEG = 0.3
VENTANA_RECIENTE = 4.0        # s: residuos recientes para detectar una camara movida sin IMU
RES_MOVIDA = 25.0             # deg (sin IMU)
RES_MOVIDA_IMU = 40.0         # deg: con IMU en las dos camaras, el movimiento lo detecta la IMU
MIN_PARES_MOVIDA = 40
DURACION_AVISO = 6.0          # s en pantalla

# ------------------------------------------------------------------ calibracion automatica: posicion
VIS_MUESTRA_T = 0.7           # visibilidad minima de un punto para situar camaras
VEL_MAX_T = 0.4               # m/s: con el punto moviendose mas rapido no se usa
MIN_MUESTRAS_T = 60
RES_T_ESCALA = 0.04           # m: escala del peso robusto
RES_T_INLIER = 0.12           # m
ERROR_T_OK = 0.02             # m: error tipico de la posicion estimada para darla por buena
                              # (cada muestra puede tener 10 cm de ruido si la camara esta lejos;
                              # con muchas muestras la media es precisa igualmente)
RUIDO_T_MAX = 0.20            # m: muestras mas ruidosas que esto (por eje): algo va mal
REFINO_T = 0.2
RES_T_MOVIDA = 0.20           # m: residuos recientes por encima: la camara se ha movido
DESPL_CONFIRMADO = 0.03       # m: tras un aviso de desplazamiento de la IMU, lo que tiene que haberse
                              # movido segun las medidas del cuerpo (antes frente a despues) para creerselo
DESPL_CONFIRMADO_SIN_REF = 0.10   # m: idem si no habia medidas de antes: se compara con la posicion
                                  # guardada, y la del cuerpo tiene unos cm de sesgo (piel, ruido)
T_VERIFICAR_MAX = 20.0        # s: sin medidas para comprobarlo, se deja como estaba


# ================================================================== utilidades
def _unitario(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def angulo_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _suave(x):
    """0 -> 0, 1 -> 1, con derivada nula en los extremos (smoothstep)."""
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def rot_eje(eje, ang):
    """Rotacion de 'ang' radianes alrededor de 'eje' (Rodrigues)."""
    k = _unitario(np.asarray(eje, dtype=float))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def log_rot(R):
    """Vector eje * angulo (rad) de una rotacion."""
    ang = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
    if ang < 1e-9:
        return np.zeros(3)
    if np.pi - ang < 1e-4:   # ~180 grados: el eje sale de R + I
        M = R + np.eye(3)
        return _unitario(M[:, int(np.argmax(np.linalg.norm(M, axis=0)))]) * ang
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return v * ang


def interpolar_rot(R0, R1, f):
    """Rotacion a una fraccion f del camino de R0 a R1."""
    d = log_rot(R1 @ R0.T)
    n = np.linalg.norm(d)
    return R0.copy() if n < 1e-12 else rot_eje(d, n * f) @ R0


def alinear(u, v):
    """Rotacion minima que lleva el vector u al v."""
    u, v = _unitario(u), _unitario(v)
    eje = np.cross(u, v)
    s, c = np.linalg.norm(eje), float(np.dot(u, v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-3:
            perp = np.cross(u, [0.0, 1.0, 0.0])
        return rot_eje(perp, np.pi)
    return rot_eje(eje, np.arctan2(s, c))


def angulo_rotacion_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def media_rotaciones(Rs):
    """Rotacion media (cordal): suma de matrices proyectada a SO(3) con SVD."""
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def marco_por_defecto(arriba=None):
    """R camara -> operador suponiendo el operador de frente a la camara. Con la
    gravedad de la IMU se corrige la inclinacion de la camara; sin ella, horizontal."""
    if arriba is None:
        return R_CAM_DEFECTO.copy()
    z = _unitario(np.asarray(arriba, dtype=float))
    x = np.array([0.0, 0.0, -1.0])
    x = x - z * np.dot(x, z)
    if np.linalg.norm(x) < 0.2:      # camara mirando al suelo o al techo
        return R_CAM_DEFECTO.copy()
    x = _unitario(x)
    return np.vstack([x, np.cross(z, x), z])


class _ListaLandmarks(list):
    """Lista de landmarks que ademas tiene .landmark (ella misma): vale tanto para el
    formato de la API Tasks (lista) como para el de la API antigua (obj.landmark[i])."""

    @property
    def landmark(self):
        return self


def _rotar_landmarks(lms, R):
    """Copia de una lista de landmarks (x, y, z) rotados con R, con los demas campos."""
    lista = getattr(lms, "landmark", lms)
    P = np.array([[l.x, l.y, l.z] for l in lista]) @ R.T
    return _ListaLandmarks(
        SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                        visibility=getattr(l, "visibility", None), presence=getattr(l, "presence", None),
                        name=getattr(l, "name", None))
        for p, l in zip(P, lista))


def punto_camara(pt, K, indice):
    """Centro de la articulacion en el marco de la camara (m), o None sin profundidad.
    La camara mide la piel de su lado: el centro esta RADIO mas alla, por el rayo."""
    if pt is None or pt["z"] is None:
        return None
    (u, v), z = pt["uv"], pt["z"]
    p = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    return p + RADIO[indice] * _unitario(p)


class FiltroOneEuro:
    """Filtro One Euro (Casiez et al., 2012) para vectores: poco temblor en reposo y
    poco retraso al moverse. Se reinicia solo si deja de recibir datos un rato."""

    def __init__(self, min_cutoff=None, beta=None, d_cutoff=1.0):
        self.min_cutoff = FILTRO_MIN_CUTOFF if min_cutoff is None else min_cutoff
        self.beta = FILTRO_BETA if beta is None else beta
        self.d_cutoff = d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alfa(corte, dt):
        tau = 1.0 / (2 * np.pi * corte)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None or t - self.t > 0.3:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self.t
        if dt < 1e-4:
            return self.x.copy()
        a_d = self._alfa(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alfa(self.min_cutoff + self.beta * np.linalg.norm(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


# ================================================================== pesos
def peso_fuente(fuente):
    base = PESO_FUENTE.get(fuente.split()[0], 0.4)
    return base * (FACTOR_CODO_TAPADO if "codo tapado" in fuente else 1.0)


def peso_visibilidad(brazo):
    vis = brazo.get("vis", 1.0)   # sin el campo (version antigua de seguimiento_brazos): neutro
    return PESO_VIS_MIN + (1 - PESO_VIS_MIN) * _suave((vis - VIS_MIN) / (VIS_PLENA - VIS_MIN))


def peso_perspectiva(d_cam):
    """d_cam: direccion unitaria en el marco de SU camara (z = eje optico)."""
    return max(PESO_PERSPECTIVA_MIN, float(np.sqrt(max(0.0, 1.0 - d_cam[2] ** 2))))


def peso_edad(edad):
    return float(np.exp(-max(0.0, edad - EDAD_LIBRE) / TAU_EDAD))


def calidad_hueso(brazo, d_cam):
    """Lo fiable que es la medida de un hueso en su camara (sin edad ni rampa)."""
    return peso_fuente(brazo["fuente"]) * peso_visibilidad(brazo) * peso_perspectiva(d_cam)


def combinar(vectores, pesos, previo=None):
    """Media ponderada de direcciones unitarias. Devuelve (direccion, desacuerdo_deg).
    La referencia es la vista de mas peso; si se da 'previo' (la direccion fusionada
    del frame anterior) se favorece la vista que la continua. Las vistas que se alejan
    de la referencia pierden peso entre DESACUERDO_SUAVE_DEG y DESACUERDO_MAX_DEG."""
    puntuacion = list(pesos)
    if previo is not None:
        puntuacion = [w * (0.6 + 0.4 * max(0.0, float(np.dot(v, previo))))
                      for v, w in zip(vectores, pesos)]
    ref = vectores[int(np.argmax(puntuacion))]
    suma, desacuerdo = np.zeros(3), 0.0
    for v, w in zip(vectores, pesos):
        a = angulo_deg(v, ref)
        if w > 1e-3:
            desacuerdo = max(desacuerdo, a)
        x = (a - DESACUERDO_SUAVE_DEG) / (DESACUERDO_MAX_DEG - DESACUERDO_SUAVE_DEG)
        suma += w * (1.0 - _suave(x)) * v
    n = np.linalg.norm(suma)
    return (suma / n if n > 1e-6 else ref), desacuerdo


def tamano_mano(manos_dibujo, lado):
    """Diagonal (px) de la mano de ese lado, para decidir que camara la ve mejor."""
    for m in manos_dibujo or []:
        if m["lado"] == lado:
            p = m["puntos_px"]
            return float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
    return 0.0


def _cruzar(snap):
    """Copia del resultado de una camara con izquierda y derecha intercambiadas."""
    s = dict(snap)
    s["brazos"] = {OTRO[l]: b for l, b in snap["brazos"].items()}
    s["aperturas"] = {OTRO[l]: a for l, a in snap["aperturas"].items()}
    s["manos"] = [dict(m, lado=OTRO[m["lado"]]) for m in (snap.get("manos") or [])]
    if snap.get("puntos"):
        s["puntos"] = {CRUCE_PUNTO.get(i, i): p for i, p in snap["puntos"].items()}
    if snap.get("linea_hombros") is not None:
        s["linea_hombros"] = -snap["linea_hombros"]
    return s


def _t_disp(snap):
    """Instante en que el resultado de una camara estuvo disponible (si el hilo no lo
    da, el de captura)."""
    return snap.get("t_proc", snap["t"])


def _frontalidad(snap):
    """Cuanto se le ve la cara al operador (0-1): visibilidad de nariz y ojos."""
    lm = snap.get("lm2d")
    if not lm:
        return None
    return float(np.mean([lm[i][2] for i in (0, 2, 5)]))


def _rasgos(snap):
    """Por lado: (flexion del codo, elevacion del brazo respecto al tronco) en grados.
    No dependen de como este girada la camara, asi que permiten comparar dos camaras
    aunque no esten calibradas entre si."""
    tronco = None
    pts, K = snap.get("puntos"), snap.get("K")
    if pts and K is not None:
        p = {i: punto_camara(pts.get(i), K, i) for i in (11, 12, 23, 24)}
        if all(v is not None for v in p.values()):
            tronco = _unitario((p[11] + p[12]) / 2 - (p[23] + p[24]) / 2)
    out = {}
    for lado, b in snap["brazos"].items():
        if "codo tapado" in b["fuente"]:
            continue
        out[lado] = (angulo_deg(b["dir_brazo"], b["dir_antebrazo"]),
                     angulo_deg(b["dir_brazo"], -tronco) if tronco is not None else None)
    return out


def cruce_invariante(sa, sb):
    """True si los brazos de la camara b estan cambiados respecto a los de a, False si
    no, None si no se puede saber (brazos parecidos o no se ven los dos)."""
    ra, rb = _rasgos(sa), _rasgos(sb)
    if len(ra) < 2 or len(rb) < 2:
        return None

    def d(x, y):
        s = abs(x[0] - y[0])
        if x[1] is not None and y[1] is not None:
            s += abs(x[1] - y[1])
        return s

    d_dir = d(ra["L"], rb["L"]) + d(ra["R"], rb["R"])
    d_cru = d(ra["L"], rb["R"]) + d(ra["R"], rb["L"])
    if d_cru + MARGEN_INVARIANTE < d_dir:
        return True
    if d_dir + MARGEN_INVARIANTE < d_cru:
        return False
    return None


def _distancias_cruce(ba, bb, Ra=None, Rb=None):
    """Angulo medio entre los huesos de dos juegos de brazos (directo y cruzando izq/der).
    Si se dan Ra/Rb se rotan antes (camaras distintas ya calibradas). None si no hay con
    que comparar."""
    directo, cruzado = [], []
    for lado, b in ba.items():
        for h in HUESOS:
            va = b[h] if Ra is None else Ra @ b[h]
            for otro, lista in ((lado, directo), (OTRO[lado], cruzado)):
                if otro in bb:
                    vb = bb[otro][h] if Rb is None else Rb @ bb[otro][h]
                    lista.append(angulo_deg(va, vb))
    if not directo or not cruzado:
        return None
    return float(np.mean(directo)), float(np.mean(cruzado))


class FiltroCruces:
    """Probabilidad de que cada camara tenga izquierda y derecha cambiadas (filtro de
    Markov sobre las 2^N combinaciones). Evidencias:
      - continuidad: entre dos fotos seguidas de una camara los brazos apenas se mueven;
        si de repente su brazo 'L' se parece al 'R' de la foto anterior, MediaPipe ha
        cambiado las etiquetas (no hace falta calibracion);
      - acuerdo entre camaras: SOLO entre camaras calibradas y en uso (el mismo brazo apunta
        igual en las dos);
      - MediaPipe suele acertar, mas aun si ve la cara: solo como tendencia, no como regla.
    Con los brazos simetricos no hay evidencia y no se decide nada a la ligera.

    (oct-2026) Antes tambien se comparaban camaras SIN calibrar entre si con rasgos
    invariantes (flexion + elevacion). Con los brazos colgando esos rasgos son iguales en
    los dos brazos y la diferencia es ruido/sesgo de cada camara, que se acumulaba foto a
    foto como si fuera evidencia independiente. El filtro solo sabe que "A y B no cuadran":
    para decidir CUAL esta cruzada tira de la cara, y en la demo A (que movia el robot) no
    veia la cabeza y B si -> se daba la vuelta a A y el robot cruzaba los brazos con el
    operador quieto. Una camara sin calibrar no esta en la fusion: sus cruces no importan
    para el robot (la calibracion automatica decide los suyos con cruce_invariante())."""

    def __init__(self, n):
        self.n = n
        self.p = np.zeros(2 ** n)
        self.p[0] = 1.0
        self.prev = [None] * n          # (brazos sin corregir, t) de la foto anterior de cada camara

    def _bit(self, s, i):
        return (s >> i) & 1

    def actualizar(self, snaps, nuevos, calibradas, R):
        estados = np.arange(2 ** self.n)
        for i in nuevos:
            s = snaps[i]
            b = s["brazos"]
            prev = self.prev[i]
            fr = _frontalidad(s)
            kappa = KAPPA_CRUCE_CARA if (fr is not None and fr > 0.5) else KAPPA_CRUCE
            d, p_cambio = None, P_CAMBIO_CRUCE
            if prev is not None and 0 < s["t"] - prev[1] < T_CONTINUIDAD:
                d = _distancias_cruce(b, prev[0])
            else:
                p_cambio = P_SIN_CONTINUIDAD
            if d is None:
                l_igual, l_cambio = 1.0, 1.0
            else:
                r = float(np.clip((d[1] - d[0]) / TAU_CONTINUIDAD, -np.log(RATIO_MAX), np.log(RATIO_MAX)))
                l_igual, l_cambio = 1.0, float(np.exp(-r))
            nuevo = np.zeros_like(self.p)
            for est in estados:
                if self.p[est] == 0:
                    continue
                a = self._bit(est, i)
                for bnuevo in (0, 1):
                    tr = p_cambio if bnuevo != a else 1 - p_cambio
                    em = (l_igual if bnuevo == a else l_cambio) * (kappa if bnuevo else 1.0)
                    nuevo[(est & ~(1 << i)) | (bnuevo << i)] += self.p[est] * tr * em
            self.p = nuevo / max(nuevo.sum(), 1e-300)
            if b:
                self.prev[i] = (b, s["t"])
        # acuerdo entre camaras (una vez por foto nueva de alguna de las dos)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if (i not in nuevos and j not in nuevos) or snaps[i] is None or snaps[j] is None:
                    continue
                if not snaps[i]["brazos"] or not snaps[j]["brazos"]:
                    continue
                if i not in calibradas or j not in calibradas:
                    continue        # sin calibracion entre ellas no hay evidencia fiable (ver arriba)
                d = _distancias_cruce(snaps[i]["brazos"], snaps[j]["brazos"], R[i], R[j])
                tau = TAU_ENTRE_CAMARAS
                if d is None:
                    continue
                r = float(np.clip((d[1] - d[0]) / tau, -np.log(RATIO_MAX), np.log(RATIO_MAX)))
                l_ac, l_des = 1.0, float(np.exp(-r))
                for est in estados:
                    self.p[est] *= l_ac if self._bit(est, i) == self._bit(est, j) else l_des
                self.p /= max(self.p.sum(), 1e-300)
        self.p = np.maximum(self.p, 1e-12)
        self.p /= self.p.sum()
        mejor = int(np.argmax(self.p))
        return [bool(self._bit(mejor, i)) for i in range(self.n)]

    def probabilidad_cruce(self, i):
        return float(sum(self.p[s] for s in range(2 ** self.n) if self._bit(s, i)))


class KalmanPunto:
    """Posicion y velocidad de una articulacion (velocidad casi constante)."""

    def __init__(self, p, C, t, q):
        self.x = np.r_[p, 0.0, 0.0, 0.0]
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = C
        self.P[3:, 3:] = np.eye(3) * 0.5 ** 2
        self.t, self.q, self.rechazos = t, q, 0

    def predecir(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        Fm = np.eye(6)
        Fm[:3, 3:] = dt * np.eye(3)
        I3 = np.eye(3)
        Q = self.q * np.block([[dt ** 3 / 3 * I3, dt ** 2 / 2 * I3], [dt ** 2 / 2 * I3, dt * I3]])
        self.x = Fm @ self.x
        self.P = Fm @ self.P @ Fm.T + Q
        self.t = t

    def actualizar(self, z, R):
        y = z - self.x[:3]
        S = self.P[:3, :3] + R
        Si = np.linalg.inv(S)
        if float(y @ Si @ y) > CHI2_KALMAN:
            self.rechazos += 1
            return False
        K = self.P[:, :3] @ Si
        self.x = self.x + K @ y
        self.P = self.P - K @ self.P[:3, :]
        self.rechazos = 0
        return True


def _imu_de(hilo):
    return getattr(getattr(hilo, "camara", None), "imu", None)


# ================================================================== calibracion con [c]
def calibrar(brazos, linea_hombros, arriba=None):
    """Con los brazos colgando: 'abajo' = media de las direcciones de los brazos (o la
    gravedad, si se da 'arriba' y cuadra), 'izquierda' = linea de hombros.
    Devuelve (R camara -> operador, angulo IMU-brazos en grados o None)."""
    abajo = sum(brazos[l]["dir_brazo"] + brazos[l]["dir_antebrazo"] for l in ("L", "R"))
    z = -abajo / np.linalg.norm(abajo)
    ang_imu = None
    if arriba is not None:
        ang_imu = angulo_deg(_unitario(arriba), z)
        if ang_imu < MAX_DIF_IMU:
            z = _unitario(arriba)
    y = linea_hombros - z * np.dot(linea_hombros, z)
    y /= np.linalg.norm(y)
    return np.vstack([np.cross(y, z), y, z]), ang_imu


def postura_de_calibracion(snap):
    """None si el frame vale para calibrar; si no, el motivo (texto corto)."""
    if snap is None:
        return "sin imagen"
    b = snap["brazos"]
    if "L" not in b or "R" not in b:
        return "no ve los dos brazos"
    if snap.get("linea_hombros") is None:
        return "no ve los hombros"
    if any("codo tapado" in b[l]["fuente"] for l in ("L", "R")):
        return "codo tapado"
    for l in ("L", "R"):
        if angulo_deg(b[l]["dir_brazo"], b[l]["dir_antebrazo"]) > MAX_FLEXION_CALIB:
            return "brazos doblados"
    abajo = {l: _unitario(b[l]["dir_brazo"] + b[l]["dir_antebrazo"]) for l in ("L", "R")}
    if angulo_deg(abajo["L"], abajo["R"]) > MAX_ENTRE_BRAZOS_CALIB:
        return "brazos no paralelos"
    if abs(float(np.dot(snap["linea_hombros"], _unitario(abajo["L"] + abajo["R"])))) > MAX_COS_HOMBROS_CALIB:
        return "hombros raros"
    return None


class CalibracionCamaras:
    """Junta DURACION_CALIB s de frames nuevos de cada camara (solo los que tienen
    postura de calibracion) y promedia la rotacion camara -> operador de cada una.
    Uso: crearla al pulsar [c], llamar a alimentar() en cada vuelta del bucle y,
    cuando terminada() sea True, leer resultado()."""

    def __init__(self, hilos, duracion=DURACION_CALIB, reloj=time.monotonic, fusion=None):
        """fusion: si se da, se usan sus resultados ya corregidos de cruces izq/der (la
        postura de calibracion depende de que izquierda y derecha esten bien)."""
        self.hilos = list(hilos)
        self.fusion = fusion
        self.reloj = reloj
        self.imus = [_imu_de(h) for h in self.hilos]
        self.t_fin = reloj() + duracion
        self.muestras = [[] for _ in self.hilos]
        self.angulos_imu = [[] for _ in self.hilos]
        self.rechazos = [{} for _ in self.hilos]
        self._n = [-1] * len(self.hilos)

    def alimentar(self):
        for i, h in enumerate(self.hilos):
            s = self.fusion.corregido(i) if self.fusion is not None else None
            if s is None:
                s = h.ultimo()
            if s is None or s["n"] == self._n[i]:
                continue
            self._n[i] = s["n"]
            motivo = postura_de_calibracion(s)
            fr = _frontalidad(s)
            if motivo is None and fr is not None and fr < FRONTALIDAD_C_MIN:
                motivo = "no te ve la cara"
            if motivo is None:
                imu = self.imus[i]
                arriba = imu.arriba() if (imu is not None and imu.R is not None) else None
                R, ang = calibrar(s["brazos"], s["linea_hombros"], arriba)
                self.muestras[i].append(R)
                if ang is not None:
                    self.angulos_imu[i].append(ang)
            else:
                self.rechazos[i][motivo] = self.rechazos[i].get(motivo, 0) + 1

    def restante(self):
        return max(0.0, self.t_fin - self.reloj())

    def terminada(self):
        return self.restante() == 0.0

    def resultado(self):
        """[(R o None, frames_usados, dispersion_deg, motivo_principal_de_rechazo,
        angulo_medio_IMU_brazos_deg o None)] por camara."""
        salida = []
        for Rs, angs, rech in zip(self.muestras, self.angulos_imu, self.rechazos):
            motivo = max(rech, key=rech.get) if rech else None
            ang = float(np.mean(angs)) if angs else None
            if len(Rs) < MIN_FRAMES_CALIB:
                salida.append((None, len(Rs), None, motivo or "pocos frames", ang))
                continue
            R = media_rotaciones(Rs)
            dispersion = float(np.mean([angulo_rotacion_deg(Rk @ R.T) for Rk in Rs]))
            salida.append((R, len(Rs), dispersion, motivo, ang))
        return salida


# ================================================================== calibracion automatica
class SolucionadorRotacion:
    """Parejas (a, b) con a ~= R @ b: 'a' es la direccion de un hueso en W (vista por
    una camara ya calibrada) y 'b' la del mismo hueso en el marco de ESTA camara.
    Resuelve R (camara -> W) de forma robusta: unas pocas parejas malas (izquierda y
    derecha cambiadas, frames desfasados) no la estropean. Las parejas siguen valiendo
    aunque la camara de referencia se mueva despues; solo las invalida que se mueva ESTA."""

    def __init__(self):
        self.pares = deque(maxlen=MAX_PARES)     # (t, a, b, w)
        self._rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.pares)

    def anadir(self, t, a, b, w):
        self.pares.append((t, np.asarray(a, float), np.asarray(b, float), float(w)))

    def vaciar(self, antes_de=None):
        """Sin argumento borra todo; con 'antes_de' solo lo anterior a ese instante."""
        if antes_de is None:
            self.pares.clear()
        else:
            self.pares = deque((p for p in self.pares if p[0] >= antes_de), maxlen=MAX_PARES)

    def purgar(self, ahora):
        if self.pares and self.pares[0][0] < ahora - VENTANA_PARES:
            self.vaciar(antes_de=ahora - VENTANA_PARES)

    def transformar(self, G):
        """W se ha girado con G (tecla c): las direcciones de referencia tambien."""
        self.pares = deque(((t, G @ a, b, w) for t, a, b, w in self.pares), maxlen=MAX_PARES)

    def _arrays(self, desde=None):
        sel = [p for p in self.pares if desde is None or p[0] >= desde]
        if not sel:
            return None, None, None
        return (np.array([p[1] for p in sel]), np.array([p[2] for p in sel]),
                np.array([p[3] for p in sel]))

    @staticmethod
    def residuos_deg(R, A, B):
        return np.degrees(np.arccos(np.clip(np.sum(A * (B @ R.T), axis=1), -1.0, 1.0)))

    @staticmethod
    def _wahba(A, B, W):
        """R que minimiza sum w |a - R b|^2 (SVD)."""
        M = (A * W[:, None]).T @ B
        U, _, Vt = np.linalg.svd(M)
        return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt

    def _libre(self, A, B, W, R0=None):
        """Sin gravedad: 3 grados de libertad. Arranque por muestreo de parejas (tipo
        RANSAC) y despues minimos cuadrados con pesos robustos (Cauchy)."""
        n = len(W)
        if R0 is None:
            mejor, puntos = None, -1.0
            p = W / W.sum()
            for _ in range(80):
                i, j = self._rng.choice(n, 2, replace=False, p=p)
                if angulo_deg(B[i], B[j]) < 20 or angulo_deg(A[i], A[j]) < 20:
                    continue
                R = self._wahba(A[[i, j]], B[[i, j]], np.ones(2))
                s = float(np.sum(W * (self.residuos_deg(R, A, B) < RES_INLIER)))
                if s > puntos:
                    mejor, puntos = R, s
            R0 = mejor if mejor is not None else self._wahba(A, B, W)
        R = R0
        for _ in range(6):
            r = self.residuos_deg(R, A, B)
            R = self._wahba(A, B, W / (1.0 + (r / RES_ESCALA) ** 2))
        return R

    @staticmethod
    def _con_gravedad(A, B, W, arriba_obj, arriba_cam):
        """Con IMU: la gravedad fija la inclinacion y solo queda el giro psi alrededor de
        la vertical. Arranque con el histograma de psi de todas las parejas (robusto) y
        afinado con pesos robustos."""
        u = _unitario(arriba_obj)
        Ral = alinear(arriba_cam, u)
        Bp = B @ Ral.T
        Ah = A - np.outer(A @ u, u)
        Bh = Bp - np.outer(Bp @ u, u)
        na, nb = np.linalg.norm(Ah, axis=1), np.linalg.norm(Bh, axis=1)
        ok = (na > 0.2) & (nb > 0.2)            # huesos casi verticales no dicen nada del giro
        if ok.sum() < MIN_PARES // 2:
            return None
        s = np.cross(Bh[ok], Ah[ok]) @ u
        c = np.sum(Bh[ok] * Ah[ok], axis=1)
        psi_k = np.arctan2(s, c)
        wk = W[ok] * na[ok] * nb[ok]
        hist = np.bincount(((psi_k + np.pi) / (2 * np.pi) * 72).astype(int) % 72, weights=wk, minlength=72)
        hist = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
        psi = (np.argmax(hist) + 0.5) / 72 * 2 * np.pi - np.pi
        escala = np.radians(RES_ESCALA)
        for _ in range(8):
            d = np.angle(np.exp(1j * (psi_k - psi)))
            wr = wk / (1.0 + (d / escala) ** 2)
            psi += float(np.arctan2(np.sum(wr * np.sin(d)), np.sum(wr * np.cos(d))))
        return rot_eje(u, psi) @ Ral

    def resolver(self, R0=None, arriba_obj=None, arriba_cam=None):
        """dict(R, ok, n, mediana, inliers, variedad, modo) o None si aun no hay datos."""
        A, B, W = self._arrays()
        if A is None or len(W) < MIN_PARES:
            return None
        R, modo = None, "libre"
        imu_incoherente = None
        if arriba_obj is not None and arriba_cam is not None:
            R, modo = self._con_gravedad(A, B, W, arriba_obj, arriba_cam), "gravedad"
            if R is not None:
                # (demo oct-2026) Si la gravedad de la IMU de esta camara esta desviada (extrinseca
                # IMU->RGB, sesgo), el modo gravedad hereda ese error en la inclinacion y aun asi
                # encaja "bastante" (residuo 12-17 grados): se acepto una calibracion a ~20-30
                # grados de la buena. La solucion libre (sin IMU) lo destapa: si encaja claramente
                # mejor y es otra, la IMU de esta camara no es fiable y se usa la libre.
                R_l = self._libre(A, B, W, R)
                med = lambda Rx: float(np.median(self.residuos_deg(Rx, A, B)))
                dif = angulo_rotacion_deg(R_l @ R.T)
                if dif > IMU_INCOHERENTE_DEG and med(R_l) + MEJORA_LIBRE_DEG < med(R):
                    R, modo, imu_incoherente = R_l, "libre", dif
        if R is None:
            R, modo = self._libre(A, B, W, R0), "libre"
        r = self.residuos_deg(R, A, B)
        dentro = r < RES_INLIER
        inliers = float(W[dentro].sum() / W.sum())
        mediana = float(np.median(r[dentro])) if dentro.any() else 180.0
        if modo == "gravedad":
            u = _unitario(arriba_obj)
            Bo = B @ R.T
            variedad = float(np.sum(W * (1 - (Bo @ u) ** 2)) / W.sum())
            calidad = self._calidad_horizontal(A, Bo, W, u)
            # (demo oct-2026) tambien el residuo GLOBAL: se acepto una solucion con 17 grados de
            # mediana (parejas sin estructura: con ruido real una buena da 8-13) que estaba a 30
            # grados de la buena; la fusion la tuvo que apartar enseguida ("NO CUADRA").
            ok = (len(W) >= MIN_PARES_ACEPTAR and mediana <= RES_MAX_OK and calidad["n_h"] >= MIN_HORIZONTALES
                  and calidad["mediana_h"] <= RES_H_OK and calidad["inliers_h"] >= INLIERS_H_OK
                  and calidad["dispersion"] <= DISPERSION_AZ_MAX and calidad["ambiguedad"] <= AMBIGUEDAD_MAX)
        else:
            C = (B * W[:, None]).T @ B / W.sum()
            variedad = float(np.sort(np.linalg.eigvalsh(C))[-2])
            calidad = {}
            ok = (len(W) >= MIN_PARES_ACEPTAR and inliers >= INLIERS_LIBRE_OK
                  and mediana <= RES_LIBRE_OK and variedad >= VARIEDAD_MIN[modo])
        return dict(R=R, ok=ok, n=len(W), mediana=mediana, inliers=inliers, variedad=variedad, modo=modo,
                    imu_incoherente=imu_incoherente, **calidad)

    @staticmethod
    def _calidad_horizontal(A, Bo, W, u):
        """Como de bien queda fijado el giro alrededor de la vertical (u), usando solo los
        huesos claramente no verticales: error de giro, variedad de direcciones y si hay
        otra solucion distinta con apoyo (ambiguedad)."""
        e1 = _unitario(np.cross(u, [1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.cross(u, [0.0, 1.0, 0.0]))
        e2 = np.cross(u, e1)
        az_a = np.arctan2(A @ e2, A @ e1)
        az_b = np.arctan2(Bo @ e2, Bo @ e1)
        horiz = (np.sqrt((A @ e1) ** 2 + (A @ e2) ** 2) > 0.5) & (np.sqrt((Bo @ e1) ** 2 + (Bo @ e2) ** 2) > 0.5)
        n_h = int(horiz.sum())
        if n_h < 10:
            return dict(n_h=n_h, mediana_h=180.0, inliers_h=0.0, dispersion=1.0, ambiguedad=1.0)
        w = W[horiz]
        res = np.degrees(np.angle(np.exp(1j * (az_b[horiz] - az_a[horiz]))))     # con signo
        ab = np.abs(res)
        dentro = ab < 20.0
        inliers_h = float(w[dentro].sum() / w.sum())
        mediana_h = float(np.median(ab[dentro])) if dentro.any() else 180.0
        dispersion = float(np.abs(np.sum(w * np.exp(2j * az_a[horiz]))) / w.sum())   # 1 = todo en un plano
        apoyo = w[ab < 20.0].sum()
        otra = max(w[np.abs(np.angle(np.exp(1j * np.radians(res - c)))) < np.radians(20)].sum()
                   for c in range(-160, 181, 10) if abs(c) >= 60)
        ambiguedad = float(otra / max(apoyo, 1e-9))
        return dict(n_h=n_h, mediana_h=mediana_h, inliers_h=inliers_h, dispersion=dispersion,
                    ambiguedad=ambiguedad)

    def residuos_recientes(self, R, desde):
        """(mediana de residuos en grados, numero de parejas) de las parejas desde 'desde'."""
        A, B, W = self._arrays(desde)
        if A is None:
            return 0.0, 0
        return float(np.median(self.residuos_deg(R, A, B))), len(W)


class SolucionadorTraslacion:
    """Muestras (P, q): P es un punto del cuerpo en W (medido por una camara ya
    situada) y q el mismo punto en el marco de ESTA camara (centro de la articulacion,
    ya corregido de la piel). Cada muestra dice t = P - R q; se combinan de forma
    robusta. Se guarda q (y no t) para usar siempre la R mas reciente de la camara."""

    def __init__(self):
        self.muestras = deque(maxlen=MAX_PARES)    # (t, P, q, w)

    def __len__(self):
        return len(self.muestras)

    def anadir(self, t, P, q, w):
        self.muestras.append((t, np.asarray(P, float), np.asarray(q, float), float(w)))

    def vaciar(self, antes_de=None):
        if antes_de is None:
            self.muestras.clear()
        else:
            self.muestras = deque((m for m in self.muestras if m[0] >= antes_de), maxlen=MAX_PARES)

    def purgar(self, ahora):
        if self.muestras and self.muestras[0][0] < ahora - VENTANA_PARES:
            self.vaciar(antes_de=ahora - VENTANA_PARES)

    def transformar(self, G):
        self.muestras = deque(((t, G @ P, q, w) for t, P, q, w in self.muestras), maxlen=MAX_PARES)

    def _estimaciones(self, R, desde=None):
        sel = [m for m in self.muestras if desde is None or m[0] >= desde]
        if not sel:
            return None, None
        P = np.array([m[1] for m in sel])
        Q = np.array([m[2] for m in sel])
        return P - Q @ R.T, np.array([m[3] for m in sel])

    def resolver(self, R, t0=None):
        """dict(t, ok, n, mediana, inliers, ruido, error) o None si aun no hay datos.
        'ruido' es el de cada muestra (por eje) y 'error' el de la posicion estimada."""
        X, W = self._estimaciones(R)
        if X is None or len(W) < MIN_MUESTRAS_T:
            return None
        t = np.median(X, axis=0) if t0 is None else np.asarray(t0, float)
        for _ in range(8):
            r = np.linalg.norm(X - t, axis=1)
            esc = max(RES_T_ESCALA, float(np.median(r)) / 1.54)    # ruido por eje (robusto)
            wr = W / (1.0 + (r / (2.0 * esc)) ** 2)
            t = (X * wr[:, None]).sum(axis=0) / wr.sum()
        r = np.linalg.norm(X - t, axis=1)
        ruido = max(RES_T_ESCALA, float(np.median(r)) / 1.54)
        dentro = r < max(RES_T_INLIER, 3.5 * ruido)
        inliers = float(W[dentro].sum() / W.sum())
        mediana = float(np.median(r[dentro])) if dentro.any() else 9.9
        error = ruido * np.sqrt(3.0 / max(int(dentro.sum()), 1))
        ok = inliers >= MIN_INLIERS and error <= ERROR_T_OK and ruido <= RUIDO_T_MAX
        return dict(t=t, ok=ok, n=len(W), mediana=mediana, inliers=inliers, ruido=ruido, error=error)

    def residuos_recientes(self, R, t, desde):
        X, W = self._estimaciones(R, desde)
        if X is None:
            return 0.0, 0
        return float(np.median(np.linalg.norm(X - t, axis=1))), len(W)


# ================================================================== extrinsecas (ChArUco)
EXTRINSECAS_ARCHIVO = "camaras_extrinsecas.json"
motivo_sin_extrinsecas = None     # por que no se cargaron en la ultima llamada a cargar_extrinsecas()


def ruta_extrinsecas(ruta=None):
    """Por defecto, junto a este archivo (donde lo guarda calibrar_extrinseca.py)."""
    ruta = ruta or EXTRINSECAS_ARCHIVO
    return ruta if os.path.isabs(ruta) else os.path.join(os.path.dirname(os.path.abspath(__file__)), ruta)


def cargar_extrinsecas(mxid_a=None, mxid_b=None, ruta=None):
    """Lee camaras_extrinsecas.json (lo genera calibrar_extrinseca.py) y devuelve
    (R_ab, t_ab, info) con  p_A = R_ab @ p_B + t_ab,  o None si no hay archivo o no
    corresponde a estas camaras. Si se abren en orden contrario al de la calibracion
    (A<->B) se invierte la transformacion."""
    global motivo_sin_extrinsecas
    ruta = ruta_extrinsecas(ruta)
    motivo_sin_extrinsecas = None
    if not os.path.exists(ruta):
        motivo_sin_extrinsecas = f"no existe {os.path.basename(ruta)}"
        print(f"[fusion] No existe {os.path.basename(ruta)}: la camara B solo entrara en la fusion "
              f"cuando se autocalibre sola (lento y poco preciso). Ejecuta  python calibrar_extrinseca.py")
        return None
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    R = np.array(d["R_ab"], dtype=float).reshape(3, 3)
    t = np.array(d["t_ab"], dtype=float).reshape(3)
    fa, fb = d.get("mxid_a"), d.get("mxid_b")
    if mxid_a and mxid_b and fa and fb:
        if (str(mxid_a), str(mxid_b)) == (str(fa), str(fb)):
            pass
        elif (str(mxid_a), str(mxid_b)) == (str(fb), str(fa)):
            print("[fusion] Camaras abiertas en orden inverso al de la calibracion: se invierte la transformacion")
            R, t = R.T, -R.T @ t
        else:
            motivo_sin_extrinsecas = f"{os.path.basename(ruta)} es de otras camaras"
            print(f"[fusion] Las camaras abiertas ({mxid_a} / {mxid_b}) NO coinciden con las de "
                  f"{os.path.basename(ruta)} ({fa} / {fb}): se ignora. Recalibra con calibrar_extrinseca.py")
            return None
    if abs(np.linalg.det(R) - 1.0) > 1e-3 or np.linalg.norm(R @ R.T - np.eye(3)) > 1e-3:
        motivo_sin_extrinsecas = f"{os.path.basename(ruta)} no es valido"
        print(f"[fusion] {os.path.basename(ruta)}: R_ab no es una rotacion valida, se ignora")
        return None
    return R, t, d


# ================================================================== fusion
class FusionCamaras:
    def __init__(self, hilos, reloj=time.monotonic, extrinsecas=None):
        """extrinsecas: (R_ab, t_ab) de la camara B (hilos[1]) respecto a la A (hilos[0]),
        con  p_A = R_ab @ p_B + t_ab  (calibrar_extrinseca.py). Con ellas B entra en la
        fusion desde el primer fotograma, ya situada en 3D, sin esperar a autocalibrarse."""
        self.hilos = list(hilos)                 # hilos[0] = camara principal
        self.reloj = reloj
        N = len(self.hilos)
        self.imus = [_imu_de(h) for h in self.hilos]
        self.imu_ok = [imu is not None and getattr(imu, "R", None) is not None for imu in self.imus]
        # pose de cada camara en W
        self.R_op = [R_CAM_DEFECTO.copy() for _ in range(N)]
        self.t = [np.zeros(3) for _ in range(N)]
        self.valida = [i == 0 for i in range(N)]     # rotacion conocida
        self.desalineada = [False] * N               # calibrada pero sus medidas no cuadran: apartada
        self.R_ref = [None] * N                      # calibracion de referencia (c o aceptada)
        self._corregidos = [None] * N                # ultimo resultado de cada camara ya corregido
        self.t_valida = [i == 0 for i in range(N)]   # posicion conocida (A = origen al arrancar)
        self.R_F = R_CAM_DEFECTO.copy()
        self.R = [np.eye(3) for _ in range(N)]
        self._cache_manos = None                     # (clave, manos fundidas) del ultimo fotograma
        self._actualizar_R()
        self.calibrada_c = False
        self._inclinacion_hecha = False
        self._fija = [False] * N                     # calibracion entre camaras de archivo (ChArUco) vigente
        if extrinsecas is not None and N >= 2:
            self._cargar_extrinsecas(*extrinsecas[:2])
        # calibracion automatica
        self.solvers = [SolucionadorRotacion() for _ in range(N)]
        self.solvers_t = [SolucionadorTraslacion() for _ in range(N)]
        self.info_solver = [None] * N
        self.info_t = [None] * N
        self._t_solver = -1e9
        self._mov_vistos = [getattr(imu, "movimientos", 0) if imu is not None else 0 for imu in self.imus]
        self._arriba_ref = [None] * N
        self._hist = [dict() for _ in range(N)]     # (lado, hueso) -> (dir, t)
        self._vel = [dict() for _ in range(N)]      # (lado, hueso) -> deg/s
        self._hist_p = [dict() for _ in range(N)]   # indice -> historia del punto (uv, z, t, p, vel)
        self._n_par = [-1] * N
        # combinacion
        self._visto = [dict() for _ in range(N)]    # lado -> {brazo, t, t_inicio}
        self._t_uso = [None] * N                    # desde cuando entra la camara en la fusion 3D
        self._P_previo = {}                         # (lado, indice) -> (punto en W, t)
        self._kalman = {}                           # (lado, indice) -> KalmanPunto
        self.cruces = FiltroCruces(len(self.hilos))
        self._n_cruces = [-1] * len(self.hilos)
        self._kalman_n = {}                         # (lado, indice, camara) -> ultimo frame usado
        self._kalman_cams = {}                      # (lado, indice) -> {camara: t de su ultima medida}
        self._alfa_puntos = {}                      # lado -> 0 (direcciones) .. 1 (puntos)
        self._ult_pts = {}                          # lado -> (ultimas direcciones por puntos, instante)
        self._t_ultimo = None
        self.n = 0
        self._n_vistos = [-1] * N
        self._mano_cam = {}
        self._hist_mano = {}                      # (camara, lado) -> (px, t, n) para adelantar pixeles
        self._codo_previo = {}                    # lado -> ultimo codo fusionado en W (para reconstruirlo si se tapa)
        self._previo, self._t_previo = {}, -1e9
        self._avisos = {}
        self._t_excluida = [None] * N             # desde cuando la excluye su IMU
        self._verificar_t = [None] * N            # desplazamiento (IMU) por comprobar con medidas: desde cuando
        self._t_pre = [None] * N                  # posicion segun el cuerpo justo antes de ese aviso
        self._ultimos = [None] * N                # ultimo resultado de cada camara (diagnostico)

    # ------------------------------------------------------------ estado
    @property
    def error(self):
        return next((h.error for h in self.hilos if h.error), None)

    @property
    def fps(self):
        return min(h.fps for h in self.hilos)

    @property
    def R_cam(self):
        """Marco de salida -> operador: lo que el teleop usa como R_cam."""
        return self.R_F

    @property
    def calibrada(self):
        return all(self.valida)

    def _actualizar_R(self):
        self.R = [self.R_F.T @ Ro for Ro in self.R_op]
        self._cache_manos = None

    def _aviso(self, texto, duracion=DURACION_AVISO):
        print(f"[fusion] {texto}")
        self._avisos[texto] = self.reloj() + duracion

    def avisos(self):
        ahora = self.reloj()
        self._avisos = {t: f for t, f in self._avisos.items() if f > ahora}
        return list(self._avisos)

    def _girando(self, i):
        return self.imus[i] is not None and self.imus[i].girando()

    def _moviendose(self, i):
        """Girando o desplazandose de forma apreciable (segun su IMU)."""
        imu = self.imus[i]
        if imu is None:
            return False
        despl = getattr(imu, "desplazandose", None)
        return imu.girando() or (callable(despl) and despl())

    def _usables(self, vivos, ahora):
        """Camaras calibradas que se pueden usar. Las que su IMU da por moviendose se
        dejan fuera, PERO la IMU nunca puede dejar la fusion sin camaras: si todas
        estan "moviendose", se usan igual (mejor un dato regular que ninguno)."""
        validas = [i for i in vivos if self.valida[i] and not self.desalineada[i]]
        quietas = [i for i in validas if not self._moviendose(i)]
        for i in range(len(self.hilos)):
            if i in validas and i not in quietas:
                if self._t_excluida[i] is None:
                    self._t_excluida[i] = ahora
                elif ahora - self._t_excluida[i] > T_AVISO_IMU:
                    self._aviso(f"Camara {ETIQUETAS[i]}: su IMU dice que se esta moviendo. Si esta "
                                "quieta: python camaras.py --imu, o arranca con --sin-imu", 3.0)
                    self._t_excluida[i] = ahora
            else:
                self._t_excluida[i] = None
        return quietas if quietas else validas

    def corregido(self, i):
        """Ultimo resultado de la camara i con izquierda/derecha ya corregidos."""
        return self._corregidos[i]

    def diagnostico(self):
        """Una linea por camara: por que se usa o no. Para la consola."""
        ahora = self.reloj()
        lineas = []
        for i, h in enumerate(self.hilos):
            s = self._ultimos[i]
            letra = ETIQUETAS[i]
            if h.error:
                lineas.append(f"  {letra}: ERROR en el hilo: {h.error}")
                continue
            if s is None:
                lineas.append(f"  {letra}: aun no ha llegado ningun resultado")
                continue
            retraso = (_t_disp(s) - s["t"]) * 1000
            edad = (ahora - _t_disp(s)) * 1000
            ve = "+".join(sorted(s["brazos"])) or "NINGUNO"
            imu = self.imus[i]
            if imu is None:
                e_imu = "sin IMU"
            elif self._moviendose(i):
                e_imu = "MOVIENDOSE"
            else:
                e_imu = "quieta" if getattr(imu, "_listo", True) else "midiendo sesgo"
            motivo = []
            if edad > EDAD_MAX * 1000:
                motivo.append("dato demasiado viejo")
            if not self.valida[i]:
                motivo.append("sin calibrar")
            lineas.append(f"  {letra}: {h.fps:4.1f} fps | captura->resultado {retraso:4.0f} ms | edad {edad:4.0f} ms"
                          f" | MediaPipe ve brazos: {ve} | IMU: {e_imu}"
                          + (f" | NO SE USA: {', '.join(motivo)}" if motivo else ""))
        return "\n".join(lineas)

    def _arriba(self, i):
        """Gravedad de la camara i en su marco, si su IMU es fiable."""
        return self.imus[i].arriba() if self.imu_ok[i] else None

    def _ancla(self):
        """Camara de referencia de rotacion: la principal si esta bien; si no, la primera valida."""
        cand = [i for i in range(len(self.hilos)) if self.valida[i] and not self._moviendose(i)]
        return cand[0] if cand else None

    def _ancla_t(self):
        cand = [i for i in range(len(self.hilos))
                if self.valida[i] and self.t_valida[i] and not self._moviendose(i)]
        return cand[0] if cand else None

    def angulo_con_principal(self, i):
        """Angulo (grados) entre el eje optico de la camara i y el de la principal."""
        return angulo_deg(self.R_op[i][:, 2], self.R_op[0][:, 2])

    def distancia_con_principal(self, i):
        if not (self.t_valida[i] and self.t_valida[0]):
            return None
        return float(np.linalg.norm(self.t[i] - self.t[0]))

    def estado_texto(self):
        varias = len(self.hilos) > 1
        partes = []
        for i in range(len(self.hilos)):
            info = self.info_solver[i]
            if self._moviendose(i):
                e = "moviendose"
            elif self.valida[i] and self.desalineada[i]:
                e = "NO CUADRA: pulsa c"
            elif self.valida[i]:
                if not varias:
                    e = "ok"
                elif self.t_valida[i]:
                    e = "ok 3D (ChArUco)" if self._fija[i] else "ok 3D"
                else:
                    e = f"ok, situando ({len(self.solvers_t[i])})"
            elif self._ancla() is None:
                e = "sin referencia: pulsa c"
            else:
                e = f"calibrando {len(self.solvers[i])} pares"
                if info is not None:
                    e += f", {info['mediana']:.0f} deg"
                    if info["variedad"] < VARIEDAD_MIN[info["modo"]]:
                        e += ", mueve mas los brazos"
            partes.append(f"{ETIQUETAS[i]} {e}")
        return " | ".join(partes)

    # ------------------------------------------------------------ calibracion con [c]
    def fijar_calibracion(self, R_cam, angulos_imu=None):
        """R_cam[i]: rotacion camara i -> operador obtenida con la postura (calibrar()),
        o None si esa camara no la consiguio. Reglas:
          - Si varias camaras ya calibradas entre si ven la postura y COINCIDEN (a menos de
            DISCREPANCIA_C_MAX), se conserva su relacion y solo se gira el conjunto.
          - Si NO coinciden, la relacion que habia esta mal: cada camara toma su propia
            postura (nunca se promedian orientaciones que no cuadran).
          - Si solo una camara calibrada ve la postura, se gira el conjunto con ella.
          - Las que no estaban calibradas y ven la postura, la toman directamente.
        Devuelve dict(ok, conservada, discrepancia_deg o None, directas[letras])."""
        N = len(self.hilos)
        ok = [i for i, R in enumerate(R_cam) if R is not None]
        if not ok:
            return dict(ok=False, conservada=False, discrepancia=None, directas=[])
        for i in ok:
            if angulos_imu is not None and angulos_imu[i] is not None and angulos_imu[i] > MAX_DIF_IMU \
                    and self.imu_ok[i]:
                self.imu_ok[i] = False
                self._aviso(f"IMU de la camara {ETIQUETAS[i]}: gravedad a {angulos_imu[i]:.0f} deg de los "
                            "brazos; no se usara (python camaras.py --imu para revisarla)", 10.0)
        vigentes = [i for i in range(N) if self.valida[i] and not self.desalineada[i] and not self._moviendose(i)]
        con_relacion = [i for i in ok if i in vigentes]
        discrepancia, directas, conservada = None, [], False
        G = None
        if con_relacion:
            Gs = [np.asarray(R_cam[i], float) @ self.R_op[i].T for i in con_relacion]
            G = media_rotaciones(Gs)
            if len(Gs) > 1:
                discrepancia = max(angulo_rotacion_deg(Gs[a] @ Gs[b].T)
                                   for a in range(len(Gs)) for b in range(a + 1, len(Gs)))
                if discrepancia > DISCREPANCIA_C_MAX:
                    if any(self._fija[i] for i in con_relacion):
                        # la relacion viene de ChArUco (mucho mas precisa que una postura): se conserva
                        self._aviso(f"Al calibrar con c, las posturas de las camaras discrepan {discrepancia:.0f} deg; "
                                    "se conserva la calibracion ChArUco entre ellas", 8.0)
                    else:
                        G = None    # la relacion entre ellas estaba mal: nada de promediar
        if G is not None:
            conservada = True
            for i in vigentes:
                self.R_op[i] = G @ self.R_op[i]
                self.t[i] = G @ self.t[i]
                self.R_ref[i] = self.R_op[i].copy()
                self.solvers[i].transformar(G)
                self.solvers_t[i].transformar(G)
                self._arriba_ref[i] = self._arriba(i)
            self._P_previo = {k: (G @ P, tt) for k, (P, tt) in self._P_previo.items()}
            for i in ok:
                if i not in vigentes:
                    directas.append(ETIQUETAS[i])
                    self._fijar_directa(i, R_cam[i])
        else:
            # Cada camara con su postura; las que no la vieron, sin calibrar (se calibraran solas).
            for i in range(N):
                if R_cam[i] is not None:
                    directas.append(ETIQUETAS[i])
                    self._fijar_directa(i, R_cam[i])
                elif self.valida[i]:
                    self.valida[i] = self.t_valida[i] = False
                    self.solvers[i].vaciar()
                    self.solvers_t[i].vaciar()
            self.t[ok[0]], self.t_valida[ok[0]] = np.zeros(3), True
            if discrepancia is not None:
                self._aviso(f"Al calibrar, las camaras discrepaban {discrepancia:.0f} deg: se ha calibrado "
                            "cada una por separado con tu postura", 8.0)
        self._kalman = {}
        for i in ok:
            self.desalineada[i] = False
        ancla = self._ancla()
        self.R_F = self.R_op[ancla if ancla is not None else ok[0]].copy()
        self.calibrada_c = self._inclinacion_hecha = True
        self._actualizar_R()
        self._mano_cam.clear()
        self._previo = {}
        return dict(ok=True, conservada=conservada, discrepancia=discrepancia, directas=directas)

    def _cargar_extrinsecas(self, R_ab, t_ab):
        """Situa la camara B respecto a la A con la calibracion ChArUco."""
        R_ab = np.asarray(R_ab, dtype=float).reshape(3, 3)
        t_ab = np.asarray(t_ab, dtype=float).reshape(3)
        self.R_op[1] = self.R_op[0] @ R_ab
        self.t[1] = self.R_op[0] @ t_ab
        self.valida[1] = self.t_valida[1] = True
        self.desalineada[1] = False
        self.R_ref[1] = self.R_op[1].copy()
        self._fija[1] = True
        self._actualizar_R()
        giro = angulo_rotacion_deg(R_ab)
        print(f"[fusion] Extrinsecas ChArUco cargadas: B a {np.linalg.norm(t_ab):.2f} m de A, "
              f"girada {giro:.0f} grados. Las dos camaras entran en la fusion desde el primer fotograma.")

    def _fijar_directa(self, i, R):
        self._fija = [False] * len(self.hilos)
        self.R_op[i] = np.array(R, dtype=float)
        self.R_ref[i] = self.R_op[i].copy()
        self.desalineada[i] = False
        self.valida[i] = True
        self.t_valida[i] = False
        self.solvers[i].vaciar()
        self.solvers_t[i].vaciar()
        self.info_solver[i] = self.info_t[i] = None
        self._visto[i] = {}
        self._arriba_ref[i] = self._arriba(i)

    # ------------------------------------------------------------ IMU
    def _inclinacion_inicial(self):
        """Antes de calibrar con [c]: si la principal tiene IMU, el marco por defecto se
        corrige con su inclinacion real (camara mirando hacia abajo, ladeada...)."""
        if self._inclinacion_hecha or self.calibrada_c:
            return
        arriba = self._arriba(0)
        if arriba is None:
            return
        R0_ant = self.R_op[0].copy()
        self.R_op[0] = marco_por_defecto(arriba)
        G = self.R_op[0] @ R0_ant.T          # giro del marco W: se aplica a las camaras ya situadas respecto a A
        for i in range(1, len(self.hilos)):
            if self._fija[i]:
                self.R_op[i] = G @ self.R_op[i]
                self.t[i] = G @ self.t[i]
                self.R_ref[i] = self.R_op[i].copy()
        self.R_F = self.R_op[0].copy()
        self._arriba_ref[0] = arriba
        self._actualizar_R()
        self._inclinacion_hecha = True
        print(f"[fusion] inclinacion de la camara A con la IMU: arriba = {np.round(arriba, 2)}")

    def _revisar_imus(self):
        for i, imu in enumerate(self.imus):
            if imu is None or imu.movimientos == self._mov_vistos[i]:
                continue
            self._mov_vistos[i] = imu.movimientos
            mov = getattr(imu, "ultimo_mov", None)
            self._camara_movida(i, "IMU", rotacion=True if mov is None else bool(mov.get("rotacion", True)))

    def _camara_movida(self, i, motivo, rotacion=True):
        """La camara i se ha movido. Si solo se ha desplazado (sin girar), su rotacion
        sigue valiendo y solo hay que volver a situarla.

        (demo oct-2026) Un desplazamiento SIN giro solo lo dice el acelerometro integrado
        dos veces, que con vibraciones da "desplazada 5-8 cm" sin que nadie la toque (se vio
        en la demo, y con eso se tiraba la calibracion ChArUco). Si la camara estaba situada
        y hay otra con la que compararla, sigue en uso y se COMPRUEBA con las medidas del
        cuerpo (_resolver_t): solo si su posicion ya no cuadra se resitua."""
        letra = ETIQUETAS[i]
        N = len(self.hilos)
        otras_t = [j for j in range(N) if j != i and self.valida[j] and self.t_valida[j]
                   and not self._moviendose(j)]
        if not rotacion and self.valida[i] and self.t_valida[i] and otras_t:
            # posicion segun el CUERPO justo antes del aviso: se compara con la de despues
            # (las dos con el mismo sesgo del modelo de piel, que asi se cancela)
            r_pre = self.solvers_t[i].resolver(self.R_op[i], self.t[i]) if len(self.solvers_t[i]) else None
            self._t_pre[i] = r_pre["t"] if (r_pre is not None and r_pre["ok"]) else None
            self._verificar_t[i] = self.reloj()
            self.solvers_t[i].vaciar()           # solo cuentan las medidas de despues
            self.info_t[i] = None
            mov = getattr(self.imus[i], "ultimo_mov", None) or {}
            cm = f" ~{mov['despl'] * 100:.0f} cm" if mov.get("despl") is not None else ""
            self._aviso(f"Camara {letra}: su IMU dice que se ha desplazado{cm}; sigue en uso y "
                        f"lo compruebo con tus medidas")
            return
        if any(self._fija):
            self._aviso("Se ha movido una camara: la calibracion ChArUco ya no vale (vuelvo a la automatica). "
                        "Cuando puedas: python calibrar_extrinseca.py", 10.0)
        self._fija = [False] * N
        self.solvers_t[i].vaciar()
        self.info_t[i] = None
        self._verificar_t[i] = None
        self._hist_p[i] = {}
        self._t_uso[i] = None
        if not rotacion:
            if otras_t:
                self.t_valida[i] = False
                self._aviso(f"Camara {letra} desplazada ({motivo}): se vuelve a situar sola")
            else:
                self.t[i], self.t_valida[i] = np.zeros(3), True    # nada con que comparar: nuevo origen
            return
        self.solvers[i].vaciar()
        self.info_solver[i] = None
        self._visto[i] = {}
        self._mano_cam = {l: c for l, c in self._mano_cam.items() if c != i}
        otras = [j for j in range(N) if j != i and self.valida[j] and not self._moviendose(j)]
        if otras:
            self.valida[i] = self.t_valida[i] = False
            self._aviso(f"Camara {letra} movida ({motivo}): se recalibra sola con "
                        f"{'+'.join(ETIQUETAS[j] for j in otras)}; mueve los brazos")
            return
        # Ninguna otra camara puede sostener el marco: se sigue con esta, corrigiendo
        # la inclinacion con la gravedad (el giro alrededor de la vertical se desconoce).
        arriba = self._arriba(i)
        if not self.calibrada_c and i == 0:
            if arriba is not None:
                self.R_op[0] = marco_por_defecto(arriba)
        elif arriba is not None and self._arriba_ref[i] is not None:
            self.R_op[i] = self.R_op[i] @ alinear(arriba, self._arriba_ref[i])
        self._arriba_ref[i] = arriba
        self.valida[i] = True
        self.t[i], self.t_valida[i] = np.zeros(3), True
        self._actualizar_R()
        self._aviso(f"Camara {letra} movida ({motivo}) y no hay otra calibrada: pulsa c", 10.0)

    # ------------------------------------------------------------ historia de cada camara
    def _actualizar_historia(self, snaps, nuevos):
        """Con cada frame nuevo: velocidad de cada hueso y de cada punto (para no
        emparejar cosas que se mueven deprisa) y el frame anterior de cada punto
        (para adelantar una camara al instante de otra)."""
        for k in nuevos:
            sk = snaps[k]
            vel, hist = {}, {}
            for lado, b in sk["brazos"].items():
                for h in HUESOS:
                    prev = self._hist[k].get((lado, h))
                    dt = sk["t"] - prev[1] if prev is not None else 1.0
                    vel[(lado, h)] = (angulo_deg(b[h], prev[0]) / dt) if 1e-3 < dt < 0.25 else np.inf
                    hist[(lado, h)] = (b[h], sk["t"])
            self._hist[k], self._vel[k] = hist, vel
            pts, K = sk.get("puntos"), sk.get("K")
            if not pts or K is None:
                self._hist_p[k] = {}
                continue
            nuevo = {}
            for j, pt in pts.items():
                if j not in RADIO:
                    continue
                prev = self._hist_p[k].get(j)
                p = punto_camara(pt, K, j) if pt["vis"] >= VIS_PUNTO else None
                e = dict(uv=np.array(pt["uv"], float), z=pt["z"], t=sk["t"], p=p,
                         uv0=None, z0=None, t0=None, vel=np.inf)
                if prev is not None and 1e-3 < sk["t"] - prev["t"] < 0.15:
                    e.update(uv0=prev["uv"], z0=prev["z"], t0=prev["t"])
                    if p is not None and prev["p"] is not None:
                        e["vel"] = float(np.linalg.norm(p - prev["p"]) / (sk["t"] - prev["t"]))
                nuevo[j] = e
            self._hist_p[k] = nuevo

    # ------------------------------------------------------------ calibracion automatica
    def _emparejar(self, snaps, ancla, nuevos):
        """Con cada frame nuevo de una camara, busca otra calibrada que haya capturado
        casi a la vez y guarda parejas de direcciones (rotacion) y de puntos (posicion)."""
        N = len(self.hilos)
        ancla_t = self._ancla_t()
        for k in nuevos:
            sk = snaps[k]
            if self._moviendose(k):
                continue
            refs = [j for j in range(N) if j != k and self.valida[j] and not self.desalineada[j]
                    and not self._moviendose(j)
                    and snaps[j] is not None and abs(snaps[j]["t"] - sk["t"]) < DT_PAR]
            if not refs:
                continue
            j = ancla if ancla in refs else refs[0]
            self._pares_direcciones(k, j, sk, snaps[j])
            refs_t = [j for j in refs if self.t_valida[j]]
            if self.valida[k] and refs_t:
                jt = ancla_t if ancla_t in refs_t else refs_t[0]
                self._muestras_posicion(k, jt, sk, snaps[jt])

    def _pares_direcciones(self, k, j, sk, sj):
        mapa = {"L": "L", "R": "R"}
        if not self.valida[k]:
            # Sin calibrar no se puede corregir un cruce izq/der con las direcciones: se
            # decide con rasgos que no dependen de la camara. Si no se puede saber, nada:
            # una pareja cruzada estropea la calibracion mas de lo que ayuda una buena.
            cruce = cruce_invariante(sj, sk)
            if cruce is None:
                return
            if cruce:
                mapa = {"L": "R", "R": "L"}
        for lado, bk in sk["brazos"].items():
            bj = sj["brazos"].get(mapa[lado])
            if bj is None or "codo tapado" in bk["fuente"] or "codo tapado" in bj["fuente"]:
                continue
            for h in HUESOS:
                v = max(self._vel[k].get((lado, h), np.inf), self._vel[j].get((mapa[lado], h), np.inf))
                if v > VEL_MAX_PAR:
                    continue
                w = min(calidad_hueso(bk, bk[h]), calidad_hueso(bj, bj[h])) * (1 - v / VEL_MAX_PAR)
                if w >= PESO_MIN_PAR:
                    self.solvers[k].anadir(sk["t"], self.R_op[j] @ bj[h], bk[h], w)

    def _muestras_posicion(self, k, j, sk, sj):
        pk, pj = sk.get("puntos") or {}, sj.get("puntos") or {}
        for idx in PUNTOS_3D:
            ek, ej = self._hist_p[k].get(idx), self._hist_p[j].get(idx)
            if ek is None or ej is None or ek["p"] is None or ej["p"] is None:
                continue
            if pk[idx]["fuente"] != "oak" or pj[idx]["fuente"] != "oak":
                continue
            vis = min(pk[idx]["vis"], pj[idx]["vis"])
            v = max(self._velocidad_punto(k, idx), self._velocidad_punto(j, idx))
            if vis < VIS_MUESTRA_T or v > VEL_MAX_T:
                continue
            P = self.R_op[j] @ ej["p"] + self.t[j]
            self.solvers_t[k].anadir(sk["t"], P, ek["p"], vis * (1 - v / VEL_MAX_T))

    def _velocidad_punto(self, k, idx):
        """Velocidad aproximada (m/s) de un punto segun la camara k. No se deriva de los
        puntos (la profundidad es ruidosa y la dispararia) sino de lo deprisa que giran
        los huesos, que vienen filtrados: codo = brazo; muneca = brazo + antebrazo.
        Hombros y caderas se mueven poco (solo si el operador se desplaza)."""
        lado = LADO_PUNTO.get(idx)
        if lado is None:
            return 0.0
        vb = np.radians(self._vel[k].get((lado, "dir_brazo"), np.inf)) * L_BRAZO_TIPICO
        if idx in (13, 14):
            return vb
        return vb + np.radians(self._vel[k].get((lado, "dir_antebrazo"), np.inf)) * L_ANTEBRAZO_TIPICO

    def _arriba_operador(self):
        """'Arriba' en W segun las camaras calibradas con IMU."""
        vs = []
        for j in range(len(self.hilos)):
            a = self._arriba(j)
            if self.valida[j] and a is not None:
                vs.append(self.R_op[j] @ a)
        return _unitario(np.sum(vs, axis=0)) if vs else None

    def _resolver(self, ahora, ancla):
        cambiado = False
        arriba_op = self._arriba_operador()
        for k in range(len(self.hilos)):
            s = self.solvers[k]
            s.purgar(ahora)
            imu = self.imus[k]
            if self.valida[k] and imu is not None and not imu.en_movimiento() \
                    and imu.movimientos == self._mov_vistos[k]:
                a = self._arriba(k)
                if a is not None:
                    self._arriba_ref[k] = a
            if k == ancla or self._moviendose(k) or len(s) < MIN_PARES:
                continue
            arriba_k = self._arriba(k)
            r = s.resolver(self.R_op[k] if self.valida[k] else None,
                           arriba_op if arriba_k is not None else None, arriba_k)
            self.info_solver[k] = r
            if r is None:
                continue
            letra = ETIQUETAS[k]
            if r.get("imu_incoherente") is not None and r["ok"] and self.imu_ok[k]:
                self.imu_ok[k] = False
                self._aviso(f"IMU de la camara {letra}: su gravedad no cuadra con tus brazos "
                            f"({r['imu_incoherente']:.0f} deg); se calibra sin ella (python camaras.py --imu)", 10.0)
            if not self.valida[k]:
                if r["ok"]:
                    self.R_op[k], self.valida[k] = r["R"], True
                    self.R_ref[k] = r["R"].copy()
                    self.desalineada[k] = False
                    self._visto[k] = {}
                    cambiado = True
                    otra = 1 if k == 0 else 0
                    ang = angulo_deg(self.R_op[k][:, 2], self.R_op[otra][:, 2])
                    self._aviso(f"Camara {letra} calibrada sola ({r['modo']}, {r['n']} pares, "
                                f"residuo {r['mediana']:.1f} deg, a {ang:.0f} deg de {ETIQUETAS[otra]})")
                continue
            # Ya valida: siguen cuadrando sus medidas?
            med, cnt = s.residuos_recientes(self.R_op[k], ahora - VENTANA_RECIENTE)
            con_imu = self.imus[k] is not None and self.imus[ancla] is not None
            if con_imu and cnt >= MIN_PARES_MOVIDA:
                # Con IMU, si la camara se moviera lo diria la IMU: si no cuadra es que su
                # calibracion esta mal. No se recoloca sola (asi empezo el desastre del video):
                # se aparta de la fusion y se avisa; vuelve sola si las medidas vuelven a cuadrar.
                if not self.desalineada[k] and med > RES_DESALINEADA:
                    self.desalineada[k] = True
                    self._visto[k] = {}
                    self._aviso(f"Camara {letra}: sus medidas no cuadran con {ETIQUETAS[ancla]} ({med:.0f} deg): "
                                "se aparta. Colocate y pulsa c", 10.0)
                elif self.desalineada[k] and med < RES_REALINEADA:
                    self.desalineada[k] = False
                    self._aviso(f"Camara {letra}: vuelve a cuadrar ({med:.0f} deg), se usa otra vez")
                if self.desalineada[k]:
                    continue
            elif cnt >= MIN_PARES_MOVIDA and med > RES_MOVIDA:
                if self._fija[k]:
                    # calibrada con ChArUco: unas medidas que no cuadran (oclusiones, izq/der) no
                    # bastan para tirar una calibracion de 1 grado; si la camara se mueve lo dice la IMU
                    self._aviso(f"Camara {letra}: sus medidas no cuadran con {ETIQUETAS[ancla]} ({med:.0f} deg) "
                                "pero se conserva la calibracion ChArUco (si la has movido, recalibra)", 10.0)
                    s.vaciar(antes_de=ahora - VENTANA_RECIENTE)
                    continue
                s.vaciar(antes_de=ahora - VENTANA_RECIENTE)
                self.valida[k] = self.t_valida[k] = False
                self._fija = [False] * len(self.hilos)
                self.solvers_t[k].vaciar()
                self._visto[k] = {}
                self._aviso(f"Camara {letra}: ya no cuadra con {ETIQUETAS[ancla]} ({med:.0f} deg). "
                            f"Se recalibra sola suponiendo que la movida es la {letra}; "
                            f"si has movido la {ETIQUETAS[ancla]}, pulsa c", 10.0)
                continue
            if r["ok"] and not self._fija[k] and angulo_rotacion_deg(r["R"] @ self.R_op[k].T) > REFINO_MIN_DEG:
                ref = self.R_ref[k] if self.R_ref[k] is not None else self.R_op[k]
                if angulo_rotacion_deg(r["R"] @ ref.T) <= MAX_DERIVA:     # afinado acotado
                    self.R_op[k] = interpolar_rot(self.R_op[k], r["R"], REFINO)
                    cambiado = True
        if cambiado:
            self._actualizar_R()

    def _resolver_t(self, ahora):
        N = len(self.hilos)
        ancla_t = self._ancla_t()
        if ancla_t is None:      # ninguna camara situada: la primera calibrada hace de origen
            cand = [i for i in range(N) if self.valida[i] and not self._moviendose(i)]
            if not cand:
                return
            ancla_t = cand[0]
            self.t[ancla_t], self.t_valida[ancla_t] = np.zeros(3), True
        for k in range(N):
            s = self.solvers_t[k]
            s.purgar(ahora)
            if k == ancla_t or not self.valida[k] or self._moviendose(k) or len(s) < MIN_MUESTRAS_T:
                continue
            r = s.resolver(self.R_op[k], self.t[k] if self.t_valida[k] else None)
            self.info_t[k] = r
            if r is None:
                continue
            letra = ETIQUETAS[k]
            if not self.t_valida[k]:
                if r["ok"]:
                    self.t[k], self.t_valida[k] = r["t"], True
                    self._t_uso[k] = None
                    d = float(np.linalg.norm(self.t[k] - self.t[ancla_t]))
                    self._aviso(f"Camara {letra} situada: a {d:.2f} m de {ETIQUETAS[ancla_t]} "
                                f"({r['n']} muestras, error estimado {r['error'] * 100:.1f} cm)")
                continue
            if self._verificar_t[k] is not None:
                # la IMU dijo "desplazada": el solver solo tiene medidas de despues
                if not r["ok"]:
                    if ahora - self._verificar_t[k] > T_VERIFICAR_MAX:
                        self._verificar_t[k] = None
                        self._aviso(f"Camara {letra}: no he podido comprobar si se ha desplazado "
                                    "(ponte delante de las dos camaras)")
                    continue
                ref = self._t_pre[k] if self._t_pre[k] is not None else self.t[k]
                umbral = DESPL_CONFIRMADO if self._t_pre[k] is not None else DESPL_CONFIRMADO_SIN_REF
                delta = r["t"] - ref
                d = float(np.linalg.norm(delta))
                self._verificar_t[k] = None
                if d > max(umbral, 3.0 * r["error"]):
                    # se corrige SOLO el desplazamiento medido: la rotacion ChArUco sigue valiendo
                    # (no ha girado) y la posicion del cuerpo tiene su propio sesgo de unos cm
                    self.t[k] = self.t[k] + delta
                    self._kalman = {}
                    self._aviso(f"Camara {letra}: confirmado, se ha desplazado {d * 100:.0f} cm; "
                                "resituada con tus medidas (cuando puedas: python calibrar_extrinseca.py)", 10.0)
                else:
                    self._aviso(f"Camara {letra}: sigue en su sitio ({d * 100:.1f} cm); falsa alarma de la IMU")
                continue
            med, cnt = s.residuos_recientes(self.R_op[k], self.t[k], ahora - VENTANA_RECIENTE)
            if cnt >= MIN_PARES_MOVIDA and med > max(RES_T_MOVIDA, 2.5 * r["mediana"]):
                s.vaciar(antes_de=ahora - VENTANA_RECIENTE)
                if self._fija[k]:      # ChArUco: se conserva (ver _resolver)
                    continue
                self.t_valida[k] = False
                self._fija = [False] * N
                self._aviso(f"Camara {letra}: su posicion ya no cuadra ({med * 100:.0f} cm): "
                            "se vuelve a situar sola")
                continue
            if r["ok"] and not self._fija[k]:
                self.t[k] = self.t[k] + REFINO_T * (r["t"] - self.t[k])

    # ------------------------------------------------------------ fusion
    def ultimo(self):
        ahora = self.reloj()
        dt = 0.0 if self._t_ultimo is None else float(np.clip(ahora - self._t_ultimo, 0.0, 0.1))
        self._t_ultimo = ahora
        N = len(self.hilos)
        snaps = [h.ultimo() for h in self.hilos]
        if all(s is None for s in snaps):
            return None
        self._inclinacion_inicial()
        self._revisar_imus()
        ancla = self._ancla()

        self._ultimos = [s if s is not None else u for s, u in zip(snaps, self._ultimos)]
        vivos = [i for i, s in enumerate(snaps) if s is not None and ahora - _t_disp(s) < EDAD_MAX]
        usar = self._usables(vivos, ahora)

        if any(s is not None and s["n"] != self._n_vistos[i] for i, s in enumerate(snaps)):
            self.n += 1
            self._n_vistos = [s["n"] if s is not None else -1 for s in snaps]

        previo = self._previo if ahora - self._t_previo < EDAD_PREVIO else {}
        nuevos_c = [k for k, s in enumerate(snaps) if s is not None and s["n"] != self._n_cruces[k]]
        for k in nuevos_c:
            self._n_cruces[k] = snaps[k]["n"]
        cruzar = self.cruces.actualizar(snaps, nuevos_c, [i for i in usar], self.R)
        cruzadas = []
        for i, c in enumerate(cruzar):
            if c and snaps[i] is not None:
                snaps[i] = _cruzar(snaps[i])
                cruzadas.append(ETIQUETAS[i])
        self._corregidos = list(snaps)

        nuevos = [k for k, s in enumerate(snaps) if s is not None and s["n"] != self._n_par[k]]
        for k in nuevos:
            self._n_par[k] = snaps[k]["n"]
        self._actualizar_historia(snaps, nuevos)

        # Calibracion automatica (rotacion y posicion de cada camara)
        if N > 1:
            self._emparejar(snaps, ancla, nuevos)
            if ahora - self._t_solver > PERIODO_SOLVER:
                self._t_solver = ahora
                self._resolver(ahora, ancla)
                self._resolver_t(ahora)
                usar = self._usables(vivos, ahora)
        base = usar[0] if usar else next(i for i, s in enumerate(snaps) if s is not None)

        # Modo direcciones: ultima medida de cada brazo en cada camara (se desvanece)
        for i in range(N):
            if i not in usar:
                self._visto[i] = {}
                continue
            t = snaps[i]["t"]
            for lado, b in snaps[i]["brazos"].items():
                e = self._visto[i].get(lado)
                if e is not None and t <= e["t"]:
                    continue
                t_ini = t if (e is None or t - e["t"] > HUECO_RAMPA) else e["t_inicio"]
                self._visto[i][lado] = {"brazo": b, "t": t, "t_disp": _t_disp(snaps[i]), "t_inicio": t_ini}

        # Modo puntos: camaras situadas con puntos
        usar_t = [i for i in usar if self.t_valida[i] and snaps[i].get("puntos")
                  and snaps[i].get("K") is not None] if N > 1 else []
        for i in range(N):
            if i not in usar_t:
                self._t_uso[i] = None
            elif self._t_uso[i] is None:
                self._t_uso[i] = ahora
        # Mientras alguna camara en uso no este SITUADA, solo por direcciones (la fusion 3D
        # necesita saber donde esta cada camara). Antes se exigia ademas que TODAS las camaras
        # tuviesen puntos en ese mismo fotograma: si una perdia la deteccion un instante, todo
        # el modo cambiaba a direcciones de golpe y volvia despues -> saltos del robot.
        # Ahora una camara situada que no ve nada en este fotograma simplemente no aporta.
        sin_situar = [i for i in usar if not self.t_valida[i]] if N > 1 else []
        modo_puntos_ok = bool(usar_t) and not sin_situar

        brazos, desacuerdo, modos = {}, {}, {}
        for lado in ("L", "R"):
            r_dir, des = None, 0.0
            cand = []
            for i in usar:
                e = self._visto[i].get(lado)
                if e is not None and ahora - e["t_disp"] < EDAD_MAX:
                    rampa = max(0.02, _suave((ahora - e["t_inicio"]) / T_RAMPA))
                    cand.append((i, e["brazo"], ahora - e["t_disp"], rampa))
            if cand:
                r_dir, des = self._fundir_brazo(cand, previo.get(lado))
            r_pts = self._brazo_puntos(lado, snaps, usar_t, ahora) if modo_puntos_ok else None
            a = self._alfa_puntos.get(lado, 0.0)
            if r_pts is not None:
                self._ult_pts[lado] = (r_pts, ahora)
            elif lado in self._ult_pts and a > 0.0 and r_dir is not None \
                    and ahora - self._ult_pts[lado][1] < T_MODO:
                # un fotograma sin puntos 3D: se sigue un momento con los ultimos (en lugar de
                # saltar de golpe a direcciones) y se va pasando a direcciones poco a poco
                r_pts = self._ult_pts[lado][0]
                a = max(0.0, a - dt / T_MODO)
                self._alfa_puntos[lado] = a
                sal = {h: _unitario(a * r_pts[h] + (1 - a) * r_dir[h]) for h in HUESOS}
                sal["fuente"] = r_dir["fuente"]
                sal["vis"] = max(r_pts["vis"], r_dir["vis"])
                brazos[lado], desacuerdo[lado], modos[lado] = sal, des, a
                continue
            if r_pts is None:
                a, sal = 0.0, r_dir
            elif r_dir is None:
                a, sal = 1.0, r_pts
            else:
                a = min(1.0, a + dt / T_MODO)
                sal = {h: _unitario(a * r_pts[h] + (1 - a) * r_dir[h]) for h in HUESOS}
                sal["fuente"] = r_pts["fuente"] if a >= 0.5 else r_dir["fuente"]
                sal["vis"] = max(r_pts["vis"], r_dir["vis"])
            self._alfa_puntos[lado] = a
            if sal is not None:
                brazos[lado], desacuerdo[lado], modos[lado] = sal, des, a
        if brazos:
            self._previo = {**previo, **brazos}
            self._t_previo = ahora

        # Las manos solo dependen de los fotogramas (no del instante): mientras no llegue uno
        # nuevo se reutiliza el resultado. Antes se volvian a triangular las dos manos en cada
        # ciclo del robot (60 Hz con camaras a 30): ~40 % del tiempo de la fusion.
        clave_manos = (self.n, tuple(cruzar), tuple(usar))
        if self._cache_manos is not None and self._cache_manos[0] == clave_manos:
            aperturas = self._cache_manos[1]
        else:
            aperturas = {}
            for lado in ("L", "R"):
                cand = [(i, snaps[i]["aperturas"][lado]) for i in usar if lado in snaps[i]["aperturas"]]
                if cand:
                    aperturas[lado] = self._fundir_mano(lado, cand, snaps, ahora)
            self._cache_manos = (clave_manos, aperturas)

        res = dict(snaps[base])
        res["n"] = self.n
        # "t" = cuando estuvo disponible (lo que el teleop mira para saber si es reciente);
        # "t_captura" = cuando se capturo.
        res["t"] = max(_t_disp(snaps[i]) for i in usar) if usar else _t_disp(snaps[base])
        res["t_captura"] = max(snaps[i]["t"] for i in usar) if usar else snaps[base]["t"]
        res["brazos"], res["aperturas"] = brazos, aperturas
        res["fusion"] = {"n_camaras": N, "calibrada": self.calibrada,
                         "activas": [ETIQUETAS[i] for i in usar], "desacuerdo": desacuerdo,
                         "modo": "3D" if modos and max(modos.values()) >= 0.5 else "direcciones",
                         "cruzadas": cruzadas, "estado": self.estado_texto(), "avisos": self.avisos()}
        return res

    # ------------------------------------------------------------ modo direcciones
    def _fundir_brazo(self, cand, previo=None):
        """cand: [(camara, {"dir_brazo", "dir_antebrazo", "fuente", "vis"}, edad, rampa)]."""
        salida, desacuerdos = {}, []
        for clave in HUESOS:
            vs, ws, tapado = [], [], []
            for i, b, edad, rampa in cand:
                d = b[clave]
                vs.append(_unitario(self.R[i] @ d))
                ws.append(calidad_hueso(b, d) * peso_edad(edad) * rampa)
                tapado.append("codo tapado" in b["fuente"])
            # si alguna vista ve el codo, las del codo tapado se apagan (gradualmente)
            w_buenas = sum(w for w, t in zip(ws, tapado) if not t)
            f_tapado = 1.0 - _suave(w_buenas / W_REF_CODO)
            ws = [w * (f_tapado if t else 1.0) for w, t in zip(ws, tapado)]
            salida[clave], des = combinar(vs, ws, None if previo is None else previo[clave])
            desacuerdos.append(des)
        if len(self.hilos) == 1:
            salida["fuente"] = cand[0][1]["fuente"]
        elif len(cand) > 1:
            salida["fuente"] = "fusion " + "+".join(ETIQUETAS[i] for i, *_ in cand)
        else:
            salida["fuente"] = f"cam {ETIQUETAS[cand[0][0]]}: {cand[0][1]['fuente']}"
        salida["vis"] = max(b.get("vis", 1.0) for _, b, *_ in cand)
        return salida, max(desacuerdos)

    # ------------------------------------------------------------ modo puntos 3D
    def _extrapolar(self, i, j, pt, t_snap, t_ref):
        """Pixel y profundidad del punto j de la camara i adelantados de t_snap a t_ref."""
        uv, z = np.array(pt["uv"], float), pt["z"]
        dt = min(max(t_ref - t_snap, 0.0), MAX_EXTRAP)
        h = self._hist_p[i].get(j)
        if dt > 0 and h is not None and h["t0"] is not None and abs(h["t"] - t_snap) < 1e-9:
            d = h["t"] - h["t0"]
            uv = uv + FACTOR_EXTRAP * (uv - h["uv0"]) / d * dt
            if z is not None and h["z0"] is not None:
                z = z + FACTOR_EXTRAP * (z - h["z0"]) / d * dt
        return uv, z

    def _gaussiana(self, i, j, uv, z, fuente, K, peso):
        """Lo que dice la camara i del punto j, en W: media m y matriz de informacion L
        (inversa de la covarianza): estrecha de lado, alargada a lo largo del rayo."""
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        r = _unitario(np.array([(uv[0] - cx) / fx, (uv[1] - cy) / fy, 1.0]))
        d = self.R_op[i] @ r
        if z is None:
            dist, s_rayo = 2.0, SIGMA_RAYO
        else:
            dist = z / r[2] + RADIO[j]
            s_rayo = (SIGMA_OAK[0] + SIGMA_OAK[1] * z * z) if fuente == "oak" else SIGMA_PRED
        s_lado = max(dist, 0.3) * SIGMA_PX / fx
        D = np.outer(d, d)
        L0 = (np.eye(3) - D) / s_lado ** 2 + D / s_rayo ** 2
        return self.t[i] + d * dist, L0 * peso, L0, z is not None

    def _fundir_punto(self, gs, previo, ahora):
        """gs: [(camara, m, L, L_sin_peso, tiene_profundidad)]. Devuelve (punto, camaras)."""
        if not gs:
            return None, set()

        def fundir(sub):
            if np.linalg.eigvalsh(sum(g[3] for g in sub))[0] < 1.0 / SIGMA_MAX_PUNTO ** 2:
                return None
            L = sum(g[2] for g in sub)
            return np.linalg.solve(L, sum(g[2] @ g[1] for g in sub))

        P = fundir(gs)
        if P is not None and len(gs) > 1:
            chi2 = [float((P - g[1]) @ g[2] @ (P - g[1])) for g in gs]
            if max(chi2) > CHI2_MAX:
                # No cuadran: se queda la camara (con profundidad) que mejor sigue al punto anterior
                solos = [(g, fundir([g])) for g in gs if g[4]]
                solos = [(g, p) for g, p in solos if p is not None]
                if not solos:
                    return None, set()
                if previo is not None and ahora - previo[1] < EDAD_PREVIO:
                    g, P = min(solos, key=lambda x: np.linalg.norm(x[1] - previo[0]))
                else:
                    g, P = max(solos, key=lambda x: np.trace(x[0][2]))
                gs = [g]
        if P is None:
            return None, set()
        return P, {g[0] for g in gs}

    def _rampa_cam(self, i, ahora):
        t0 = self._t_uso[i]
        return 0.02 if t0 is None else max(0.02, _suave((ahora - t0) / T_RAMPA))

    def _punto_kalman(self, lado, j, snaps, cams, ahora):
        """Actualiza el Kalman de la articulacion j con las medidas NUEVAS de cada camara
        (cada una en su instante de captura, con su incertidumbre) y devuelve su posicion
        en W, o None. Una medida que no cuadra (izq/der cambiados, deteccion mala) se
        descarta; si se descartan varias seguidas, el filtro se reinicia."""
        clave = (lado, j)
        kf = self._kalman.get(clave)
        medidas = []
        for i in cams:
            pt = snaps[i]["puntos"].get(j)
            if pt is None or pt["vis"] < VIS_PUNTO or self._kalman_n.get((lado, j, i)) == snaps[i]["n"]:
                continue
            self._kalman_n[(lado, j, i)] = snaps[i]["n"]
            peso = max(1e-3, _suave((pt["vis"] - VIS_PUNTO) / (VIS_PLENA - VIS_PUNTO))) * self._rampa_cam(i, ahora)
            # calidad del punto segun su camara (codo con longitud imposible = inventado): la
            # informacion que aporta se reduce con el CUADRADO (equivale a multiplicar su sigma)
            peso *= max(0.05, float(pt.get("calidad", 1.0))) ** 2
            m, L, L0, con_z = self._gaussiana(i, j, np.array(pt["uv"], float), pt["z"], pt["fuente"],
                                             snaps[i]["K"], peso)
            medidas.append((snaps[i]["t"], i, m, L, con_z))
        for t_m, i, m, L, con_z in sorted(medidas, key=lambda x: x[0]):
            Rm = np.linalg.inv(L + np.eye(3) * 1e-9) + np.eye(3) * 1e-6
            if kf is None or t_m - kf.t > T_KALMAN_MAX or kf.rechazos >= 4:
                if not con_z:          # solo con el rayo no se sabe donde empezar
                    continue
                C = Rm.copy()
                w, V = np.linalg.eigh(C)
                C = V @ np.diag(np.minimum(w, 0.3 ** 2)) @ V.T
                kf = self._kalman[clave] = KalmanPunto(m, C, t_m, Q_KALMAN[j])
                self._kalman_cams[clave] = {i: ahora}
                continue
            kf.predecir(t_m)
            if kf.actualizar(m, Rm):
                self._kalman_cams.setdefault(clave, {})[i] = ahora
        if kf is None or ahora - max(self._kalman_cams.get(clave, {0: -1e9}).values()) > EDAD_MAX:
            return None
        return kf.x[:3].copy()

    def _brazo_puntos(self, lado, snaps, usar_t, ahora):
        """Direcciones de brazo y antebrazo a partir de los puntos 3D (filtrados con
        Kalman) de hombro, codo y muneca. None si no hay hombro y muneca, o si no es
        plausible."""
        cams = [i for i in usar_t if ahora - _t_disp(snaps[i]) < EDAD_MAX]
        if not cams:
            return None
        P, usadas, vis = {}, set(), 0.0
        for j in PUNTOS_BRAZO[lado]:
            P[j] = self._punto_kalman(lado, j, snaps, cams, ahora)
            if P[j] is not None:
                usadas |= {i for i, t in self._kalman_cams.get((lado, j), {}).items() if ahora - t < EDAD_MAX}
                vis = max([vis] + [snaps[i]["puntos"][j]["vis"] for i in cams if j in snaps[i]["puntos"]])
        hombro, codo, muneca = (P[j] for j in PUNTOS_BRAZO[lado])
        if hombro is None or muneca is None:
            return None
        codo_visto = codo is not None and any(snaps[i]["puntos"].get(PUNTOS_BRAZO[lado][1], {}).get("vis", 0) >= VIS_PUNTO
                                              for i in cams)
        tapado = False
        if codo_visto:
            lb, la = np.linalg.norm(codo - hombro), np.linalg.norm(muneca - codo)
            if not (RANGO_BRAZO[0] < lb < RANGO_BRAZO[1] and RANGO_ANTEBRAZO[0] < la < RANGO_ANTEBRAZO[1]):
                codo_visto = False      # codo imposible (inventado): se reconstruye como si no se viera
        if not codo_visto:
            # nadie ve el codo: en vez de dar el brazo por recto (saltos de 20-30 cm en la muneca
            # del robot), se reconstruye con las longitudes del operador y el ultimo codo conocido
            if not RANGO_HOMBRO_MUNECA[0] < np.linalg.norm(muneca - hombro) < RANGO_HOMBRO_MUNECA[1]:
                return None
            from seguimiento_brazos import LONGITUDES
            fuera = np.array([0.0, 1.0 if lado == "L" else -1.0, 0.0])    # W: y hacia la izquierda del operador
            codo, tapado = brazo_geometria.codo_en_circulo(hombro, muneca, LONGITUDES.brazo, LONGITUDES.antebrazo,
                                                           self._codo_previo.get(lado), abajo=[0.0, 0.0, 1.0], fuera=fuera)
        self._codo_previo[lado] = codo
        d_b, d_a = _unitario(codo - hombro), _unitario(muneca - codo)
        if d_b is None or d_a is None:
            return None
        salida = {"dir_brazo": _unitario(self.R_F.T @ d_b), "dir_antebrazo": _unitario(self.R_F.T @ d_a)}
        salida["fuente"] = "3D " + "+".join(ETIQUETAS[i] for i in sorted(usadas)) + (" codo reconstruido" if tapado else "")
        salida["vis"] = vis
        return salida

    # ------------------------------------------------------------ manos
    def _actualizar_hist_mano(self, i, lado, datos, snap):
        """Guarda los pixeles de la mano de la camara i para poder adelantarlos (velocidad)."""
        px = datos.get("px")
        if px is None:
            return None, None
        clave = (i, lado)
        prev = self._hist_mano.get(clave)
        if prev is None or prev[2] != snap["n"]:
            self._hist_mano[clave] = (np.asarray(px, float), snap["t"], snap["n"],
                                      None if prev is None else prev[0], None if prev is None else prev[1])
        h = self._hist_mano[clave]
        return h[3], h[4]      # px y t del fotograma ANTERIOR

    def _fundir_mano(self, lado, cand, snaps, ahora):
        """Una mano a partir de lo que ven las camaras que la detectan.
          1) Si dos (o mas) camaras SITUADAS la ven casi a la vez: TRIANGULACION de los 21
             landmarks -> orientacion y puntos 3D metricos reales ("3D A+B").
          2) Si no, la camara que mejor la ve (tamano x calidad, con histeresis): puntos
             "2.5D" (pixeles + profundidad OAK-D de la palma + profundidad relativa del
             modelo) o, sin profundidad, el marco y los world landmarks del modelo.
        La apertura (0..1) viene siempre de la mejor camara: depende de la vista.
        El marco y los puntos se dan en el marco de SALIDA (R_F), como las direcciones."""
        puntuacion = {}
        for i, m in cand:
            puntuacion[i] = tamano_mano(snaps[i].get("manos"), lado) * (0.3 + 0.7 * float(m.get("calidad", 1.0)))
        mejor = max(puntuacion, key=puntuacion.get)
        actual = self._mano_cam.get(lado)
        if actual in puntuacion and puntuacion[actual] * HISTERESIS_MANO >= puntuacion[mejor]:
            mejor = actual
        self._mano_cam[lado] = mejor
        datos = dict(next(m for i, m in cand if i == mejor))
        datos["cam"] = ETIQUETAS[mejor]
        datos["tam_px"] = tamano_mano(snaps[mejor].get("manos"), lado)
        datos["bruto"] = next((m["bruto"] for m in snaps[mejor].get("manos") or []
                               if m["lado"] == lado), None)
        datos.pop("px", None)

        # --- 1) triangulacion entre camaras situadas ---
        vistas, usadas = [], []
        con_px = [(i, m) for i, m in cand if m.get("px") is not None and snaps[i].get("K") is not None
                  and self.t_valida[i] and self.valida[i]]
        if len(con_px) >= 2:
            t_ref = max(snaps[i]["t"] for i, _ in con_px)
            for i, m in con_px:
                if t_ref - snaps[i]["t"] > DT_MANOS:
                    continue
                px_prev, t_prev = self._actualizar_hist_mano(i, lado, m, snaps[i])
                px = manos3d.extrapolar_px(m["px"], px_prev, snaps[i]["t"], t_prev, t_ref)
                vistas.append((px, snaps[i]["K"], self.R_op[i], self.t[i]))
                usadas.append(i)
        else:
            for i, m in con_px:
                self._actualizar_hist_mano(i, lado, m, snaps[i])
        if len(vistas) >= 2:
            tri = manos3d.triangular(vistas)
            if tri is not None and manos3d.triangulacion_valida(*tri):
                P_w = tri[0]
                R_w = manos3d.marco_de_puntos(P_w)
                if R_w is not None:
                    P_out = manos3d.centrar(P_w) @ self.R_F            # W -> marco de salida (R_F^T p)
                    datos["marco"] = self.R_F.T @ R_w
                    datos["mundo"] = manos3d.a_puntos(P_out)
                    datos["puntos3d"] = P_w
                    datos["fuente_mano"] = "3D " + "+".join(ETIQUETAS[i] for i in sorted(usadas))
                    datos["reproy_px"] = float(max(tri[1]))
                    return datos

        # --- 2) una camara: 2.5D si hay profundidad de la palma; si no, el modelo ---
        m = next(m for i, m in cand if i == mejor)
        R = self.R[mejor]
        K = snaps[mejor].get("K")
        if m.get("px") is not None and m.get("z_palma") is not None and K is not None and m.get("mundo") is not None:
            P_cam = manos3d.levantar_25d(m["px"], K, m["z_palma"], m["mundo"])
            R_m = manos3d.marco_de_puntos(P_cam)
            if R_m is not None:
                P_out = manos3d.centrar(P_cam) @ R.T
                datos["marco"] = R @ R_m
                datos["mundo"] = manos3d.a_puntos(P_out)
                datos["fuente_mano"] = f"2.5D {ETIQUETAS[mejor]}"
                return datos
        if not np.allclose(R, np.eye(3), atol=1e-9):
            if datos.get("marco") is not None:
                datos["marco"] = R @ datos["marco"]
            if datos.get("mundo") is not None:
                datos["mundo"] = _rotar_landmarks(datos["mundo"], R)
        datos["fuente_mano"] = f"modelo {ETIQUETAS[mejor]}"
        return datos
