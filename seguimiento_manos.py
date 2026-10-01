"""
Deteccion de mano abierta / puno cerrado con MediaPipe Hand Landmarker (API Tasks).

Para cada mano devuelve una APERTURA continua de 0 (puno) a 1 (palma abierta),
la ORIENTACION de la mano como matriz 3x3 en el marco de la camara, y el pixel
de la muneca, que sirve para emparejar cada mano con su brazo.

Como se mide: con los puntos 3D metricos de la mano (hand_world_landmarks) se
calcula, para indice, corazon, anular y menique, la distancia de la punta a su
nudillo dividida por el tamano de la palma. Con el dedo estirado esa razon vale
cerca de 0.9; con el dedo doblado la punta vuelve casi al nudillo y baja a 0.3.
Al ser un cociente, no depende de la distancia a la camara ni del tamano de la
mano. El pulgar se ignora porque es el mas ruidoso.
"""
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")

MUNECA = 0
NUDILLO_INDICE, NUDILLO_MENIQUE = 5, 17
PALMA = (0, 9)                  # muneca -> nudillo del corazon: escala de la mano
DEDOS = ((5, 8), (9, 12), (13, 16), (17, 20))   # (nudillo, punta)

# Umbrales de la razon punta-nudillo / palma. La ventana muestra el valor bruto,
# asi que se pueden afinar mirando la pantalla con la mano abierta y cerrada.
RAZON_CERRADA = 0.40
RAZON_ABIERTA = 0.85

ZONA_MUERTA = 0.12              # pega la apertura a 0 o a 1 cerca de los extremos
DIST_MAX_MUNECA = 0.20          # fraccion del ancho de imagen para emparejar mano y brazo
FILTRO_MIN_CUTOFF = 1.0         # Hz
FILTRO_BETA = 0.5

CONEXIONES_MANO = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                   (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                   (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


class FiltroApertura:
    """One Euro escalar: quieto cuando la mano esta quieta, rapido al abrir o cerrar."""

    def __init__(self, min_cutoff=FILTRO_MIN_CUTOFF, beta=FILTRO_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alfa(corte, dt):
        tau = 1.0 / (2 * np.pi * corte)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-3)
        a_d = self._alfa(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alfa(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


class SeguidorManos:
    def __init__(self, num_manos=2):
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
        if not os.path.exists(ruta):
            print(f"Descargando el modelo de manos en {ruta} ...")
            urllib.request.urlretrieve(URL_MODELO, ruta)
        opciones = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=ruta),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_manos,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(opciones)
        self.filtros = {"L": FiltroApertura(), "R": FiltroApertura()}
        self.t_ms = 0
        self.manos = []   # [{lado, apertura, bruto, puntos_px}] del ultimo frame, para dibujar

    # ---------------------------------------------------------------- medida
    @staticmethod
    def razon_apertura(mundo):
        """Media de (punta-nudillo)/palma en los cuatro dedos largos. None si no vale."""
        p = np.array([[l.x, l.y, l.z] for l in mundo])
        palma = np.linalg.norm(p[PALMA[1]] - p[PALMA[0]])
        if palma < 1e-4:
            return None
        return float(np.mean([np.linalg.norm(p[t] - p[n]) for n, t in DEDOS]) / palma)

    @staticmethod
    def marco(mundo):
        """Orientacion de la mano como matriz 3x3 (columnas = ejes de la mano
        expresados en el marco de la camara):
           columna 0: hacia los dedos (muneca -> nudillo del corazon)
           columna 1: a lo ancho de la palma (indice -> menique)
           columna 2: normal de la palma
        Los hand_world_landmarks vienen ya orientados como la imagen, asi que
        esta matriz esta en el mismo marco que las direcciones de los brazos."""
        p = np.array([[l.x, l.y, l.z] for l in mundo])
        f = p[PALMA[1]] - p[PALMA[0]]
        a = p[NUDILLO_MENIQUE] - p[NUDILLO_INDICE]
        nf = np.linalg.norm(f)
        if nf < 1e-5:
            return None
        f = f / nf
        a = a - f * np.dot(a, f)          # ortogonaliza respecto a los dedos
        na = np.linalg.norm(a)
        if na < 1e-5:
            return None
        a = a / na
        return np.column_stack([f, a, np.cross(f, a)])

    @staticmethod
    def a_apertura(razon):
        a = (razon - RAZON_CERRADA) / (RAZON_ABIERTA - RAZON_CERRADA)
        a = float(np.clip(a, 0.0, 1.0))
        if a < ZONA_MUERTA:          # asegura puno del todo cerrado
            return 0.0
        if a > 1.0 - ZONA_MUERTA:    # y palma del todo abierta
            return 1.0
        return (a - ZONA_MUERTA) / (1.0 - 2 * ZONA_MUERTA)

    # ---------------------------------------------------------------- ciclo
    def procesar(self, bgr, lm2d_pose):
        """Devuelve {"L": {"apertura": .., "marco": R}, ...} en lados del OPERADOR.
        Cada mano se asigna al brazo cuya muneca de pose le queda mas cerca en la
        imagen, que es mas fiable que la etiqueta izquierda/derecha de MediaPipe."""
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.t_ms = max(self.t_ms + 1, int(time.monotonic() * 1000))
        res = self.detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.t_ms)

        self.manos = []
        if not res.hand_landmarks or not res.hand_world_landmarks:
            return {}

        t = time.monotonic()
        # munecas del cuerpo: 15 = izquierda del operador, 16 = derecha
        ref = {}
        if lm2d_pose is not None:
            for lado, i in (("L", 15), ("R", 16)):
                if lm2d_pose[i][2] >= 0.5:
                    ref[lado] = np.array(lm2d_pose[i][:2])

        candidatos = []
        for lm, mundo in zip(res.hand_landmarks, res.hand_world_landmarks):
            razon = self.razon_apertura(mundo)
            R = self.marco(mundo)
            if razon is None or R is None:
                continue
            px = np.array([[l.x * w, l.y * h] for l in lm])
            lado, dist = None, np.inf
            for s, p in ref.items():
                d = float(np.linalg.norm(px[MUNECA] - p))
                if d < dist:
                    lado, dist = s, d
            if lado is None or dist > DIST_MAX_MUNECA * w:
                continue
            candidatos.append((dist, lado, razon, px, R, mundo))

        datos = {}
        for dist, lado, razon, px, R, mundo in sorted(candidatos, key=lambda c: c[0]):
            if lado in datos:
                continue
            apertura = self.filtros[lado](self.a_apertura(razon), t)
            datos[lado] = {"apertura": apertura, "marco": R, "mundo": mundo}
            self.manos.append({"lado": lado, "apertura": apertura,
                               "bruto": razon, "puntos_px": px})
        return datos


def dibujar_manos(img, manos):
    """Esqueleto de la mano: verde cuanto mas abierta, rojo cuanto mas cerrada."""
    for m in manos:
        px, a = m["puntos_px"], m["apertura"]
        color = (int(60 * (1 - a)), int(70 + 160 * a), int(235 - 175 * a))  # BGR
        for i, j in CONEXIONES_MANO:
            cv2.line(img, tuple(px[i].astype(int)), tuple(px[j].astype(int)), color, 2, cv2.LINE_AA)
        for p in px:
            cv2.circle(img, tuple(p.astype(int)), 3, color, -1, cv2.LINE_AA)