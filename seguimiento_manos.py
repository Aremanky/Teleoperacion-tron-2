"""
Manos del operador con MediaPipe Hand Landmarker (API Tasks), sobre RECORTES.

QUE CAMBIA (oct-2026) Y POR QUE
  Antes Hand Landmarker recibia el fotograma ENTERO (640x480). A 2 m una mano son
  40-60 px, y MediaPipe reescala la imagen a ~224 px para buscar las palmas: la mano
  acababa midiendo ~15 px. De ahi los dedos con errores de centimetros, las palmas
  "de canto" y la profundidad relativa aplastada, que es lo que arruinaba la
  orientacion de la muneca y la imitacion de los dedos.

  Ahora MediaPipe Pose dice DONDE esta cada mano (muneca, indice, menique y pulgar
  del cuerpo) y Hand Landmarker se ejecuta sobre un recorte cuadrado alrededor de
  ella, reescalado a RECORTE_PX. Si la camara entrega ademas un fotograma HD
  (CamaraOAK(hd=True)), el recorte sale de el: 2-4 veces mas pixeles de mano.
  Cada lado (L/R) tiene su propio detector en modo VIDEO (asi el seguimiento entre
  fotogramas no se confunde con dos recortes distintos) y la izquierda/derecha la
  decide la Pose, no la etiqueta de MediaPipe.

  Los hand_world_landmarks vienen en un marco alineado con el RAYO del recorte, no
  con el eje optico: se giran con la rotacion que lleva el eje optico a ese rayo
  (hasta 30 grados en los bordes de la imagen). Con K desconocida no se corrige.

Para cada mano devuelve:
  apertura   0 (puno) .. 1 (palma abierta), filtrada y con rango adaptativo
  marco      orientacion 3x3 (columnas: dedos, ancho de la palma indice->menique,
             normal) en el marco de la camara, de los world landmarks corregidos
  mundo      los 21 world landmarks corregidos (objetos con .x .y .z, metros)
  px         21x2 pixeles en la imagen de 640x480 (para triangular entre camaras)
  z_palma    profundidad OAK-D (m) del centro de la palma, o None
  tam_px     diagonal de la mano en la imagen (px), para elegir la mejor camara
  calidad    0..1: lo grande que sale la mano en el recorte y la confianza de MediaPipe
"""
import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from suavizado import FiltroOneEuro

URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")

MUNECA = 0
NUDILLO_INDICE, NUDILLO_MENIQUE = 5, 17
PALMA = (0, 9)                  # muneca -> nudillo del corazon: escala de la mano
PALMA_PUNTOS = (0, 5, 9, 13, 17)
DEDOS = ((5, 8), (9, 12), (13, 16), (17, 20))   # (nudillo, punta)

# Umbrales de la razon punta-nudillo / palma (la ventana muestra el valor bruto).
RAZON_CERRADA = 0.40
RAZON_ABIERTA = 0.85
HISTORIA_RAZON = 1800           # muestras (~1 min a 30 fps) para el rango adaptativo
MIN_MUESTRAS_ADAPT = 60
CERRADA_MAX = 0.62
SPAN_MIN = 0.25
ABIERTA_MIN = 0.75
ZONA_MUERTA = 0.12              # pega la apertura a 0 o a 1 cerca de los extremos
FILTRO_MIN_CUTOFF = 1.0         # Hz (apertura)
FILTRO_BETA = 0.5

# Pose del cuerpo: indices de muneca, indice, menique y pulgar de cada lado (33 landmarks)
POSE_MANO = {"L": (15, 19, 17, 21), "R": (16, 20, 18, 22)}
POSE_CODO = {"L": 13, "R": 14}
VIS_MIN_POSE = 0.35

# Recorte
RECORTE_PX = 256                # lado al que se reescala el recorte antes de MediaPipe
RECORTE_FACTOR = 2.6            # lado del recorte = FACTOR x extension de la mano en la imagen
RECORTE_ANTEBRAZO = 1.0         # ... o x la longitud del antebrazo en la imagen (lo mayor)
RECORTE_MIN = 0.12              # lado minimo, fraccion del alto de la imagen
RECORTE_MAX = 0.60              # lado maximo
RECORTE_PREVIO = 1.8            # con la mano vista hace poco, lado >= PREVIO x su diagonal
T_PREVIO = 0.3                  # s: cuanto vale la deteccion anterior para centrar el recorte
MANO_PX_PLENA = 160.0           # px de mano (en el recorte original) a partir de los que calidad = 1
Z_PALMA_TOL = 0.30              # m: la profundidad de la palma debe cuadrar con la muneca de la Pose

CONEXIONES_MANO = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                   (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                   (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


class _Punto:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


def rotacion_al_rayo(K, u, v):
    """Rotacion minima que lleva el eje optico (0,0,1) al rayo del pixel (u, v)."""
    r = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    r /= np.linalg.norm(r)
    z = np.array([0.0, 0.0, 1.0])
    eje = np.cross(z, r)
    s, c = np.linalg.norm(eje), float(np.dot(z, r))
    if s < 1e-9:
        return np.eye(3)
    k = eje / s
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)


def _ruta_modelo():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    if not os.path.exists(ruta):
        print(f"Descargando el modelo de manos en {ruta} ...")
        urllib.request.urlretrieve(URL_MODELO, ruta)
    return ruta


def _crear_detector(ruta, num_manos):
    opciones = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=ruta),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=num_manos,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(opciones)


class SeguidorManos:
    def __init__(self, num_manos=2, recorte=True, recorte_px=RECORTE_PX):
        ruta = _ruta_modelo()
        self.recorte = recorte
        self.recorte_px = int(recorte_px)
        # un detector por lado (recortes) + uno de respaldo para la imagen completa
        self.detectores = {lado: _crear_detector(ruta, 1) for lado in ("L", "R")} if recorte else {}
        self.detector_completo = _crear_detector(ruta, num_manos)
        self.filtros = {"L": FiltroOneEuro(FILTRO_MIN_CUTOFF, FILTRO_BETA), "R": FiltroOneEuro(FILTRO_MIN_CUTOFF, FILTRO_BETA)}
        self.historia = {"L": deque(maxlen=HISTORIA_RAZON), "R": deque(maxlen=HISTORIA_RAZON)}
        self.t_ms = {"L": 0, "R": 0, "completo": 0}
        self.previa = {}      # lado -> (centro px, diagonal px, t) de la ultima deteccion
        self.manos = []       # [{lado, apertura, bruto, puntos_px, recorte}] del ultimo frame, para dibujar
        self.ultimo_recorte = {}   # lado -> imagen del recorte (depuracion)

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
        """Orientacion de la mano como matriz 3x3 (columnas = ejes de la mano en el marco
        de la camara): 0 hacia los dedos (muneca -> nudillo del corazon), 1 a lo ancho de la
        palma (indice -> menique), 2 su producto vectorial (normal de la palma en la mano
        derecha; dorso en la izquierda, que se compara con el mismo marco del robot)."""
        p = np.array([[l.x, l.y, l.z] for l in mundo])
        f = p[PALMA[1]] - p[PALMA[0]]
        a = p[NUDILLO_MENIQUE] - p[NUDILLO_INDICE]
        nf = np.linalg.norm(f)
        if nf < 1e-5:
            return None
        f = f / nf
        a = a - f * np.dot(a, f)
        na = np.linalg.norm(a)
        if na < 1e-5:
            return None
        a = a / na
        return np.column_stack([f, a, np.cross(f, a)])

    def rango(self, lado):
        """(razon de puno cerrado, razon de mano abierta) para esa mano."""
        h = self.historia[lado]
        if len(h) < MIN_MUESTRAS_ADAPT:
            return RAZON_CERRADA, RAZON_ABIERTA
        p5, p95 = np.percentile(h, (5, 95))
        cerrada = float(np.clip(p5, RAZON_CERRADA, CERRADA_MAX))
        abierta = float(max(p95, cerrada + SPAN_MIN, ABIERTA_MIN))
        return cerrada, abierta

    @staticmethod
    def a_apertura(razon, cerrada=RAZON_CERRADA, abierta=RAZON_ABIERTA):
        a = (razon - cerrada) / (abierta - cerrada)
        a = float(np.clip(a, 0.0, 1.0))
        if a < ZONA_MUERTA:
            return 0.0
        if a > 1.0 - ZONA_MUERTA:
            return 1.0
        return (a - ZONA_MUERTA) / (1.0 - 2 * ZONA_MUERTA)

    # ---------------------------------------------------------------- recorte
    def _ventana(self, lado, lm2d, w, h, t):
        """(centro (u, v), lado) del recorte cuadrado de esa mano, o None si no se sabe
        donde esta."""
        if lm2d is None:
            return None
        i_mun, i_ind, i_men, i_pul = POSE_MANO[lado]
        if lm2d[i_mun][2] < VIS_MIN_POSE:
            return None
        pts = [np.array(lm2d[i][:2]) for i in (i_mun, i_ind, i_men, i_pul) if lm2d[i][2] >= VIS_MIN_POSE]
        P = np.array(pts)
        # centro: entre la muneca y los nudillos (la mano ocupa mas alla de la muneca)
        centro = P.mean(axis=0) if len(P) > 1 else P[0]
        extension = 0.0
        for a in range(len(P)):
            for b in range(a + 1, len(P)):
                extension = max(extension, float(np.linalg.norm(P[a] - P[b])))
        lado_rec = RECORTE_FACTOR * extension
        i_codo = POSE_CODO[lado]
        if lm2d[i_codo][2] >= VIS_MIN_POSE:
            antebrazo = float(np.linalg.norm(np.array(lm2d[i_codo][:2]) - np.array(lm2d[i_mun][:2])))
            lado_rec = max(lado_rec, RECORTE_ANTEBRAZO * antebrazo)
        prev = self.previa.get(lado)
        if prev is not None and t - prev[2] < T_PREVIO:
            lado_rec = max(lado_rec, RECORTE_PREVIO * prev[1])
            centro = 0.5 * (centro + prev[0])
        lado_rec = float(np.clip(lado_rec, RECORTE_MIN * h, RECORTE_MAX * h))
        return centro, lado_rec

    @staticmethod
    def _recortar(img, centro, lado_rec, escala, salida):
        """Recorte cuadrado de 'img' (que puede ser la HD: 'escala' px por px de la imagen
        base) centrado en 'centro' (px de la imagen base), reescalado a salida x salida.
        Devuelve (recorte, (u0, v0, lado) en px de la imagen base)."""
        H, W = img.shape[:2]
        c = np.asarray(centro, float) * escala
        L = lado_rec * escala
        u0, v0 = c[0] - L / 2, c[1] - L / 2
        # matriz afin: recorte -> imagen (asi no hay que rellenar bordes a mano)
        s = L / salida
        M = np.array([[1.0 / s, 0.0, -u0 / s], [0.0, 1.0 / s, -v0 / s]])
        rec = cv2.warpAffine(img, M, (salida, salida), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return rec, (u0 / escala, v0 / escala, lado_rec)

    # ---------------------------------------------------------------- profundidad
    @staticmethod
    def _z_palma(depth, px, z_ref=None, radio=4):
        if depth is None:
            return None
        h, w = depth.shape[:2]
        c = px[list(PALMA_PUNTOS)].mean(axis=0)
        u, v = int(round(c[0])), int(round(c[1]))
        if not (0 <= u < w and 0 <= v < h):
            return None
        ventana = depth[max(0, v - radio):v + radio + 1, max(0, u - radio):u + radio + 1]
        validos = ventana[(ventana > 250) & (ventana < 4000)]
        if validos.size < 6:
            return None
        z = float(np.median(validos)) / 1000.0
        if z_ref is not None and abs(z - z_ref) > Z_PALMA_TOL:
            return None
        return z

    # ---------------------------------------------------------------- ciclo
    def _detectar(self, clave, rgb, t):
        self.t_ms[clave] = max(self.t_ms[clave] + 1, int(t * 1000))
        det = self.detectores[clave] if clave in self.detectores else self.detector_completo
        return det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.t_ms[clave])

    def procesar(self, bgr, lm2d_pose, depth=None, K=None, bgr_hd=None, t=None, z_munecas=None):
        """Devuelve {"L": {...}, "R": {...}} en lados del OPERADOR (ver cabecera).
        lm2d_pose: landmarks 2D de la Pose (33 x (u, v, vis)) o None.
        bgr_hd: version de mas resolucion del MISMO fotograma, o None.
        z_munecas: {"L": z, "R": z} profundidad (m) de las munecas segun la Pose, para validar."""
        h, w = bgr.shape[:2]
        t = time.monotonic() if t is None else t
        self.manos = []
        datos = {}
        if self.recorte and lm2d_pose is not None:
            for lado in ("L", "R"):
                res = self._procesar_lado(lado, bgr, bgr_hd, lm2d_pose, depth, K, t, w, h, z_munecas)
                if res is not None:
                    datos[lado] = res
            return datos
        return self._procesar_completo(bgr, lm2d_pose, depth, K, t, w, h, z_munecas)

    def _procesar_lado(self, lado, bgr, bgr_hd, lm2d, depth, K, t, w, h, z_munecas):
        ventana = self._ventana(lado, lm2d, w, h, t)
        if ventana is None:
            return None
        centro, lado_rec = ventana
        fuente, escala = bgr, 1.0
        if bgr_hd is not None and bgr_hd.shape[1] > w:
            fuente, escala = bgr_hd, bgr_hd.shape[1] / w
        rec, (u0, v0, L) = self._recortar(fuente, centro, lado_rec, escala, self.recorte_px)
        self.ultimo_recorte[lado] = rec
        rgb = np.ascontiguousarray(cv2.cvtColor(rec, cv2.COLOR_BGR2RGB))
        res = self._detectar(lado, rgb, t)
        if not res.hand_landmarks or not res.hand_world_landmarks:
            return None
        lm, mundo = res.hand_landmarks[0], res.hand_world_landmarks[0]
        px = np.array([[u0 + l.x * L, v0 + l.y * L] for l in lm])       # px de la imagen base
        confianza = 1.0
        try:
            confianza = float(res.handedness[0][0].score)
        except Exception:
            pass
        return self._empaquetar(lado, px, mundo, confianza, depth, K, t, w, h, z_munecas, escala)

    def _procesar_completo(self, bgr, lm2d_pose, depth, K, t, w, h, z_munecas):
        """Respaldo: Hand Landmarker sobre la imagen entera (sin Pose, o sin recorte)."""
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self._detectar("completo", rgb, t)
        datos = {}
        if not res.hand_landmarks or not res.hand_world_landmarks:
            return datos
        ref = {}
        if lm2d_pose is not None:
            for lado, (i_mun, *_r) in POSE_MANO.items():
                if lm2d_pose[i_mun][2] >= VIS_MIN_POSE:
                    ref[lado] = np.array(lm2d_pose[i_mun][:2])
        candidatos = []
        for k, (lm, mundo) in enumerate(zip(res.hand_landmarks, res.hand_world_landmarks)):
            px = np.array([[l.x * w, l.y * h] for l in lm])
            lado, dist = None, np.inf
            if ref:
                for s, p in ref.items():
                    d = float(np.linalg.norm(px[MUNECA] - p))
                    if d < dist:
                        lado, dist = s, d
                if dist > 0.2 * w:
                    continue
            else:   # sin cuerpo: la etiqueta de MediaPipe (imagen sin voltear: es correcta)
                try:
                    lado = "L" if res.handedness[k][0].category_name.lower().startswith("left") else "R"
                except Exception:
                    continue
                dist = 0.0
            diag = float(np.linalg.norm(px.max(axis=0) - px.min(axis=0)))
            candidatos.append((dist, lado, px, mundo, diag))
        for dist, lado, px, mundo, diag in sorted(candidatos, key=lambda c: c[0]):
            if lado in datos:
                continue
            res_l = self._empaquetar(lado, px, mundo, 1.0, depth, K, t, w, h, z_munecas, 1.0)
            if res_l is not None:
                datos[lado] = res_l
        return datos

    def _empaquetar(self, lado, px, mundo, confianza, depth, K, t, w, h, z_munecas, escala):
        razon = self.razon_apertura(mundo)
        if razon is None:
            return None
        # world landmarks: al marco del eje optico (vienen alineados con el rayo del recorte)
        centro = px[list(PALMA_PUNTOS)].mean(axis=0)
        P = np.array([[l.x, l.y, l.z] for l in mundo])
        if K is not None:
            P = P @ rotacion_al_rayo(K, centro[0], centro[1]).T
        mundo_c = [_Punto(*p) for p in P]
        R = self.marco(mundo_c)
        if R is None:
            return None
        diag = float(np.linalg.norm(px.max(axis=0) - px.min(axis=0)))
        self.previa[lado] = (px[list(PALMA_PUNTOS)].mean(axis=0), diag, t)
        self.historia[lado].append(razon)
        apertura = float(self.filtros[lado](self.a_apertura(razon, *self.rango(lado)), t))
        z_ref = None if z_munecas is None else z_munecas.get(lado)
        z_palma = self._z_palma(depth, px, z_ref)
        # calidad: tamano de la mano en pixeles REALES de la imagen de la que salio el recorte
        # (HD: diag x escala) y confianza de MediaPipe
        mano_px = diag * escala
        calidad = float(np.clip(confianza, 0.0, 1.0)) * float(np.clip(mano_px / MANO_PX_PLENA, 0.0, 1.0))
        self.manos.append({"lado": lado, "apertura": apertura, "bruto": razon, "puntos_px": px})
        return {"apertura": apertura, "marco": R, "mundo": mundo_c, "px": px, "z_palma": z_palma,
                "tam_px": diag, "calidad": calidad, "confianza": confianza}


def dibujar_manos(img, manos):
    """Esqueleto de la mano: verde cuanto mas abierta, rojo cuanto mas cerrada."""
    for m in manos:
        px, a = m["puntos_px"], m["apertura"]
        color = (int(60 * (1 - a)), int(70 + 160 * a), int(235 - 175 * a))  # BGR
        for i, j in CONEXIONES_MANO:
            cv2.line(img, tuple(px[i].astype(int)), tuple(px[j].astype(int)), color, 2, cv2.LINE_AA)
        for p in px:
            cv2.circle(img, tuple(p.astype(int)), 3, color, -1, cv2.LINE_AA)
