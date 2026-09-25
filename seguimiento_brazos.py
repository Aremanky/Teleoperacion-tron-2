"""
Seguimiento de los brazos del operador con MediaPipe Pose Landmarker (API Tasks,
la unica disponible en las versiones actuales de mediapipe).

Para cada brazo devuelve la DIRECCION 3D del brazo (hombro->codo) y del
antebrazo (codo->muneca) en el marco de la camara (x derecha, y abajo,
z hacia delante). Solo se usan direcciones, asi que no importa que el
operador sea mas alto o mas bajo que el robot.

Fuente de los puntos 3D (se muestra en la ventana):
  "profundidad": pixel de MediaPipe + profundidad de la OAK-D en los 3 puntos
  "mixta":       profundidad en el hombro; en codo/muneca, donde la OAK-D no da
                 un valor coherente, se usa la profundidad relativa de MediaPipe
  "mediapipe3d": solo coordenadas 3D de MediaPipe (sin OAK-D o sin dato en el hombro)

HiloSeguimiento ejecuta camara + MediaPipe en segundo plano para que el bucle
del robot vaya a su ritmo aunque el seguimiento sea mas lento.
"""
import os
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")

# Indices de MediaPipe Pose (hombro, codo, muneca). L/R = lado anatomico del operador.
PUNTOS = {"L": (11, 13, 15), "R": (12, 14, 16)}
CONEXIONES = {(11, 12): None, (11, 23): None, (12, 24): None, (23, 24): None,
              (11, 13): "L", (13, 15): "L", (12, 14): "R", (14, 16): "R"}
COLOR_LADO = {"L": (255, 170, 0), "R": (0, 140, 255)}  # BGR: azul (izq), naranja (der)

VIS_MIN = 0.5                     # visibilidad minima de un landmark
RANGO_BRAZO = (0.15, 0.50)        # m, longitudes plausibles para aceptar
RANGO_ANTEBRAZO = (0.12, 0.45)    # la lectura de profundidad
TOL_Z = 0.20                      # m: diferencia maxima entre la profundidad de la OAK-D
                                  #    y la que predice MediaPipe para fiarse de la OAK-D
FILTRO_MIN_CUTOFF = 0.8           # Hz: mas bajo = menos temblor en reposo
FILTRO_BETA = 0.7                 # mas alto = menos retraso en movimientos rapidos


class FiltroOneEuro:
    """Filtro One Euro (Casiez et al., 2012): poco temblor en reposo y poco retraso al moverse."""

    def __init__(self, min_cutoff=FILTRO_MIN_CUTOFF, beta=FILTRO_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alfa(corte, dt):
        tau = 1.0 / (2 * np.pi * corte)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t, 1e-3)
        a_d = self._alfa(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alfa(self.min_cutoff + self.beta * np.linalg.norm(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


def unitario(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def profundidad_en(depth, u, v, radio=4):
    """Profundidad robusta (m) alrededor del pixel (u, v). Toma el percentil 30 de
    la ventana: el brazo suele ser lo mas cercano, asi se evita 'caer' al fondo."""
    h, w = depth.shape[:2]
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < w and 0 <= v < h):
        return None
    ventana = depth[max(0, v - radio):v + radio + 1, max(0, u - radio):u + radio + 1]
    validos = ventana[(ventana > 200) & (ventana < 4000)]  # mm
    if validos.size < 8:
        return None
    return float(np.percentile(validos, 30)) / 1000.0


class SeguidorBrazos:
    def __init__(self, variante="full", usar_profundidad=True):
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"pose_landmarker_{variante}.task")
        if not os.path.exists(ruta):
            print(f"Descargando el modelo de MediaPipe ({variante}) en {ruta} ...")
            urllib.request.urlretrieve(URL_MODELO.format(v=variante), ruta)
        opciones = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=ruta),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(opciones)
        self._iniciar_estado(usar_profundidad)

    def _iniciar_estado(self, usar_profundidad):
        self.usar_profundidad = usar_profundidad
        self.t_ms = 0
        self.filtros = {lado: (FiltroOneEuro(), FiltroOneEuro()) for lado in PUNTOS}
        self.lm2d = None            # [(u, v, visibilidad)] en pixeles, para dibujar
        self.linea_hombros = None   # hombro izq - hombro der (unitario), para calibrar

    def procesar(self, bgr, depth=None, K=None):
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.t_ms = max(self.t_ms + 1, int(time.monotonic() * 1000))  # estrictamente creciente
        res = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.t_ms)
        return self._extraer(res, w, h, depth, K, time.monotonic())

    def _extraer(self, res, w, h, depth, K, t):
        self.lm2d, self.linea_hombros = None, None
        brazos = {}
        if not res.pose_landmarks:
            return brazos
        lm = res.pose_landmarks[0]
        mundo = res.pose_world_landmarks[0] if res.pose_world_landmarks else None

        def vis(p):
            return 1.0 if p.visibility is None else p.visibility

        self.lm2d = [(p.x * w, p.y * h, vis(p)) for p in lm]
        if mundo is not None:
            self.linea_hombros = unitario(np.array([mundo[11].x - mundo[12].x,
                                                    mundo[11].y - mundo[12].y,
                                                    mundo[11].z - mundo[12].z]))

        for lado, idx in PUNTOS.items():
            if min(vis(lm[i]) for i in idx) < VIS_MIN:
                continue
            pts, fuente = None, None
            if self.usar_profundidad and depth is not None and K is not None and mundo is not None:
                pts, fuente = self._puntos_fusion(lm, mundo, idx, depth, K, w, h)
            if pts is None and mundo is not None:
                pts = [np.array([mundo[i].x, mundo[i].y, mundo[i].z]) for i in idx]
                fuente = "mediapipe3d"
            if pts is None:
                continue
            d_b, d_a = unitario(pts[1] - pts[0]), unitario(pts[2] - pts[1])
            if d_b is None or d_a is None:
                continue
            f_b, f_a = self.filtros[lado]
            brazos[lado] = {"dir_brazo": unitario(f_b(d_b, t)),
                            "dir_antebrazo": unitario(f_a(d_a, t)),
                            "fuente": fuente}
        return brazos

    def _puntos_fusion(self, lm, mundo, idx, depth, K, w, h):
        """Pixel (u, v) de MediaPipe + profundidad. El hombro (sobre el torso) da la
        profundidad de referencia. En codo y muneca se usa la de la OAK-D si es
        coherente con la que predice MediaPipe; si no, la prediccion. Asi un pixel
        sin dato no hace saltar de fuente a todo el brazo."""
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        uv = [(lm[i].x * w, lm[i].y * h) for i in idx]
        z_hombro = profundidad_en(depth, *uv[0])
        if z_hombro is None:
            return None, None
        pts, n_oak = [], 0
        for k, (i, (u, v)) in enumerate(zip(idx, uv)):
            z = z_hombro
            if k > 0:
                z_pred = z_hombro + (mundo[i].z - mundo[idx[0]].z)
                z_oak = profundidad_en(depth, u, v)
                if z_oak is not None and abs(z_oak - z_pred) < TOL_Z:
                    z, n_oak = z_oak, n_oak + 1
                else:
                    z = z_pred
            pts.append(np.array([(u - cx) * z / fx, (v - cy) * z / fy, z]))
        l_b, l_a = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1])
        if not (RANGO_BRAZO[0] < l_b < RANGO_BRAZO[1] and RANGO_ANTEBRAZO[0] < l_a < RANGO_ANTEBRAZO[1]):
            return None, None
        return pts, ("profundidad" if n_oak == 2 else "mixta")


class HiloSeguimiento(threading.Thread):
    """Lee la camara y ejecuta MediaPipe en segundo plano. El bucle principal pide
    el ultimo resultado con ultimo() sin tener que esperar."""

    def __init__(self, camara, seguidor):
        super().__init__(daemon=True)
        self.camara, self.seguidor = camara, seguidor
        self._lock = threading.Lock()
        self._ultimo = None
        self.activo, self.error, self.fps = True, None, 0.0

    def run(self):
        t_ant, n = time.monotonic(), 0
        try:
            while self.activo:
                bgr, depth = self.camara.leer()
                if bgr is None:
                    self.error = "no llegan imagenes de la camara"
                    return
                brazos = self.seguidor.procesar(bgr, depth, self.camara.K)
                ahora = time.monotonic()
                self.fps = 0.9 * self.fps + 0.1 / max(ahora - t_ant, 1e-3)
                t_ant, n = ahora, n + 1
                with self._lock:
                    self._ultimo = dict(n=n, t=ahora, bgr=bgr, depth=depth, brazos=brazos,
                                        lm2d=self.seguidor.lm2d,
                                        linea_hombros=self.seguidor.linea_hombros)
        except Exception as e:  # se informa desde el hilo principal
            self.error = repr(e)

    def ultimo(self):
        with self._lock:
            return self._ultimo

    def parar(self):
        self.activo = False
        self.join(timeout=2.0)


def dibujar_esqueleto(img, lm2d):
    """Dibuja tronco y brazos sobre la imagen (sin texto, para poder voltearla despues)."""
    if lm2d is None:
        return
    p = lambda i: (int(lm2d[i][0]), int(lm2d[i][1]))
    for (a, b), lado in CONEXIONES.items():
        if lm2d[a][2] >= VIS_MIN and lm2d[b][2] >= VIS_MIN:
            cv2.line(img, p(a), p(b), COLOR_LADO.get(lado, (200, 200, 200)), 3, cv2.LINE_AA)
    for lado, idx in PUNTOS.items():
        for i in idx:
            if lm2d[i][2] >= VIS_MIN:
                cv2.circle(img, p(i), 6, COLOR_LADO[lado], -1, cv2.LINE_AA)
                cv2.circle(img, p(i), 6, (255, 255, 255), 1, cv2.LINE_AA)