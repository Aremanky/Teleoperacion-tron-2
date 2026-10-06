"""
Seguimiento de los brazos del operador con MediaPipe Pose Landmarker (API Tasks).

Para cada brazo devuelve la DIRECCION 3D del brazo (hombro->codo) y del
antebrazo (codo->muneca) en el marco de la camara (x derecha, y abajo,
z hacia delante). Solo se usan direcciones, asi que no importa que el
operador sea mas alto o mas bajo que el robot.

Fuente de los puntos 3D (se muestra en la ventana):
  "profundidad": pixel de MediaPipe + profundidad de la OAK-D en los 3 puntos
  "mixta":       profundidad en el hombro; en codo/muneca, donde la OAK-D no da
                 un valor coherente, se usa la profundidad relativa de MediaPipe
  "mediapipe3d": solo coordenadas 3D de MediaPipe (sin OAK-D o sin dato en el hombro)
  "... codo tapado": el codo no se ve (brazo apuntando a la camara). Se toma el
                 brazo como RECTO, de hombro a muneca, en vez de perderlo.

CAMBIOS (oct-2026, imitacion):
  - La profundidad del HOMBRO se mide en el pecho (desplazada hacia el centro del torso),
    no en el pixel del hombro: con la mano delante del hombro se leia la profundidad de
    la MANO y todo el brazo se desplazaba medio metro. Ademas se valida contra la cadera
    y contra el valor anterior (los hombros no saltan 30 cm en una decima de segundo).
  - En codo y muneca, la OAK-D ya no se descarta por discrepar mas de 20 cm con la 'z'
    relativa de MediaPipe (que es su peor dato: con el brazo estirado hacia la camara
    la diferencia real es de 40-60 cm y se tiraba la medida CORRECTA). Ahora se elige la
    profundidad que deja la LONGITUD del hueso mas cerca de la del operador, que se
    aprende sola (LongitudesHumano) y se comparte entre las camaras.
  - Cada punto lleva una 'calidad' (0..1): baja si su hueso sale con una longitud
    imposible (codo inventado por MediaPipe). La fusion 3D lo usa para no creerse
    un codo falso con la confianza de uno bien visto.
  - El hilo pasa a las manos el fotograma HD, la profundidad y la 'z' de las munecas.
  - (demo oct-2026) Un punto FUERA DE LA IMAGEN no se ha visto: MediaPipe lo sigue dando
    (extrapolado, con visibilidad 0.5-0.9) pero es inventado, y ahi no hay profundidad.
    Con la camara A cortando la cabeza y las manos levantadas, el robot copiaba esas munecas
    inventadas. Ahora cuentan como no vistas (vis_en_cuadro): esa camara no da ese brazo y
    lo pone la otra camara, o el robot se queda quieto avisando "no se ve tu brazo".
"""
import os
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

import brazo_geometria

URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")

# Indices de MediaPipe Pose (hombro, codo, muneca). L/R = lado anatomico del operador.
PUNTOS = {"L": (11, 13, 15), "R": (12, 14, 16)}
CONEXIONES = {(11, 12): None, (11, 23): None, (12, 24): None, (23, 24): None,
              (11, 13): "L", (13, 15): "L", (12, 14): "R", (14, 16): "R"}
COLOR_LADO = {"L": (255, 170, 0), "R": (0, 140, 255)}  # BGR: azul (izq), naranja (der)

VIS_MIN = 0.35
RANGO_BRAZO = (0.15, 0.50)        # m, longitudes plausibles para aceptar
RANGO_ANTEBRAZO = (0.12, 0.45)    # la lectura de profundidad
RANGO_HOMBRO_MUNECA = (0.15, 0.85)  # m, idem cuando el codo esta tapado
PUNTOS_3D = (11, 12, 13, 14, 15, 16, 23, 24)   # hombros, codos, munecas, caderas
REF_PUNTO = {13: 11, 15: 11, 14: 12, 16: 12}    # codo/muneca: su profundidad se refiere al hombro
OTRO_HOMBRO = {11: 12, 12: 11}
CADERA_DE = {11: 23, 12: 24}
RANGO_HOMBROS = (0.22, 0.55)      # m, distancia plausible entre los dos hombros
MARGEN_CUADRO = 0.01              # fraccion del ancho/alto: mas alla del borde un punto no se ha visto
TOL_Z = 0.20                      # m: OAK-D y MediaPipe "cuadran" por debajo de esto
TOL_LONGITUD = 0.30               # fraccion: un hueso puede desviarse esto de la longitud aprendida
HOMBRO_HACIA_TORSO = 0.40         # el hombro se mide a esta fraccion del camino hacia el otro hombro
HOMBRO_HACIA_CADERA = 0.25        # (si no se ve el otro hombro) hacia la cadera del mismo lado
HOMBRO_SALTO_MAX = 0.30           # m: salto maximo de la profundidad del hombro entre fotos cercanas
HOMBRO_SALTO_T = 0.25             # s
HOMBRO_CADERA_TOL = 0.35          # m: la profundidad del hombro no se aleja mas de esto de la cadera
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
        if self.x is None or t - self.t > 0.5:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t, 1e-3)
        a_d = self._alfa(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alfa(self.min_cutoff + self.beta * np.linalg.norm(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


def vis_en_cuadro(vis, uv, w, h, margen=MARGEN_CUADRO):
    """Visibilidad de un landmark teniendo en cuenta si cae DENTRO de la imagen (uv en px).
    Fuera de la imagen MediaPipe extrapola: el punto es inventado aunque diga que lo ve."""
    u, v = uv
    if -margen * w <= u <= (1 + margen) * w and -margen * h <= v <= (1 + margen) * h:
        return vis
    return 0.0


def unitario(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def profundidad_en(depth, u, v, radio=4, percentil=30):
    """Profundidad robusta (m) alrededor del pixel (u, v). Toma el percentil 30 de
    la ventana: el brazo suele ser lo mas cercano, asi se evita 'caer' al fondo."""
    if depth is None:
        return None
    h, w = depth.shape[:2]
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < w and 0 <= v < h):
        return None
    ventana = depth[max(0, v - radio):v + radio + 1, max(0, u - radio):u + radio + 1]
    validos = ventana[(ventana > 200) & (ventana < 4000)]  # mm
    if validos.size < 8:
        return None
    return float(np.percentile(validos, percentil)) / 1000.0


class LongitudesHumano:
    """Longitudes de brazo y antebrazo del operador, aprendidas solas con las medidas
    claras (las dos profundidades de la OAK-D presentes y de acuerdo con MediaPipe, hueso
    no alineado con el eje optico). Una instancia compartida entre las camaras."""

    def __init__(self, brazo=0.30, antebrazo=0.27, minimo=20, maximo=300):
        self._defecto = (brazo, antebrazo)
        self.minimo = minimo
        self._b, self._a = [], []
        self._lock = threading.Lock()
        self._max = maximo

    def anadir(self, brazo=None, antebrazo=None):
        with self._lock:
            if brazo is not None and RANGO_BRAZO[0] < brazo < RANGO_BRAZO[1]:
                self._b.append(float(brazo))
                self._b = self._b[-self._max:]
            if antebrazo is not None and RANGO_ANTEBRAZO[0] < antebrazo < RANGO_ANTEBRAZO[1]:
                self._a.append(float(antebrazo))
                self._a = self._a[-self._max:]

    @property
    def fiable(self):
        return len(self._b) >= self.minimo and len(self._a) >= self.minimo

    @property
    def brazo(self):
        with self._lock:
            return float(np.median(self._b)) if len(self._b) >= self.minimo else self._defecto[0]

    @property
    def antebrazo(self):
        with self._lock:
            return float(np.median(self._a)) if len(self._a) >= self.minimo else self._defecto[1]

    def texto(self):
        return (f"brazo {self.brazo * 100:.0f} cm, antebrazo {self.antebrazo * 100:.0f} cm"
                + ("" if self.fiable else " (por defecto, aprendiendo)"))


LONGITUDES = LongitudesHumano()   # compartida por todas las camaras del proceso


class SeguidorBrazos:
    @staticmethod
    def _es_brazo_valido(pts, codo_tapado=False):
        if pts is None or len(pts) != 3:
            return False
        if any(not np.all(np.isfinite(p)) for p in pts):
            return False
        if codo_tapado:
            l = np.linalg.norm(pts[2] - pts[0])
            return bool(RANGO_HOMBRO_MUNECA[0] < l < RANGO_HOMBRO_MUNECA[1])
        l_b = np.linalg.norm(pts[1] - pts[0])
        l_a = np.linalg.norm(pts[2] - pts[1])
        return bool(RANGO_BRAZO[0] < l_b < RANGO_BRAZO[1]
                    and RANGO_ANTEBRAZO[0] < l_a < RANGO_ANTEBRAZO[1])

    def __init__(self, variante="full", usar_profundidad=True, longitudes=None):
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
        self.longitudes = longitudes or LONGITUDES
        self._iniciar_estado(usar_profundidad)

    def _iniciar_estado(self, usar_profundidad):
        self.usar_profundidad = usar_profundidad
        self.t_ms = 0
        self.filtros = {lado: (FiltroOneEuro(), FiltroOneEuro()) for lado in PUNTOS}
        self.lm2d = None            # [(u, v, visibilidad)] en pixeles, para dibujar
        self.puntos = None          # {indice: {uv, vis, z, fuente, calidad}} para la fusion 3D
        self.linea_hombros = None   # hombro izq - hombro der (unitario), para calibrar
        self._z_hombro = {}         # indice -> (z, t) ultima profundidad aceptada del hombro
        self._codo_prev = {}        # lado -> ultimo codo visto (marco camara), para reconstruirlo si se tapa

    def procesar(self, bgr, depth=None, K=None, t=None):
        """t: instante de captura del frame (para los filtros). Por defecto, ahora."""
        h, w = bgr.shape[:2]
        t = time.monotonic() if t is None else t
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.t_ms = max(self.t_ms + 1, int(t * 1000))  # estrictamente creciente
        res = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.t_ms)
        return self._extraer(res, w, h, depth, K, t)

    def _extraer(self, res, w, h, depth, K, t):
        self.lm2d, self.linea_hombros, self.puntos = None, None, None
        brazos = {}
        if not res.pose_landmarks:
            return brazos
        lm = res.pose_landmarks[0]
        mundo = res.pose_world_landmarks[0] if res.pose_world_landmarks else None

        def vis(p):
            return vis_en_cuadro(1.0 if p.visibility is None else p.visibility, (p.x * w, p.y * h), w, h)

        self.lm2d = [(p.x * w, p.y * h, vis(p)) for p in lm]
        con_prof = self.usar_profundidad and depth is not None and K is not None
        self.puntos = self._puntos_3d(lm, mundo, depth if con_prof else None, K, w, h, t)
        if con_prof and mundo is not None:
            self.linea_hombros = self._linea_hombros_profundidad(K)
        if self.linea_hombros is None and mundo is not None:
            self.linea_hombros = unitario(np.array([mundo[11].x - mundo[12].x,
                                                    mundo[11].y - mundo[12].y,
                                                    mundo[11].z - mundo[12].z]))

        for lado, idx in PUNTOS.items():
            v_hombro, v_codo, v_muneca = (vis(lm[i]) for i in idx)
            if v_hombro < VIS_MIN or v_muneca < VIS_MIN:
                continue                       # sin hombro o sin muneca no hay brazo
            codo_tapado = v_codo < VIS_MIN     # tipico con el brazo apuntando a la camara
            pts, fuente = None, None
            if con_prof and mundo is not None:
                pts, fuente = self._puntos_fusion(idx, K, codo_tapado)
            if pts is None and mundo is not None:
                pts = [np.array([mundo[i].x, mundo[i].y, mundo[i].z]) for i in idx]
                fuente = "mediapipe3d"
            if pts is None:
                continue
            if codo_tapado:
                # el codo que da MediaPipe es inventado: se reconstruye en la circunferencia que
                # permiten las longitudes del operador, con el giro del ultimo codo visto
                codo_rec, ok = brazo_geometria.codo_en_circulo(pts[0], pts[2], self.longitudes.brazo,
                                                               self.longitudes.antebrazo,
                                                               self._codo_prev.get(lado), abajo=[0.0, -1.0, 0.0])
                pts = [pts[0], codo_rec, pts[2]]
                d_b, d_a = unitario(pts[1] - pts[0]), unitario(pts[2] - pts[1])
                fuente += " codo reconstruido" if ok else " codo tapado"
            else:
                d_b, d_a = unitario(pts[1] - pts[0]), unitario(pts[2] - pts[1])
                self._codo_prev[lado] = pts[1]
            if d_b is None or d_a is None:
                continue
            if not self._es_brazo_valido(pts, codo_tapado=(codo_tapado and "reconstruido" not in fuente)):
                continue
            f_b, f_a = self.filtros[lado]
            vis_brazo = min(v_hombro, v_muneca) if codo_tapado else min(v_hombro, v_codo, v_muneca)
            brazos[lado] = {"dir_brazo": unitario(f_b(d_b, t)),
                            "dir_antebrazo": unitario(f_a(d_a, t)),
                            "fuente": fuente, "vis": float(vis_brazo)}
        return brazos

    # ---------------------------------------------------------------- profundidad
    def _z_hombro_robusta(self, i, uv, vis, depth, t):
        """Profundidad del hombro i medida en el PECHO: el pixel se desplaza hacia el otro
        hombro (o hacia la cadera) para no leer la mano cuando pasa por delante. Se valida
        con la cadera y con el valor anterior."""
        if vis[i] < VIS_MIN:
            return None
        u, v = uv[i]
        j = OTRO_HOMBRO[i]
        if vis[j] >= VIS_MIN:
            u2, v2 = uv[j]
            u, v = u + HOMBRO_HACIA_TORSO * (u2 - u), v + HOMBRO_HACIA_TORSO * (v2 - v)
        elif vis[CADERA_DE[i]] >= VIS_MIN:
            u2, v2 = uv[CADERA_DE[i]]
            u, v = u + HOMBRO_HACIA_CADERA * (u2 - u), v + HOMBRO_HACIA_CADERA * (v2 - v)
        z = profundidad_en(depth, u, v, radio=5, percentil=40)
        if z is None:
            z = profundidad_en(depth, *uv[i])
        if z is None:
            return None
        z_cad = profundidad_en(depth, *uv[CADERA_DE[i]], radio=5, percentil=40) if vis[CADERA_DE[i]] >= VIS_MIN else None
        prev = self._z_hombro.get(i)
        salto = prev is not None and t - prev[1] < HOMBRO_SALTO_T and abs(z - prev[0]) > HOMBRO_SALTO_MAX
        if salto or (z_cad is not None and abs(z - z_cad) > HOMBRO_CADERA_TOL):
            # lectura contaminada (mano delante del pecho): mejor la cadera o el valor anterior
            if z_cad is not None:
                z = z_cad
            elif prev is not None and t - prev[1] < 1.0:
                z = prev[0]
            else:
                return None
        self._z_hombro[i] = (z, t)
        return z

    def _puntos_3d(self, lm, mundo, depth, K, w, h, t):
        """Para cada punto de PUNTOS_3D: pixel (sin voltear), visibilidad, profundidad z (m,
        a lo largo del eje optico) con su fuente y una calidad 0..1:
          "oak"  medida por la OAK-D (en codo y muneca, si deja el hueso con una longitud creible)
          "pred" hombro + diferencia de profundidad de MediaPipe (cuando la OAK-D no vale)
          None   sin profundidad (sin OAK-D o sin dato en el hombro): solo el rayo del pixel"""
        out = {}
        uv, vis = {}, {}
        for i in PUNTOS_3D:
            p = lm[i]
            uv[i] = (p.x * w, p.y * h)
            vis[i] = vis_en_cuadro(1.0 if p.visibility is None else p.visibility, uv[i], w, h)
            out[i] = {"uv": uv[i], "vis": vis[i], "z": None, "fuente": None, "calidad": 1.0}
        if depth is None or K is None:
            return out
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        def punto(i, z):
            u, v = uv[i]
            return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])

        z_h = {i: self._z_hombro_robusta(i, uv, vis, depth, t) for i in (11, 12)}
        for i in (23, 24):                                   # caderas: la OAK-D directamente
            z = profundidad_en(depth, *uv[i], radio=5, percentil=40) if vis[i] >= VIS_MIN else None
            if z is not None:
                out[i]["z"], out[i]["fuente"] = z, "oak"
        for i in (11, 12):
            if z_h[i] is not None:
                out[i]["z"], out[i]["fuente"] = z_h[i], "oak"
        L_b, L_a = self.longitudes.brazo, self.longitudes.antebrazo
        for hombro, codo, muneca in ((11, 13, 15), (12, 14, 16)):
            if z_h[hombro] is None:
                continue
            p_h = punto(hombro, z_h[hombro])
            # --- codo ---
            z_codo, fuente_c, cal_c, p_c = None, None, 1.0, None
            if vis[codo] >= VIS_MIN:
                z_pred = z_h[hombro] + (mundo[codo].z - mundo[hombro].z) if mundo is not None else None
                z_oak = profundidad_en(depth, *uv[codo])
                z_codo, fuente_c, cal_c, claro = self._elegir_z(z_oak, z_pred, p_h, L_b, lambda z: punto(codo, z))
                if z_codo is not None:
                    p_c = punto(codo, z_codo)
                    if self._medida_para_aprender(fuente_c, z_oak, z_pred, p_c, p_h):
                        self.longitudes.anadir(brazo=float(np.linalg.norm(p_c - p_h)))
            # --- muneca (su hueso sale del codo si lo hay; si no, del hombro) ---
            if vis[muneca] >= VIS_MIN:
                z_pred = z_h[hombro] + (mundo[muneca].z - mundo[hombro].z) if mundo is not None else None
                z_oak = profundidad_en(depth, *uv[muneca])
                if p_c is not None:
                    z_m, fuente_m, cal_m, claro = self._elegir_z(z_oak, z_pred, p_c, L_a, lambda z: punto(muneca, z))
                    if z_m is not None and cal_c >= 0.8:
                        p_m = punto(muneca, z_m)
                        if self._medida_para_aprender(fuente_m, z_oak, z_pred, p_m, p_c):
                            self.longitudes.anadir(antebrazo=float(np.linalg.norm(p_m - p_c)))
                else:
                    z_m, fuente_m, cal_m, _ = self._elegir_z(z_oak, z_pred, p_h, L_b + L_a,
                                                             lambda z: punto(muneca, z))
                if z_m is not None:
                    out[muneca].update(z=z_m, fuente=fuente_m, calidad=cal_m)
            if z_codo is not None:
                out[codo].update(z=z_codo, fuente=fuente_c, calidad=cal_c)
        return out

    @staticmethod
    def _medida_para_aprender(fuente, z_oak, z_pred, p, p_origen):
        """Una medida sirve para aprender la longitud del hueso si viene de la OAK-D, no
        discrepa mucho de MediaPipe (no es un atipico) y el hueso no apunta a la camara
        (la profundidad es la coordenada menos precisa)."""
        if fuente != "oak" or z_oak is None:
            return False
        if z_pred is not None and abs(z_oak - z_pred) > 2.0 * TOL_Z:
            return False
        return abs(p[2] - p_origen[2]) < 0.7 * np.linalg.norm(p - p_origen)

    @staticmethod
    def _elegir_z(z_oak, z_pred, p_origen, L_esperada, punto_con_z):
        """Profundidad de un codo/muneca: entre la OAK-D (z_oak) y la prevista con MediaPipe
        (z_pred) se queda la que deja el hueso desde p_origen con la longitud mas creible.
        punto_con_z(z) -> punto 3D del pixel a esa profundidad.
        Devuelve (z, fuente, calidad 0..1, claro). 'claro' = las dos coinciden (medida fiable)."""
        if z_oak is None and z_pred is None:
            return None, None, 0.0, False
        if z_oak is not None and z_pred is not None and abs(z_oak - z_pred) < TOL_Z:
            return z_oak, "oak", 1.0, True
        cands = [(z, f) for z, f in ((z_oak, "oak"), (z_pred, "pred")) if z is not None]
        longitud = lambda z: float(np.linalg.norm(punto_con_z(z) - p_origen))
        if len(cands) == 1:
            z, fuente = cands[0]
            err = abs(longitud(z) - L_esperada) / max(L_esperada, 1e-6)
            calidad = (0.9 if fuente == "oak" else 0.6) * (1.0 if err <= TOL_LONGITUD else 0.3)
            return z, fuente, calidad, False
        # dos candidatas que no coinciden: la que deje la longitud mas cerca de la esperada
        mejor = min(cands, key=lambda c: abs(longitud(c[0]) - L_esperada))
        err = abs(longitud(mejor[0]) - L_esperada) / max(L_esperada, 1e-6)
        if err > TOL_LONGITUD:
            # ninguna cuadra: seguramente un codo inventado por MediaPipe; la prevista, con poca calidad
            z, fuente = next(c for c in cands if c[1] == "pred")
            return z, fuente, 0.25, False
        return mejor[0], mejor[1], (0.8 if mejor[1] == "oak" else 0.5), False

    def _puntos_fusion(self, idx, K, codo_tapado=False):
        """Puntos 3D (marco camara) de hombro, codo y muneca a partir de self.puntos.
        Devuelve (pts, fuente) o (None, None) si falta la profundidad del hombro."""
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        P = self.puntos
        if P is None or P[idx[0]]["z"] is None:
            return None, None
        pts, n_oak = [], 0
        for k, i in enumerate(idx):
            z = P[i]["z"]
            if z is None:
                if k == 1 and codo_tapado:          # el codo no importa: cualquier cosa finita
                    z = P[idx[0]]["z"]
                else:
                    return None, None
            elif k > 0 and P[i]["fuente"] == "oak":
                n_oak += 1
            u, v = P[i]["uv"]
            pts.append(np.array([(u - cx) * z / fx, (v - cy) * z / fy, z]))
        if codo_tapado:
            l_hm = np.linalg.norm(pts[2] - pts[0])
            if not RANGO_HOMBRO_MUNECA[0] < l_hm < RANGO_HOMBRO_MUNECA[1]:
                return None, None
        else:
            l_b, l_a = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1])
            if not (RANGO_BRAZO[0] < l_b < RANGO_BRAZO[1] and RANGO_ANTEBRAZO[0] < l_a < RANGO_ANTEBRAZO[1]):
                return None, None
        if any(not np.all(np.isfinite(p)) for p in pts):
            return None, None
        return pts, ("profundidad" if n_oak == 2 else "mixta")

    def _linea_hombros_profundidad(self, K):
        """Hombro izquierdo - hombro derecho (unitario) con las profundidades robustas."""
        P = self.puntos
        if P is None or P[11]["z"] is None or P[12]["z"] is None:
            return None
        if min(P[11]["vis"], P[12]["vis"]) < VIS_MIN:
            return None
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p = {}
        for i in (11, 12):
            (u, v), z = P[i]["uv"], P[i]["z"]
            p[i] = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
        d = p[11] - p[12]
        if not RANGO_HOMBROS[0] < np.linalg.norm(d) < RANGO_HOMBROS[1]:
            return None
        return unitario(d)

    def z_munecas(self):
        """{"L": z, "R": z} profundidad (m) de las munecas, si se conoce (para las manos)."""
        if self.puntos is None:
            return {}
        return {lado: self.puntos[idx[2]]["z"] for lado, idx in PUNTOS.items()
                if self.puntos[idx[2]]["z"] is not None}


class HiloSeguimiento(threading.Thread):
    """Lee la camara y ejecuta MediaPipe (cuerpo y, si se pasa, manos) en segundo
    plano. El bucle principal pide el ultimo resultado con ultimo() sin esperar.

    DOS hilos: uno solo CAPTURA (a los fps de la camara, 30) y deja siempre el ultimo
    fotograma en bruto; el otro ejecuta MediaPipe sobre el fotograma mas reciente (a lo
    que de el PC). Asi la imagen de la ventana (ultimo_crudo()) va fluida a los fps de
    la camara aunque MediaPipe vaya a 10-15 fps, y MediaPipe nunca procesa un
    fotograma atrasado."""

    def __init__(self, camara, seguidor, seguidor_manos=None, grabadora=None, etiqueta=None):
        super().__init__(daemon=True)
        self.camara, self.seguidor, self.manos = camara, seguidor, seguidor_manos
        self.grabadora, self.etiqueta = grabadora, etiqueta
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._ultimo = None
        self._crudo = None          # (n, bgr, depth, K, t_captura, bgr_hd)
        self.activo, self.error, self.fps, self.fps_camara = True, None, 0.0, 0.0
        self._captura = threading.Thread(target=self._bucle_captura, daemon=True)

    def start(self):
        self._captura.start()
        super().start()

    # ---------------------------------------------------------- captura
    def _bucle_captura(self):
        t_ant, n = time.monotonic(), 0
        try:
            while self.activo:
                bgr, depth = self.camara.leer()
                if bgr is None:
                    self.error = "no llegan imagenes de la camara"
                    return
                t_cap = getattr(self.camara, "t_captura", None) or time.monotonic()
                bgr_hd = getattr(self.camara, "bgr_hd", None)
                ahora = time.monotonic()
                self.fps_camara = 0.9 * self.fps_camara + 0.1 / max(ahora - t_ant, 1e-3)
                t_ant, n = ahora, n + 1
                with self._cond:
                    self._crudo = (n, bgr, depth, self.camara.K, t_cap, bgr_hd)
                    self._cond.notify_all()
        except Exception as e:  # se informa desde el hilo principal
            self.error = repr(e)

    def ultimo_crudo(self):
        """Ultimo fotograma de la camara SIN procesar: dict(n, bgr, depth, t) o None."""
        c = self._crudo
        return None if c is None else dict(n=c[0], bgr=c[1], depth=c[2], t=c[4])

    # ---------------------------------------------------------- MediaPipe
    def run(self):
        t_ant, n, n_visto = time.monotonic(), 0, -1
        try:
            while self.activo:
                with self._cond:
                    self._cond.wait_for(lambda: (not self.activo) or self.error is not None
                                        or (self._crudo is not None and self._crudo[0] != n_visto),
                                        timeout=0.5)
                    crudo = self._crudo
                if self.error:
                    return
                if crudo is None or crudo[0] == n_visto:
                    continue
                n_visto, bgr, depth, K, t_cap, bgr_hd = crudo
                brazos = self.seguidor.procesar(bgr, depth, K, t_cap)
                aperturas, dibujo_manos = {}, []
                if self.manos is not None:
                    aperturas = self.manos.procesar(bgr, self.seguidor.lm2d, depth=depth, K=K, bgr_hd=bgr_hd,
                                                    t=t_cap, z_munecas=self.seguidor.z_munecas())
                    dibujo_manos = self.manos.manos
                ahora = time.monotonic()
                self.fps = 0.9 * self.fps + 0.1 / max(ahora - t_ant, 1e-3)
                t_ant, n = ahora, n + 1
                imu = getattr(self.camara, "imu", None)
                resultado = dict(n=n, t=t_cap, t_proc=ahora, bgr=bgr, depth=depth, brazos=brazos,
                                 puntos=self.seguidor.puntos, K=K,
                                 aperturas=aperturas, manos=dibujo_manos,
                                 lm2d=self.seguidor.lm2d,
                                 linea_hombros=self.seguidor.linea_hombros,
                                 arriba=None if imu is None else imu.arriba(),
                                 imu_estado=None if imu is None else dict(
                                     mov=imu.movimientos, ultimo=imu.ultimo_mov, girando=imu.girando(),
                                     despl=imu.desplazandose(), en_mov=imu.en_movimiento()))
                with self._lock:
                    self._ultimo = resultado
                if self.grabadora is not None:
                    self.grabadora.anotar(self.etiqueta, resultado)
        except Exception as e:  # se informa desde el hilo principal
            self.error = repr(e)

    def ultimo(self):
        with self._lock:
            return self._ultimo

    def parar(self):
        self.activo = False
        with self._cond:
            self._cond.notify_all()
        self.join(timeout=2.0)
        self._captura.join(timeout=2.0)


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
