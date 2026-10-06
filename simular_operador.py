"""
Operador SINTETICO visto por dos OAK-D virtuales -> sesion grabada (grabacion.py) con la
verdad conocida. Sirve para probar TODA la cadena (fusion -> retargeting -> IK -> robot)
sin camaras ni operador, y para medir lo que aporta cada cambio con numeros.

  python simular_operador.py sesion_sintetica.pkl [--duracion 20] [--ruido-px 2] [--fps 15]
  python metricas_imitacion.py sesion_sintetica.pkl --orca
  python simular_operador.py sesion_sintetica.pkl --evaluar --orca   # ademas compara con la VERDAD

El operador mueve los brazos con las trayectorias de prueba_ik.operador() (levantar, abrir,
doblar el codo, girar la palma). Las dos camaras (A de frente, B a 55 grados a su derecha)
"ven" lo que veria MediaPipe + la profundidad de la OAK-D con ruido realista:
  - pixeles con ruido gaussiano, profundidad con ruido y algun valor atipico,
  - world landmarks de la mano con la profundidad APLASTADA (x0.6), como MediaPipe,
  - el codo se "pierde" (visibilidad baja y pixel falso) cuando el antebrazo apunta a la
    camara A, y la mano tapa el hombro en esos momentos (profundidad del hombro = la mano).
Las camaras capturan a 15 fps cada una, desfasadas, con 60 ms de retraso de proceso.

  --como-video   reproduce las condiciones de la demo que fallaba (oct-2026):
    - la camara A esta CERCA y BAJA: corta la cabeza y las manos levantadas. Lo que queda
      fuera de la imagen MediaPipe lo sigue dando, pero INVENTADO (pixel extrapolado con un
      error que deriva, sin profundidad, visibilidad ~0.6) y sin cara no hay frontalidad;
    - el operador para a ratos con los brazos COLGANDO (postura simetrica: la que dejaba sin
      pistas al filtro de cruces izquierda/derecha);
    - MediaPipe cambia a veces las etiquetas izquierda/derecha en la camara B (de lado).
  --sin-extrinsecas   reproduce SIN calibracion ChArUco (B tiene que calibrarse sola).
"""
import argparse
import pickle
import time

import numpy as np

import manos3d
import fusion
from prueba_ik import operador, unit
from robot_tron2 import marco_semantico
from suavizado import exp_rot, log_rot

K640 = np.array([[465.0, 0.0, 320.0], [0.0, 465.0, 240.0], [0.0, 0.0, 1.0]])
HOMBRO = {"L": np.array([0.0, 0.19, 1.42]), "R": np.array([0.0, -0.19, 1.42])}
CADERA = {"L": np.array([0.0, 0.11, 0.95]), "R": np.array([0.0, -0.11, 0.95])}
NARIZ = np.array([0.08, 0.0, 1.62])
L_BRAZO, L_ANTEBRAZO = 0.30, 0.26
IDX = {"hombro": {"L": 11, "R": 12}, "codo": {"L": 13, "R": 14}, "muneca": {"L": 15, "R": 16},
       "cadera": {"L": 23, "R": 24}, "indice": {"L": 19, "R": 20}, "menique": {"L": 17, "R": 18},
       "pulgar": {"L": 21, "R": 22}}


def camara(pos, mira, arriba=(0.0, 0.0, 1.0)):
    """Pose (R camara->W, t) de una camara en 'pos' que mira al punto 'mira'.
    Marco de la camara: x derecha, y abajo, z hacia delante."""
    pos, mira = np.asarray(pos, float), np.asarray(mira, float)
    z = mira - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(arriba, float))      # derecha = delante x arriba
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                               # abajo
    return np.column_stack([x, y, z]), pos


def a_camara(P, R, t):
    return (np.asarray(P, float) - t) @ R


def mano_en_mundo(muneca, R_palma, lado):
    """21 puntos de la mano en W: la mano sintetica (palma en z=0, dedos +y, mano derecha)
    colocada con la muneca en 'muneca' y el marco semantico R_palma."""
    P0 = manos3d._mano_sintetica()
    if lado == "L":
        P0 = P0 * np.array([-1.0, 1.0, 1.0])     # espejo: mano izquierda
    # marco semantico de la mano sintetica en sus coordenadas
    R0 = manos3d.marco_de_puntos(P0, robusto=False)
    return muneca + (P0 - P0[0]) @ (R_palma @ R0.T).T


REPOSOS = ((7.0, 13.0), (17.0, 22.0))   # s: tramos con los brazos colgando (--como-video)
T_TRANSICION = 2.0             # s: bajar/levantar los brazos a velocidad normal
W_IMG, H_IMG = 640, 480
VIS_MIN = 0.35                 # = seguimiento_brazos.VIS_MIN


def peso_reposo(t):
    """0 = moviendose (prueba_ik.operador), 1 = brazos colgando; transiciones suaves."""
    w = 0.0
    for a, b in REPOSOS:
        x = min((t - a) / T_TRANSICION, (b - t) / T_TRANSICION)
        w = max(w, float(np.clip(x, 0.0, 1.0)))
    return w * w * (3 - 2 * w)


def operador_reposo(t, lado):
    """Brazos colgando relajados (casi simetricos) con un leve balanceo, palma al muslo."""
    s = 1.0 if lado == "L" else -1.0
    d_b = unit([0.04 * np.sin(0.7 * t + (0.0 if lado == "L" else 1.3)), s * 0.12, -1.0])
    eje = unit(np.cross(d_b, [1.0, 0.0, 1.0]))
    d_a = exp_rot(eje * np.radians(12.0)) @ d_b
    y = np.array([0.0, 1.0, 0.0])
    n = unit(y - d_a * np.dot(y, d_a))
    return d_b, d_a, marco_semantico(d_a, np.cross(n, d_a))


DESFASE_R = 2.5                # s: el brazo derecho va desfasado (un operador real no es simetrico)


def _slerp(a, b, w):
    """Interpolacion esferica entre dos direcciones (velocidad angular uniforme; la lineal
    normalizada gira casi 180 grados de golpe entre direcciones casi opuestas)."""
    ang = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    if ang < 1e-6:
        return a
    eje = np.cross(a, b)
    if np.linalg.norm(eje) < 1e-6:
        eje = np.cross(a, [0.0, 1.0, 0.0])
    return exp_rot(unit(eje) * ang * w) @ a


def operador_video(t, lado, t_rel=None):
    """t_rel: segundos desde el inicio de la sesion (los reposos van por tiempo relativo)."""
    w = peso_reposo(t if t_rel is None else t_rel)
    d_b1, d_a1, R1 = operador(t + (DESFASE_R if lado == "R" else 0.0), lado)
    if w <= 0.0:
        return d_b1, d_a1, R1
    d_b0, d_a0, R0 = operador_reposo(t, lado)
    # el antebrazo ACOMPANA al brazo (girado con el) mientras el codo pasa a su postura final;
    # interpolarlos por separado lo hacia pasar por detras y a traves del pecho
    d_b = _slerp(d_b1, d_b0, w)
    eje = np.cross(d_b1, d_b)
    giro = exp_rot(unit(eje) * np.arcsin(min(np.linalg.norm(eje), 1.0))) if np.linalg.norm(eje) > 1e-9 else np.eye(3)
    if np.dot(d_b1, d_b) < 0:      # mas de 90 grados: el arcoseno no basta
        giro = exp_rot(unit(eje) * (np.pi - np.arcsin(min(np.linalg.norm(eje), 1.0))))
    d_a = _slerp(giro @ d_a1, d_a0, w)
    return d_b, d_a, R1 @ exp_rot(w * log_rot(R1.T @ R0))


def _vis_en_cuadro():
    """Regla de seguimiento_brazos para puntos fuera de la imagen (si existe en esta version)."""
    try:
        import seguimiento_brazos
        return getattr(seguimiento_brazos, "vis_en_cuadro", None)
    except Exception:
        return None


class OperadorSintetico:
    def __init__(self, duracion, fps=15.0, ruido_px=2.0, ruido_z=0.015, semilla=0, oclusiones=True,
                 como_video=False):
        self.duracion, self.fps, self.ruido_px, self.ruido_z = duracion, fps, ruido_px, ruido_z
        self.rng = np.random.default_rng(semilla)
        self.oclusiones = oclusiones
        self.como_video = como_video
        centro = (HOMBRO["L"] + HOMBRO["R"]) / 2 - np.array([0.0, 0.0, 0.25])
        if como_video:
            # A cerca y baja (en la demo se veia de los muslos al cuello): la cabeza y las manos
            # levantadas quedan fuera de la imagen
            self.cams = {"A": camara([1.45, 0.1, 0.85], [0.0, 0.0, 0.78]),
                         "B": camara([1.4, -1.9, 1.5], centro)}
        else:
            self.cams = {"A": camara([2.3, 0.15, 1.35], centro),
                         "B": camara([1.4, -1.9, 1.5], centro)}      # 55 grados a la derecha del operador
        self.vis_cuadro = _vis_en_cuadro()
        self._deriva = {}       # (camara, punto) -> error 3D (m) de MediaPipe fuera de la imagen
        self._cruce_hasta = {"A": -1.0, "B": -1.0}
        self.verdad = []        # (t, lado, d_b, d_a, R_palma) en W
        self.t0 = 1000.0        # inicio de la sesion (grabar())

    def _fuera(self, px):
        return not (0.0 <= px[0] < W_IMG and 0.0 <= px[1] < H_IMG)

    def _inventado(self, etiqueta, nombre, P, R, tc):
        """Punto fuera de la imagen: MediaPipe lo extrapola con un error de ~10-15 cm que
        deriva despacio (no es ruido blanco: dura)."""
        clave = (etiqueta, nombre)
        e = self._deriva.get(clave, self.rng.normal(scale=0.08, size=3))
        e = 0.97 * e + self.rng.normal(scale=0.02, size=3)
        self._deriva[clave] = e
        return P + e

    def postura(self, t):
        out = {}
        for lado in ("L", "R"):
            d_b, d_a, R = (operador_video(t, lado, t - self.t0) if self.como_video else operador(t, lado))
            codo = HOMBRO[lado] + L_BRAZO * d_b
            muneca = codo + L_ANTEBRAZO * d_a
            out[lado] = dict(d_b=d_b, d_a=d_a, R=R, codo=codo, muneca=muneca,
                             mano=mano_en_mundo(muneca, R, lado))
        return out

    def _pixel(self, P, R, t):
        pc = a_camara(P, R, t)
        px = manos3d.proyectar(pc[None, :], K640)[0] + self.rng.normal(scale=self.ruido_px, size=2)
        return px, float(pc[2])

    def _z_ruidosa(self, z):
        if self.rng.random() < 0.03:                      # valor atipico (fondo / agujero)
            return None if self.rng.random() < 0.5 else z + self.rng.uniform(0.3, 1.0)
        return z + self.rng.normal(scale=self.ruido_z * (z / 2.0) ** 2 + 0.004)

    def fotograma(self, etiqueta, t_cap, n):
        R, tc = self.cams[etiqueta]
        post = self.postura(t_cap)
        lm2d = [(0.0, 0.0, 0.0)] * 33
        puntos = {}
        zs = {}
        # cara (para la frontalidad de la calibracion)
        for i, P in ((0, NARIZ), (2, NARIZ + [0, 0.03, 0.03]), (5, NARIZ + [0, -0.03, 0.03])):
            px, z = self._pixel(P, R, tc)
            vis = 0.95 if np.dot(R[:, 2], [-1.0, 0.0, 0.0]) > 0.3 else 0.2   # la camara ve la cara si mira al operador
            if self.como_video and self._fuera(px):
                vis = 0.1                                                   # cabeza cortada: sin cara
            lm2d[i] = (px[0], px[1], vis)
        brazos, aperturas, manos = {}, {}, []
        for lado in ("L", "R"):
            p = post[lado]
            # oclusion: antebrazo apuntando a la camara -> el codo no se ve y la mano tapa el hombro
            hacia_cam = float(np.dot(p["d_a"], -R[:, 2]))
            codo_tapado = self.oclusiones and hacia_cam > 0.8
            mano_delante = self.oclusiones and hacia_cam > 0.6
            vis_codo = 0.25 if codo_tapado else 0.95
            P_med, vis_p = {}, {}
            for nombre, P in (("hombro", HOMBRO[lado]), ("codo", p["codo"]), ("muneca", p["muneca"]), ("cadera", CADERA[lado])):
                px, z = self._pixel(P, R, tc)
                vis = vis_codo if nombre == "codo" else 0.95
                fuera = self.como_video and self._fuera(px)
                P_m = P
                if fuera:
                    P_m = self._inventado(etiqueta, (lado, nombre), P, R, tc)
                    px, z = self._pixel(P_m, R, tc)
                    vis = 0.6
                if nombre == "codo" and codo_tapado:
                    px = px + self.rng.normal(scale=25.0, size=2)     # MediaPipe lo inventa
                if self.vis_cuadro is not None:
                    vis = self.vis_cuadro(vis, px, W_IMG, H_IMG)
                i = IDX[nombre][lado]
                lm2d[i] = (px[0], px[1], vis)
                z_med = self._z_ruidosa(z)
                fuente_z, cal = "oak", 1.0
                if fuera:      # sin profundidad fuera de la imagen: la prevista con MediaPipe
                    z_med, fuente_z, cal = z + self.rng.normal(scale=0.08), "pred", 0.5
                if nombre == "hombro" and mano_delante:
                    # la profundidad que se leeria en el pixel del hombro es la de la MANO; seguimiento_brazos
                    # (_z_hombro_robusta) la detecta con la cadera y usa la de la cadera: eso es lo que llega
                    z_med = float(a_camara(CADERA[lado], R, tc)[2]) + self.rng.normal(scale=0.03)
                if nombre == "codo" and codo_tapado:
                    z_med = None
                zs[i] = z_med
                puntos[i] = dict(uv=(float(px[0]), float(px[1])), vis=vis, z=z_med,
                                 fuente=None if z_med is None else fuente_z,
                                 calidad=cal if z_med is not None else 0.3)
                P_med[nombre], vis_p[nombre] = P_m, vis
            # mano: pixeles de los 21 puntos y nudillos de la Pose
            mano = p["mano"]
            px_m = manos3d.proyectar(a_camara(mano, R, tc), K640) + self.rng.normal(scale=self.ruido_px, size=(21, 2))
            for nombre, k in (("indice", 5), ("menique", 17), ("pulgar", 4)):
                lm2d[IDX[nombre][lado]] = (px_m[k, 0], px_m[k, 1], 0.9)
            # direcciones que daria seguimiento_brazos (puntos en el marco de la camara)
            pc = {k: a_camara(P_med[n], R, tc) for k, n in (("h", "hombro"), ("c", "codo"), ("m", "muneca"))}
            ruido_dir = lambda v: v + self.rng.normal(scale=0.02, size=3)
            sin_brazo = vis_p["hombro"] < VIS_MIN or vis_p["muneca"] < VIS_MIN
            codo_tapado = codo_tapado or vis_p["codo"] < VIS_MIN
            if sin_brazo:
                pass                     # seguimiento_brazos no da brazo sin hombro o sin muneca
            elif codo_tapado:
                d = ruido_dir(pc["m"] - pc["h"])
                d /= np.linalg.norm(d)
                brazos[lado] = dict(dir_brazo=d, dir_antebrazo=d.copy(), fuente="profundidad codo tapado", vis=0.95)
            else:
                d_b = ruido_dir(pc["c"] - pc["h"])
                d_a = ruido_dir(pc["m"] - pc["c"])
                brazos[lado] = dict(dir_brazo=d_b / np.linalg.norm(d_b), dir_antebrazo=d_a / np.linalg.norm(d_a),
                                    fuente="profundidad", vis=float(min(vis_p.values())))
            # world landmarks como MediaPipe: relativos, en el marco de la camara, con la z aplastada
            mc = a_camara(mano, R, tc)
            rel = mc - mc.mean(axis=0)
            rel[:, 2] *= 0.6
            rel += self.rng.normal(scale=0.004, size=rel.shape)
            marco = manos3d.marco_de_puntos(rel, robusto=False)
            z_palma = float(mc[list(manos3d.PALMA_PUNTOS), 2].mean()) + self.rng.normal(scale=0.01)
            diag = float(np.linalg.norm(px_m.max(axis=0) - px_m.min(axis=0)))
            apertura = 0.5 + 0.5 * np.sin(0.9 * t_cap + (0 if lado == "L" else 1.5))
            mano_fuera = self.como_video and np.mean([self._fuera(q) for q in px_m]) > 0.3
            if not mano_fuera:      # Hand Landmarker no ve una mano que esta fuera de la imagen
                aperturas[lado] = dict(apertura=float(apertura), marco=marco, mundo=rel.astype(np.float32),
                                       px=px_m.astype(np.float32), z_palma=z_palma, tam_px=diag,
                                       calidad=float(np.clip(diag * 2 / 160.0, 0, 1)), confianza=0.95)
                manos.append(dict(lado=lado, apertura=float(apertura), bruto=0.6, puntos_px=px_m.astype(np.float32)))
            if etiqueta == "A":
                self.verdad.append((t_cap, lado, p["d_b"], p["d_a"], p["R"]))
        # linea de hombros (unitaria, marco de la camara)
        lh = a_camara(HOMBRO["L"], R, tc) - a_camara(HOMBRO["R"], R, tc)
        arriba = R.T @ np.array([0.0, 0.0, 1.0])
        res = dict(n=n, t=t_cap, t_proc=t_cap + 0.06, brazos=brazos, puntos=puntos, lm2d=lm2d,
                   linea_hombros=lh / np.linalg.norm(lh), K=K640.copy(), arriba=arriba,
                   aperturas=aperturas, manos=manos)
        if self.como_video and etiqueta == "B":
            # MediaPipe, de lado y sin ver bien la cara, cambia a veces izquierda/derecha unos instantes
            if t_cap > self._cruce_hasta["B"] + 2.0 and self.rng.random() < 0.02:
                self._cruce_hasta["B"] = t_cap + self.rng.uniform(0.3, 1.0)
            if t_cap < self._cruce_hasta["B"]:
                res = fusion._cruzar(res)
        return res

    def grabar(self, ruta, t0=1000.0):
        self.t0 = t0
        registros = {"A": [], "B": []}
        for k, etiqueta in enumerate(("A", "B")):
            n = 0
            t = t0 + 0.5 + k * 0.033           # desfase entre camaras
            while t < t0 + self.duracion:
                n += 1
                registros[etiqueta].append(self.fotograma(etiqueta, t, n))
                t += 1.0 / self.fps + self.rng.normal(scale=0.004)
        # extrinsecas B respecto a A:  p_A = R_ab p_B + t_ab
        RA, tA = self.cams["A"]
        RB, tB = self.cams["B"]
        R_ab, t_ab = RA.T @ RB, RA.T @ (tB - tA)
        # calibracion con [c] en t0 + 1.5: la postura real del operador (R camara -> operador)
        R_cal = [(RA.tolist(), 20, 0.5, None, 0.3), (RB.tolist(), 20, 0.5, None, 0.3)]
        if self.como_video:
            # en la demo B no salio de la c (seguia "calibrando"): tiene que calibrarse sola
            # (o venir de las extrinsecas ChArUco)
            R_cal[1] = (None, 3, None, "postura", None)
        eventos = [(t0 + 1.2, "pausa", dict(pausado=True)),
                   (t0 + 1.5, "calibracion", dict(resultado=R_cal)),
                   (t0 + 1.6, "pausa", dict(pausado=False))]
        # (R de camara(): columnas = ejes de la camara en W, o sea camara -> W, lo que devuelve calibrar())
        datos = dict(version=2, meta=dict(camaras=["simA", "simB"], orca=True, sintetico=True,
                                          extrinsecas=(R_ab.tolist(), t_ab.tolist()),
                                          verdad=self.verdad),
                     t0=t0, registros=registros, eventos=eventos, con_video=False)
        with open(ruta, "wb") as f:
            pickle.dump(datos, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"guardado {ruta}: {len(registros['A'])} + {len(registros['B'])} fotogramas, {self.duracion:.0f} s")
        return datos


def evaluar(ruta, xml, orca, csv=None, sin_extrinsecas=False):
    """Reproduce la sesion sin visor y compara el ROBOT con la VERDAD del operador sintetico
    (no con lo que midio la fusion): direcciones de brazo y antebrazo y marco de la palma."""
    import teleop_tron2
    from metricas_imitacion import Metricas
    datos = grabacion_cargar(ruta)
    verdad = datos["meta"]["verdad"]
    por_t = {}
    for t, lado, d_b, d_a, R in verdad:
        por_t.setdefault(lado, []).append((t, d_b, d_a, R))
    tele = teleop_tron2.preparar_reproduccion(ruta, xml, orca, metricas=Metricas(csv), sin_extrinsecas=sin_extrinsecas)
    robot = tele.robot
    errores = {"L": [], "R": []}

    def gancho(t0, snap):
        if snap is None or tele.estado["pausado"]:
            return
        t_cap = snap.get("t_captura", snap["t"])
        for lado, lista in por_t.items():
            ts = np.array([x[0] for x in lista])
            # la sesion se reproduce con el reloj simulado desde t_inicio: convertir
            k = int(np.argmin(np.abs(ts - (t_cap - tele.hilos[0].t_inicio + tele.hilos[0].t0))))
            if abs(ts[k] - (t_cap - tele.hilos[0].t_inicio + tele.hilos[0].t0)) > 0.1:
                continue
            _, d_b, d_a, R_h = lista[k]
            b = robot.brazos[lado]
            p_c, p_m = b.puntos()
            R_base = robot.R_base
            e_b = ang(R_base @ d_b, p_c - b.hombro)
            e_a = ang(R_base @ d_a, p_m - p_c)
            e_R = float(np.degrees(np.linalg.norm(log_rot((R_base @ R_h) @ b.orientacion().T))))
            errores[lado].append((t0, e_b, e_a, e_R))

    tele.gancho = gancho
    t_ini = time.monotonic()
    tele.correr_sin_visor()
    print(f"({time.monotonic() - t_ini:.1f} s de calculo)")
    print("\n--- fusion (lo que midio) frente al robot ---")
    print(tele.metricas.resumen())
    print("\n--- VERDAD del operador sintetico frente al robot (tras los 3 primeros segundos) ---")
    for lado, e in errores.items():
        E = np.array([x for x in e if x[0] > tele.hilos[0].t_inicio + 3.0])
        if len(E) == 0:
            print(f"Brazo {lado}: sin datos")
            continue
        med = np.median(E[:, 1:], axis=0)
        p95 = np.percentile(E[:, 1:], 95, axis=0)
        graves = float(np.mean(np.maximum(E[:, 1], E[:, 2]) > 30.0)) * 100
        print(f"Brazo {lado}: brazo {med[0]:.1f}/{p95[0]:.1f} deg | antebrazo {med[1]:.1f}/{p95[1]:.1f} deg | "
              f"palma {med[2]:.1f}/{p95[2]:.1f} deg   (mediana/p95, {len(E)} ciclos) | "
              f"fallos graves (>30 deg): {graves:.1f} %")
    tele.metricas.cerrar()
    return errores


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))


def grabacion_cargar(ruta):
    with open(ruta, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("salida")
    ap.add_argument("--duracion", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--ruido-px", type=float, default=2.0)
    ap.add_argument("--sin-oclusiones", action="store_true")
    ap.add_argument("--evaluar", action="store_true", help="ademas reproduce y compara con la verdad")
    ap.add_argument("--xml", default="tron2a/DACH_TRON2A/xml/robot_elecnor.xml")
    ap.add_argument("--orca", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--como-video", action="store_true",
                    help="condiciones de la demo que fallaba: A corta al operador, reposos simetricos, cruces en B")
    ap.add_argument("--sin-extrinsecas", action="store_true", help="evaluar sin la calibracion ChArUco")
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()
    OperadorSintetico(args.duracion, args.fps, args.ruido_px, semilla=args.semilla, oclusiones=not args.sin_oclusiones,
                      como_video=args.como_video).grabar(args.salida)
    if args.evaluar:
        evaluar(args.salida, args.xml, args.orca, args.csv, sin_extrinsecas=args.sin_extrinsecas)


if __name__ == "__main__":
    main()
