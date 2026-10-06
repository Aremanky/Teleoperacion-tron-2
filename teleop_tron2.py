"""
Teleoperacion de los brazos del TRON 2 (DACH_TRON2A) en MuJoCo con una o dos OAK-D.

Uso (desde la raiz del repositorio):
  python teleop_tron2.py --camara2 --orca      # DOS OAK-D fusionadas + OrcaHand (lo habitual)
  python teleop_tron2.py                       # una OAK-D: RGB + profundidad, pinzas originales
  python teleop_tron2.py --solo-mediapipe      # OAK-D sin usar la profundidad
  python teleop_tron2.py --webcam 0            # sin OAK-D, con una webcam normal
  python teleop_tron2.py --modelo lite         # MediaPipe mas rapido (menos preciso)
  python teleop_tron2.py --sin-manos           # no seguir las manos (va mas rapido)
  python teleop_tron2.py --sin-giro            # manos solo para abrir/cerrar, sin orientar la muneca
  python teleop_tron2.py --escenario tuberias  # arranca ya con el escenario de tuberias a la vista
  python teleop_tron2.py --camara2 --mxid ID_A ID_B   # fijando cual es cual (python camaras.py --listar)
  python teleop_tron2.py --proyector 0         # sin el proyector IR de las OAK-D Pro (por defecto al 70 %)
  python teleop_tron2.py --sin-hd              # sin el fotograma HD para las manos (si el USB no da)
  python teleop_tron2.py ... --grabar sesion.pkl       # graba la sesion (ver grabacion.py)
  python teleop_tron2.py --reproducir sesion.pkl --orca  # la reproduce sin camaras

QUE IMITA EL ROBOT
  - Las DIRECCIONES de tu brazo y antebrazo (no las posiciones: tu y el robot no medis igual).
  - La ORIENTACION COMPLETA de tu palma (los 3 grados de libertad de la muneca), comparando el
    marco semantico de tu mano (dedos, ancho de la palma, normal) con el mismo marco de la
    OrcaHand. Antes solo se copiaba el giro del antebrazo, y mal (ver robot_tron2.py).
  - Los DEDOS, uno a uno (--orca): imitacion pura; con la tecla [g] se cambia al modo de
    pinzas predefinidas del proyecto ARCTOS.
  Cada mano manda sobre el brazo de su lado (o el contrario en modo espejo).

Teclas (con la ventana de la camara seleccionada):
  q / ESC  salir
  p        pausar / reanudar (embrague: congela el robot mientras te recolocas)
  r        devolver los brazos a la postura de reposo
  c        calibrar: de pie, QUIETO 1.5 s, brazos relajados hacia abajo. Orienta al
           operador respecto al robot. Flujo recomendado: p (pausa) -> r (reposo) ->
           colocarse -> c -> p. Con --camara2 basta con que te vea UNA camara: la
           relacion entre camaras (camaras_extrinsecas.json, de calibrar_extrinseca.py, o
           autocalibrada) se conserva
  m        modo espejo on/off (tu brazo izquierdo mueve el derecho del robot)
  g        dedos: imitacion pura <-> pinzas predefinidas (solo --orca)
  v        ver el robot desde detras / desde delante
  1        escenario de tuberias: la primera vez lo hace aparecer; despues lo reinicia

Estructura: un hilo por camara lee y ejecuta MediaPipe (~12-25 fps), fusion.py junta las
camaras y el bucle principal (clase Teleoperador) mueve el robot a 60 Hz suavizando los
objetivos entre fotos. metricas_imitacion.py mide lo bien que imita sobre una grabacion.
"""
import argparse
import csv
import time

import cv2
import numpy as np

import escenario_tuberias
import grabacion
import mano_orca
import fusion as fusion_mod
from fusion import CalibracionCamaras, FusionCamaras, cargar_extrinsecas
from requisitos import Requisitos
from robot_tron2 import RobotTron2, ortonormalizar
from seguimiento_brazos import COLOR_LADO, PUNTOS, VIS_MIN, LONGITUDES, dibujar_esqueleto
from suavizado import ObjetivoSuave, RotacionSuave

VENTANA = "Seguimiento OAK-D"
VENTANA2 = "Seguimiento OAK-D (camara 2)"

FREC_CONTROL = 60.0        # Hz del bucle del robot
# Suavizado CONTINUO de objetivos (suavizado.py): muelle criticamente amortiguado entre fotos.
# Constante de tiempo ~ 2/omega: 18 -> ~0.11 s (mas alto = mas rapido pero menos suave).
# (oct-2026) 14 -> 18: con el operador moviendose el muelle se quedaba 2.5-4 grados por detras;
# en simulacion: antebrazo 5.4 -> 5.0 grados de mediana, p95 del escenario de la demo 23 -> 19,
# temblor (aceleracion articular RMS) +8 %. (Probado tambien darle al muelle la velocidad del
# objetivo sacada de las fotos: cuadruplicaba el temblor.)
OMEGA_OBJETIVO = 18.0
OMEGA_ORIENT = 12.0        # idem para la orientacion de la palma
SALTO_ORIENT = np.radians(60)   # un salto mayor del marco de la mano entre dos fotos se ignora
                                # (salvo que se repita 3 veces: entonces es real)
ZONA_MUERTA_OBJ = 0.004    # m: el objetivo no se mueve por temblores menores que esto
# Brazo casi estirado = singularidad de la IK. Antes se acercaba la MUNECA al hombro (92-97 %
# del alcance) sin tocar el codo: con las proporciones del TRON 2 eso empezaba ya con 47 grados
# de codo, un brazo humano recto (10 grados) salia con 30 en el robot y los objetivos de codo y
# muneca quedaban incoherentes (6 cm de error de codo, 17 grados de antebrazo). Ahora se impone
# una flexion MINIMA del codo, suave, en el plano en que dobla el operador: codo y muneca
# siguen siendo un brazo valido y por encima de 2*FLEX_MIN_ROBOT no cambia nada.
FLEX_MIN_ROBOT = 8.0       # grados
SALTO_LOG = 0.10           # m: si el objetivo de la muneca salta mas que esto entre dos fotos, se anota
CADUCIDAD = 0.5            # s: si no hay datos nuevos en este tiempo, el robot se queda quieto
W_CODO = 0.8               # peso del objetivo del codo frente al de la muneca (antes 0.6)
# Con el brazo casi recto la direccion del codo es ruido. Antes se ignoraba hasta 10 grados y se
# atenuaba hasta 30: un brazo relajado con 20 grados de codo salia casi recto. Con la fusion 3D
# el ruido es menor y se puede bajar mucho.
ANG_RECTO = (4.0, 14.0)    # grados: por debajo el codo no cuenta; por encima cuenta entero
ZONA_MUERTA_CODO = (2.0, 8.0)   # grados de flexion que se tratan como brazo recto (se atenuan)
CALIDAD_MIN_ORIENT = 0.2   # calidad minima de la mano para fiarse de su orientacion
ESPEJO_M = np.diag([1.0, -1.0, 1.0])   # reflexion izquierda-derecha en el marco del robot
UMBRAL_CONGELADO = 0.08    # m: error de muneca a partir del cual se avisa de que el robot no llega
HZ_VISOR = 30.0            # el visor de MuJoCo se refresca a esto
HZ_VENTANA = 30.0          # ventana de la camara
HZ_VENTANA2 = 15.0

COLOR_CODO = [1.0, 0.85, 0.0, 0.8]    # marcador amarillo en MuJoCo
COLOR_MUNECA = [0.1, 1.0, 0.3, 0.8]   # marcador verde en MuJoCo
OTRO_LADO = {"L": "R", "R": "L"}


# ---------------------------------------------------------------- retargeting
def enderezar(d_b, d_a):
    """Zona muerta en la flexion del codo. Con el brazo casi estirado, el ruido del
    seguimiento se convierte en pequenas flexiones en direcciones aleatorias, y el
    robot tiene que girar mucho el hombro para seguirlas (singularidad)."""
    theta = float(np.arccos(np.clip(np.dot(d_b, d_a), -1.0, 1.0)))
    s = np.sin(theta)
    if s < 1e-6:
        return d_a
    lo, hi = np.radians(ZONA_MUERTA_CODO)
    x = np.clip((theta - lo) / (hi - lo), 0.0, 1.0)
    theta_n = theta * x * x * (3 - 2 * x)
    return (np.sin(theta - theta_n) * d_b + np.sin(theta_n) * d_a) / s


def flexion_minima(d_b, d_a, minimo_deg=FLEX_MIN_ROBOT):
    """Antebrazo con al menos 'minimo' grados de codo (marco del robot: x delante, z arriba).
    Suave y de pendiente continua: phi -> minimo + phi^2 / (4 minimo) hasta 2*minimo; por
    encima, igual. Dobla en el plano en que ya dobla el operador; con el brazo del todo recto,
    hacia delante/arriba (como dobla un codo humano)."""
    minimo = np.radians(minimo_deg)
    phi = float(np.arccos(np.clip(np.dot(d_b, d_a), -1.0, 1.0)))
    if phi >= 2 * minimo:
        return d_a
    phi_n = minimo + phi * phi / (4 * minimo)
    perp = d_a - d_b * np.dot(d_a, d_b)
    if np.linalg.norm(perp) < 1e-3:
        perp = np.array([1.0, 0.0, 1.0]) - d_b * np.dot([1.0, 0.0, 1.0], d_b)
        if np.linalg.norm(perp) < 1e-3:
            perp = np.array([1.0, 0.0, 0.0]) - d_b * d_b[0]
    perp = perp / np.linalg.norm(perp)
    return np.cos(phi_n) * d_b + np.sin(phi_n) * perp


def objetivos(robot, lado_robot, datos, R_cam, espejo):
    """Direcciones del operador -> posiciones objetivo del codo y la muneca del robot."""
    brazo = robot.brazos[lado_robot]
    d_b = robot.R_base @ (R_cam @ datos["dir_brazo"])
    d_a = robot.R_base @ (R_cam @ enderezar(datos["dir_brazo"], datos["dir_antebrazo"]))
    if espejo:
        d_b = ESPEJO_M @ d_b
        d_a = ESPEJO_M @ d_a
    # brazo casi estirado = singularidad de la IK: codo minimamente doblado (codo y muneca coherentes)
    d_a_robot = flexion_minima(d_b, d_a)
    codo = brazo.hombro + brazo.L_brazo * d_b
    muneca = codo + brazo.L_antebrazo * d_a_robot
    return codo, muneca, d_b, d_a


def marco_mano(robot, mano, R_cam, espejo):
    """Marco semantico de la mano del operador (de la fusion, en el marco de salida) llevado
    al marco del robot. None si esa mano no trae marco: entonces la IK mantiene la ultima
    orientacion (nunca se inventa una). En modo espejo se refleja y se invierte el eje
    transversal de la palma para que siga siendo un marco derecho."""
    R_mano = None if not isinstance(mano, dict) else mano.get("marco")
    if R_mano is None:
        return None
    R_mano = np.asarray(R_mano, dtype=float)
    if R_mano.shape != (3, 3) or not np.all(np.isfinite(R_mano)) or abs(np.linalg.det(R_mano)) < 1e-3:
        return None
    R = robot.R_base @ R_cam @ R_mano
    if espejo:
        R = ESPEJO_M @ R @ np.diag([1.0, -1.0, 1.0])
    return ortonormalizar(R)


def peso_codo(datos):
    """Con el brazo casi estirado la 'direccion del codo' es ruido: se atenua."""
    coseno = np.clip(np.dot(datos["dir_brazo"], datos["dir_antebrazo"]), -1.0, 1.0)
    angulo = float(np.arccos(coseno))
    lo, hi = np.radians(ANG_RECTO)
    if angulo <= lo:
        return 0.0
    x = np.clip((angulo - lo) / (hi - lo), 0.0, 1.0)
    return W_CODO * float(0.05 + 0.95 * x * x * (3.0 - 2.0 * x))


# ---------------------------------------------------------------- visor MuJoCo
def colocar_camara(v, robot, desde_detras=True):
    """Vista en tercera persona (desde detras) o de frente al robot."""
    hombros = (robot.brazos["L"].hombro + robot.brazos["R"].hombro) / 2
    delante = robot.R_base[:, 0]
    rumbo = float(np.degrees(np.arctan2(delante[1], delante[0])))
    with v.lock():
        v.cam.lookat[:] = hombros + 0.3 * delante - np.array([0.0, 0.0, 0.2])
        v.cam.distance = 1.8
        v.cam.azimuth = rumbo + (180.0 if desde_detras else 0.0)
        v.cam.elevation = -25.0


def dibujar_marcadores(escena, marcadores):
    import mujoco
    escena.ngeom = 0
    for pos, rgba in marcadores:
        if escena.ngeom >= escena.maxgeom:
            break
        mujoco.mjv_initGeom(escena.geoms[escena.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.03, 0, 0], pos=np.asarray(pos, dtype=float),
                            mat=np.eye(3).flatten(), rgba=np.asarray(rgba, dtype=np.float32))
        escena.ngeom += 1


# ---------------------------------------------------------------- ventana OpenCV
def texto(img, s, org, color=(255, 255, 255), escala=0.5):
    (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base + 1), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)


def partir(s, ancho_px, escala=0.5):
    """Parte un texto en lineas que quepan en ancho_px (por palabras)."""
    lineas, actual = [], ""
    for palabra in s.split(" "):
        prueba = (actual + " " + palabra).strip()
        if actual and cv2.getTextSize(prueba, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)[0][0] > ancho_px:
            lineas.append(actual)
            actual = "   " + palabra
        else:
            actual = prueba if not actual.startswith("   ") else actual + " " + palabra
    return lineas + ([actual] if actual else [])


def colorear_profundidad(depth):
    d = np.clip(depth.astype(np.float32), 300, 3000)
    img = cv2.applyColorMap(255 - ((d - 300) / 2700 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[depth == 0] = 0  # sin dato = negro
    return img


def textos_hud(snap, estado, dos_ventanas=True):
    """Reparte los textos del HUD entre las dos ventanas. Devuelve {"A": (lineas, rojas),
    "B": (lineas, rojas)}.
      A (la principal): estado del teleop, lo que se ve del operador, error del robot,
        mensajes de las teclas (calibrando, reposo...) y avisos de robot QUIETO.
      B: camaras y fusion, longitudes del operador, escenario y requisitos (avisos rojos).
    Con dos_ventanas=False todo va a A (como antes)."""
    snap = snap or {}
    brazos, aperturas = snap.get("brazos") or {}, snap.get("aperturas") or {}
    a_lin, a_roj, b_lin, b_roj = [], [], [], []
    a_lin.append(f"Camara {estado.get('fps_cam', 0.0):3.0f} fps | seguimiento {estado['fps_seg']:4.1f} fps | robot {estado['fps_control']:3.0f} Hz | "
                 f"{'PAUSA' if estado['pausado'] else 'TELEOP'} | {'espejo' if estado['espejo'] else 'directo'} | "
                 f"{'calibrado' if estado['calibrado'] else 'sin calibrar'}"
                 + (f" | dedos: {estado['modo_dedos']}" if estado.get("modo_dedos") else ""))
    fu = snap.get("fusion")
    if fu and fu.get("n_camaras", 1) > 1:
        b_lin.append(f"Camaras: {fu['estado']}")
        des = " ".join(f"{'IZQ' if l == 'L' else 'DER'} {g:3.0f} deg" for l, g in sorted(fu["desacuerdo"].items()))
        b_lin.append(f"Fusion {fu.get('modo', '')} {'+'.join(fu['activas']) or '--'}"
                     + (f" | desacuerdo entre camaras: {des}" if des else "")
                     + "".join(f" | {c}: izq/der corregido" for c in fu.get("cruzadas", [])))
    if fu:
        b_lin += fu.get("avisos", [])
    b_lin.append(f"Operador: {LONGITUDES.texto()}")
    for lado in ("L", "R"):
        nombre = "IZQ" if lado == "L" else "DER"
        mano = aperturas.get(lado)
        if mano:
            extra = f" | mano {mano['apertura'] * 100:3.0f}%"
            if mano.get("bruto") is not None:
                extra += f" (bruto {mano['bruto']:.2f})"
            if mano.get("fuente_mano"):
                extra += f" {mano['fuente_mano']}"
            if mano.get("calidad") is not None:
                extra += f" cal {mano['calidad']:.2f}"
        else:
            extra = " | mano no vista"
        a_lin.append(f"Operador {nombre}: "
                     f"{brazos[lado]['fuente'] if lado in brazos else 'no detectado'}{extra}")
    for lado in ("L", "R"):
        if lado in estado["errores"]:
            ec, em, eo = estado["errores"][lado]
            a_lin.append(f"Robot {lado}: error codo {ec * 1000:3.0f} mm | muneca {em * 1000:3.0f} mm | "
                         f"palma {np.degrees(eo):3.0f} deg"
                         + (" | orientacion: " + estado["orient"][lado] if estado["orient"].get(lado) else ""))
    if estado.get("escena"):
        b_lin.append(estado["escena"])
    if estado["mensaje"]:
        a_lin.append(estado["mensaje"])
    a_roj += [f"Robot {'IZQ' if lado == 'L' else 'DER'} QUIETO: {motivo}"
              for lado, motivo in sorted(estado.get("congelado", {}).items())]
    b_roj += [f"! {r}" for r in estado.get("requisitos", [])]
    if not dos_ventanas:
        return {"A": (a_lin + b_lin, a_roj + b_roj), "B": ([], [])}
    return {"A": (a_lin, a_roj), "B": (b_lin, b_roj)}


def pintar_textos(img, lineas, rojas, y0=22):
    """Lineas blancas y luego las rojas (partidas al ancho de la imagen), de arriba abajo."""
    w = img.shape[1]
    y = y0
    for s in lineas:
        texto(img, s, (10, y))
        y += 22
    for r in rojas:
        for s in partir(r, w - 20):
            texto(img, s, (10, y), (0, 0, 255))
            y += 22


def componer_vista(snap, estado, textos=None):
    from seguimiento_manos import dibujar_manos
    img = snap["bgr"].copy()
    lm2d, brazos, depth = snap["lm2d"], snap["brazos"], snap["depth"]
    dibujar_esqueleto(img, lm2d)
    dibujar_manos(img, snap.get("manos") or [])
    img = cv2.flip(img, 1)  # vista espejo: tu brazo izquierdo aparece a la izquierda
    h, w = img.shape[:2]

    if depth is not None:  # miniatura de la profundidad (arriba a la derecha)
        pequena = cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        mini = cv2.flip(colorear_profundidad(pequena), 1)
        img[8:8 + h // 4, w - 8 - w // 4:w - 8] = mini

    if lm2d is not None:  # etiquetas junto a las munecas
        for lado, idx in PUNTOS.items():
            u, v, vis = lm2d[idx[2]]
            if vis >= VIS_MIN:
                texto(img, "IZQ" if lado == "L" else "DER", (int(w - 1 - u) + 10, int(v)), COLOR_LADO[lado])

    for k, lado in enumerate(("L", "R")):   # barras de apertura de cada mano
        a = estado["pinzas"].get(lado)
        x0, y0 = 10 + 130 * k, h - 46
        cv2.rectangle(img, (x0, y0), (x0 + 110, y0 + 14), (0, 0, 0), -1)
        if a is not None:
            cv2.rectangle(img, (x0 + 1, y0 + 1), (x0 + 1 + int(108 * a), y0 + 13),
                          (int(60 * (1 - a)), int(70 + 160 * a), int(235 - 175 * a)), -1)
        etiqueta = "IZQ" if lado == "L" else "DER"
        texto(img, f"mano {etiqueta} {'--' if a is None else f'{a * 100:3.0f}%'}",
              (x0, y0 - 4), escala=0.42)

    if textos is None:      # una sola camara: todo en esta ventana
        textos = textos_hud(snap, estado, dos_ventanas=False)["A"]
    pintar_textos(img, *textos)
    texto(img, "q salir | p pausa | c calibrar | r reposo | m espejo | g dedos | v vista | 1 tuberias",
          (10, h - 12), escala=0.42)
    return img


def vista_fluida(hilo, snap):
    """Copia de 'snap' con la imagen EN BRUTO mas reciente de la camara y el esqueleto encima."""
    crudo = hilo.ultimo_crudo()
    if snap is None or crudo is None:
        return snap
    v = dict(snap)
    v["bgr"], v["depth"] = crudo["bgr"], crudo["depth"]
    return v


def componer_vista_secundaria(snap, fps_cam, fps, textos=None):
    from seguimiento_manos import dibujar_manos
    img = snap["bgr"].copy()
    dibujar_esqueleto(img, snap["lm2d"])
    dibujar_manos(img, snap.get("manos") or [])
    img = cv2.flip(img, 1)
    vistos = " ".join(f"{'IZQ' if l == 'L' else 'DER'}" for l in sorted(snap["brazos"])) or "nadie"
    texto(img, f"Camara B {fps_cam:3.0f} fps | seguimiento {fps:4.1f} fps | brazos vistos: {vistos}", (10, 22))
    if textos is not None:
        pintar_textos(img, *textos, y0=44)
    return img


# ---------------------------------------------------------------- calibracion
def aplicar_calibracion(res, fusion, estado):
    """Aplica el resultado de una CalibracionCamaras a la fusion. Devuelve el mensaje."""
    letras = "ABCD"
    for k, (R, n, disp, motivo, ang) in enumerate(res):
        print(f"[calibracion] camara {letras[k]}: {n} frames"
              + (f", dispersion {disp:.1f} deg" if disp is not None else "")
              + (f", gravedad (IMU) a {ang:.1f} deg de los brazos" if ang is not None else "")
              + (f", rechazos sobre todo por: {motivo}" if motivo else ""))
    if all(R is None for R, *_ in res):
        motivos = ", ".join(f"{letras[k]}: {m}" for k, (_, _, _, m, _) in enumerate(res))
        return f"Calibracion fallida ({motivos}): quieto, brazos colgando"
    info = fusion.fijar_calibracion([R for R, *_ in res], [r[4] for r in res])
    estado["calibrado"] = True
    ok = [k for k, r in enumerate(res) if r[0] is not None]
    disp = max(res[k][2] for k in ok)
    aviso = " | dispersion alta: repite quieto" if disp > 3.0 else ""
    if len(res) == 1:
        return f"Calibrado (dispersion {disp:.1f} deg){aviso}"
    partes = [f"Calibrado con {'+'.join(letras[k] for k in ok)}"]
    if info["conservada"]:
        partes.append("relacion entre camaras conservada")
    if info["discrepancia"] is not None:
        partes.append(f"las camaras coinciden a {info['discrepancia']:.1f} deg")
    falta = [letras[k] for k in range(len(res)) if not fusion.valida[k]]
    if falta:
        partes.append(f"{'+'.join(falta)} se calibrara sola: mueve los brazos")
    return " | ".join(partes) + f" | dispersion {disp:.1f} deg{aviso}"


# ---------------------------------------------------------------- teleoperador
class Teleoperador:
    """Un ciclo de teleoperacion = paso(t, dt): fusion -> objetivos -> IK -> manos -> escena.
    No toca el visor ni las ventanas: lo usan main() (en tiempo real) y
    metricas_imitacion.py (sin visor, a toda velocidad, con reloj simulado)."""

    def __init__(self, robot, fusion, escena, hilos, girar=True, grabadora=None, metricas=None,
                 reloj=time.monotonic, log_csv=None, eventos=None, motivo_sin_extrinsecas=None):
        self.robot, self.fusion, self.escena, self.hilos = robot, fusion, escena, hilos
        self.girar, self.grabadora, self.metricas, self.reloj = girar, grabadora, metricas, reloj
        self.R_cam = fusion.R_cam
        self.estado = dict(fps_seg=0.0, fps_control=0.0, pausado=False, espejo=False, calibrado=False,
                           errores={}, mensaje="", pinzas={}, orient={}, congelado={}, modo_dedos="")
        if robot.manos:
            self.estado["modo_dedos"] = "imitacion" if not next(iter(robot.manos.values())).estado["pinza_habilitada"] else "pinzas"
        self.suaves = {}          # lado del robot -> objetivos suavizados {codo, muneca, w, R}
        self.calib = None         # CalibracionCamaras en curso (tecla c)
        self.t_mensaje = 0.0
        self.n_mano = -1
        self.ult_obj, self.n_log = {}, {}
        self.t_sin_brazos, self.t_diag = None, -1e9
        self.congelado_ant, self.t_lejos = {}, {}
        self.eventos = sorted(eventos or [], key=lambda e: e[0])   # reproduccion: (t, nombre, datos)
        self.gancho = None
        self.t_log0 = reloj()
        # condiciones de arranque/uso que se dicen en pantalla (requisitos.py)
        self.requisitos = Requisitos(hilos, fusion, reloj=reloj, extrinsecas=any(fusion._fija),
                                     motivo_extrinsecas=motivo_sin_extrinsecas)
        self.t_resumen = reloj() + 8.0
        self._f_csv = open(log_csv, "w", newline="", encoding="utf-8") if log_csv else None
        self._w_csv = csv.writer(self._f_csv) if self._f_csv else None
        if self._w_csv:
            self._w_csv.writerow(["t_s", "lado_robot", "foto", "camaras", "modo", "fuente", "desacuerdo_deg", "cruzadas",
                                  "codo_x", "codo_y", "codo_z", "muneca_x", "muneca_y", "muneca_z",
                                  "salto_m", "err_codo_m", "err_muneca_m", "err_palma_deg", "fuente_mano"])

    # ------------------------------------------------------------ eventos
    def _evento(self, nombre, **datos):
        if self.grabadora is not None:
            self.grabadora.evento(nombre, **datos)

    def _aplicar_eventos(self, t0):
        """Reproduccion: aplica los eventos grabados cuyo instante ya ha llegado."""
        while self.eventos and self.eventos[0][0] <= t0:
            _, nombre, d = self.eventos.pop(0)
            if nombre == "calibracion":
                res = [(None if R is None else np.asarray(R), n, disp, mot, ang)
                       for R, n, disp, mot, ang in d["resultado"]]
                self.estado["mensaje"] = aplicar_calibracion(res, self.fusion, self.estado)
                self.ult_obj.clear()
                self.t_mensaje = t0 + 4.0
            elif nombre == "pausa":
                self.estado["pausado"] = bool(d["pausado"])
                self.ult_obj.clear()
            elif nombre == "espejo":
                self.estado["espejo"] = bool(d["espejo"])
                self._reiniciar_objetivos()
            elif nombre == "reposo":
                self.reposo()
            elif nombre == "modo_dedos":
                self.modo_dedos(bool(d["pinzas"]))

    # ------------------------------------------------------------ acciones (teclas)
    def _reiniciar_objetivos(self):
        self.suaves.clear()
        self.estado["errores"].clear()
        self.estado["pinzas"].clear()
        self.estado["orient"].clear()

    def reposo(self):
        self.robot.reposo()
        self._reiniciar_objetivos()

    def modo_dedos(self, pinzas):
        for mano in self.robot.manos.values():
            mano.modo_pinza(pinzas)
        self.estado["modo_dedos"] = "pinzas" if pinzas else "imitacion"

    def tecla(self, tecla, t0, visor=None):
        """Devuelve True si hay que salir."""
        if tecla in (ord("q"), 27):
            return True
        if tecla == ord("p"):
            self.estado["pausado"] = not self.estado["pausado"]
            self.ult_obj.clear()     # el registro de saltos no compara con antes de la pausa
            self._evento("pausa", pausado=self.estado["pausado"])
        elif tecla == ord("m"):
            self.estado["espejo"] = not self.estado["espejo"]
            self._reiniciar_objetivos()
            self._evento("espejo", espejo=self.estado["espejo"])
        elif tecla == ord("g") and self.robot.manos:
            self.modo_dedos(self.estado["modo_dedos"] != "pinzas")
            self._evento("modo_dedos", pinzas=self.estado["modo_dedos"] == "pinzas")
            self.estado["mensaje"] = f"Dedos: {self.estado['modo_dedos']}"
            self.t_mensaje = t0 + 2.0
        elif tecla == ord("1") and self.escena is not None:
            self.estado["mensaje"] = ("Escenario de tuberias reiniciado" if self.escena.activa
                                      else "Escenario de tuberias")
            self.escena.activar(visor)
            self.t_mensaje = t0 + 2.0
        elif tecla == ord("r"):
            self.reposo()
            self._evento("reposo")
        elif tecla == ord("c"):
            self.calib = CalibracionCamaras(self.hilos, fusion=self.fusion, reloj=self.reloj)
        return False

    # ------------------------------------------------------------ ciclo
    def paso(self, t0, dt):
        """Un ciclo del bucle de control. Devuelve (snap, marcadores para el visor)."""
        estado, fusion, robot, escena = self.estado, self.fusion, self.robot, self.escena
        self._aplicar_eventos(t0)
        if self.calib is not None:
            self.calib.alimentar()
            estado["mensaje"] = f"Calibrando: quieto, brazos colgando ({self.calib.restante():.1f} s)"
            self.t_mensaje = t0 + 0.5
            if self.calib.terminada():
                res = self.calib.resultado()
                estado["mensaje"] = aplicar_calibracion(res, fusion, estado)
                self.ult_obj.clear()     # marco nuevo: no es un salto
                self._evento("calibracion", resultado=[(None if R is None else np.asarray(R).tolist(), n, disp, mot, ang)
                                                       for R, n, disp, mot, ang in res])
                self.t_mensaje = t0 + 6.0
                self.calib = None

        snap = fusion.ultimo()
        if fusion.R_cam is not self.R_cam:   # calibracion nueva (o inclinacion de la IMU)
            self.R_cam = fusion.R_cam
        R_cam = self.R_cam
        fresco = snap is not None and t0 - snap["t"] < CADUCIDAD
        brazos = snap["brazos"] if fresco else {}
        aperturas = snap["aperturas"] if fresco else {}
        if brazos:
            self.t_sin_brazos = None
        elif self.t_sin_brazos is None:
            self.t_sin_brazos = t0
        elif t0 - self.t_sin_brazos > 2.0 and t0 - self.t_diag > 2.0:
            self.t_diag = t0
            print(f"[diagnostico] sin brazos desde hace {t0 - self.t_sin_brazos:.0f} s"
                  f"{' (sin datos frescos)' if not fresco else ''}:\n{fusion.diagnostico()}")

        # Diagnostico: por que se queda quieto cada brazo (se muestra en rojo)
        estado["congelado"] = {}
        if not estado["pausado"]:
            for lado_r in ("L", "R"):
                lado_h = OTRO_LADO[lado_r] if estado["espejo"] else lado_r
                err = estado["errores"].get(lado_r)
                if not fresco:
                    motivo = "sin datos nuevos de la camara"
                elif lado_h not in brazos:
                    motivo = f"no se ve tu brazo {'IZQ' if lado_h == 'L' else 'DER'}"
                elif err is not None and err[1] > UMBRAL_CONGELADO:
                    self.t_lejos[lado_r] = self.t_lejos.get(lado_r, 0.0) + dt
                    if self.t_lejos[lado_r] < 0.5:
                        continue
                    motivo = f"no llega al objetivo (error muneca {err[1] * 100:.0f} cm)"
                else:
                    self.t_lejos[lado_r] = 0.0
                    continue
                estado["congelado"][lado_r] = motivo
        if estado["congelado"] != self.congelado_ant:
            for lado_r, motivo in estado["congelado"].items():
                print(f"[{time.strftime('%H:%M:%S')}] robot {lado_r} quieto: {motivo}")
            self.congelado_ant = dict(estado["congelado"])

        marcadores = []
        if not estado["pausado"]:
            foto_nueva = fresco and snap["n"] != self.n_mano
            if foto_nueva:
                self.n_mano = snap["n"]
            # El objetivo se recalcula en CADA ciclo con la ultima medida y el muelle lo sigue:
            # movimiento continuo a 60 Hz en lugar de un salto por cada foto de la camara.
            for lado_h, datos in brazos.items():
                lado_r = OTRO_LADO[lado_h] if estado["espejo"] else lado_h
                codo, muneca, d_b, d_a = objetivos(robot, lado_r, datos, R_cam, estado["espejo"])
                w = peso_codo(datos)
                if lado_r not in self.suaves:
                    self.suaves[lado_r] = dict(
                        f_codo=ObjetivoSuave(OMEGA_OBJETIVO, ZONA_MUERTA_OBJ),
                        f_muneca=ObjetivoSuave(OMEGA_OBJETIVO, ZONA_MUERTA_OBJ),
                        f_w=ObjetivoSuave(OMEGA_OBJETIVO), f_R=RotacionSuave(OMEGA_ORIENT, SALTO_ORIENT),
                        R_h=None, fuente_mano=None)
                s = self.suaves[lado_r]
                s["codo"] = s["f_codo"].actualizar(codo, dt)
                s["muneca"] = s["f_muneca"].actualizar(muneca, dt)
                s["w"] = float(s["f_w"].actualizar(w, dt))
                s["d_b"], s["d_a"] = d_b, d_a
                # orientacion de la palma: del marco de la mano (una vez por foto)
                mano = aperturas.get(lado_h)
                if foto_nueva and self.girar and mano is not None:
                    R_h = marco_mano(robot, mano, R_cam, estado["espejo"])
                    cal = float(mano.get("calidad", 1.0))
                    fiable = R_h is not None and (cal >= CALIDAD_MIN_ORIENT or str(mano.get("fuente_mano", "")).startswith("3D"))
                    if fiable:
                        s["f_R"].fijar_objetivo(R_h)
                        s["R_h"], s["fuente_mano"] = R_h, mano.get("fuente_mano")
                        estado["orient"][lado_r] = f"{mano.get('fuente_mano', '')} cal {cal:.2f}"
                    elif R_h is not None:
                        estado["orient"][lado_r] = f"mano poco fiable (cal {cal:.2f}): se mantiene"
                if foto_nueva:
                    self._diagnostico_salto(snap, lado_r, lado_h, muneca, codo, datos, s)
            # un brazo que deja de verse mantiene su ultimo objetivo (el muelle termina de frenar)
            vistos = {OTRO_LADO[l] if estado["espejo"] else l for l in brazos}
            for lado_r, s in self.suaves.items():
                if lado_r not in vistos:
                    s["codo"] = s["f_codo"].actualizar(s["f_codo"].ancla, dt)
                    s["muneca"] = s["f_muneca"].actualizar(s["f_muneca"].ancla, dt)
            # dedos / pinzas: una vez por foto
            for lado_h, mano in aperturas.items():
                lado_r = OTRO_LADO[lado_h] if estado["espejo"] else lado_h
                brazo = robot.brazos[lado_r]
                if brazo.mano is not None:              # OrcaHand
                    if foto_nueva:
                        if mano.get("mundo") is not None:   # dedo a dedo
                            brazo.mano.nuevo_objetivo(mano["mundo"], espejar=(lado_h == "L"))
                        else:                               # solo abrir / cerrar
                            brazo.mano.objetivo_por_apertura(mano["apertura"])
                else:                                   # pinza original
                    minimo = escena.apertura_minima(lado_r) if escena else 0.0
                    brazo.mover_pinza(max(mano["apertura"], minimo), dt)
                estado["pinzas"][lado_r] = mano["apertura"]
            for lado_r, orca in robot.manos.items():
                sujeta = escena is not None and escena.activa and escena.sujeta == lado_r
                orca.mover(dt, mano_orca.TOPE_TUBO if sujeta else None)
            # IK: posicion (codo + muneca) y orientacion de la palma, suavizadas
            for lado_r, s in self.suaves.items():
                brazo = robot.brazos[lado_r]
                R_s = s["f_R"].actualizar(dt)
                estado["errores"][lado_r] = brazo.seguir(s["codo"], s["muneca"], dt, s["w"], R_obj=R_s)
                marcadores += [(s["codo"], COLOR_CODO), (s["muneca"], COLOR_MUNECA)]
                if self.metricas is not None and foto_nueva and lado_r in vistos:
                    lado_h = OTRO_LADO[lado_r] if estado["espejo"] else lado_r
                    self.metricas.registrar(t0, lado_r, s["d_b"], s["d_a"], s.get("R_h"), brazo,
                                            estado["errores"][lado_r], t0 - snap.get("t_captura", snap["t"]),
                                            snap["n"], brazos[lado_h]["fuente"], s.get("fuente_mano"))

        robot.actualizar()
        if escena is not None:          # la pieza sigue a la pinza, encaja o cae
            escena.actualizar(robot, estado["pinzas"], dt)
            robot.actualizar()
        estado["fps_control"] = 0.95 * estado["fps_control"] + 0.05 / max(dt, 1e-3)
        estado["requisitos"] = self.requisitos.actualizar(estado["fps_control"])
        if self.t_resumen is not None and t0 > self.t_resumen:
            self.t_resumen = None
            print(self.requisitos.resumen())
        if t0 > self.t_mensaje:
            estado["mensaje"] = ""
        if self.gancho is not None:       # pruebas: comparar con una verdad conocida
            self.gancho(t0, snap)
        return snap, marcadores

    def _diagnostico_salto(self, snap, lado_r, lado_h, muneca, codo, datos, s):
        fu_l = snap.get("fusion") or {}
        ant = self.ult_obj.get(lado_r)
        salto = 0.0 if ant is None else float(np.linalg.norm(muneca - ant))
        self.ult_obj[lado_r] = np.array(muneca, dtype=float)
        des_l = fu_l.get("desacuerdo", {}).get(lado_h, 0.0)
        if salto > SALTO_LOG:
            print(f"[SALTO {time.strftime('%H:%M:%S')}] muneca {lado_r} {salto * 100:.0f} cm en una foto | "
                  f"camaras {'+'.join(fu_l.get('activas', [])) or '--'} | modo {fu_l.get('modo', '?')} | "
                  f"fuente {datos['fuente']} | desacuerdo {des_l:.0f} deg")
        if self._w_csv:
            er = self.estado["errores"].get(lado_r, (np.nan, np.nan, np.nan))
            self._w_csv.writerow([f"{self.reloj() - self.t_log0:.3f}", lado_r, snap["n"],
                                  "+".join(fu_l.get("activas", [])), fu_l.get("modo", ""), datos["fuente"],
                                  f"{des_l:.1f}", ",".join(fu_l.get("cruzadas", [])),
                                  *[f"{x:.4f}" for x in codo], *[f"{x:.4f}" for x in muneca],
                                  f"{salto:.4f}", f"{er[0]:.4f}", f"{er[1]:.4f}", f"{np.degrees(er[2]):.1f}",
                                  s.get("fuente_mano") or ""])

    def cerrar(self):
        if self._f_csv:
            self._f_csv.close()
        if self.grabadora is not None:
            self.grabadora.guardar()

    # ------------------------------------------------------------ sin visor (metricas)
    def correr_sin_visor(self, frec=FREC_CONTROL):
        """Reproduccion a toda velocidad con reloj simulado (self.reloj debe ser un RelojSimulado)."""
        dt = 1.0 / frec
        while not all(h.terminado() for h in self.hilos):
            t0 = self.reloj()
            self.paso(t0, dt)
            self.reloj.avanzar(dt)


class RelojSimulado:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def avanzar(self, dt):
        self.t += dt


# ---------------------------------------------------------------- montaje
def crear_robot(xml, orca, reloj=time.monotonic):
    robot = RobotTron2(xml, escenario_tuberias.anadir_al_modelo, orca=orca, reloj=reloj)
    escena = escenario_tuberias.EscenarioTuberias(robot)
    robot.resumen()
    return robot, escena


def preparar_reproduccion(ruta, xml, orca, velocidad=1.0, metricas=None, sin_extrinsecas=False, reloj=None,
                          log_csv=None):
    """Teleoperador que reproduce una sesion grabada. Con reloj=None se usa un RelojSimulado
    (para metricas, a toda velocidad); con time.monotonic va en tiempo real x velocidad."""
    datos = grabacion.cargar(ruta)
    reloj_sim = reloj is None
    reloj = RelojSimulado() if reloj_sim else reloj
    robot, escena = crear_robot(xml, orca, reloj=reloj)
    hilos, eventos = grabacion.hilos_reproduccion(datos, reloj=reloj, velocidad=1.0 if reloj_sim else velocidad)
    meta = datos.get("meta", {})
    extr = None
    if len(hilos) > 1 and not sin_extrinsecas and meta.get("extrinsecas") is not None:
        extr = (np.asarray(meta["extrinsecas"][0]), np.asarray(meta["extrinsecas"][1]))
    fusion = FusionCamaras(hilos, reloj=reloj, extrinsecas=extr)
    print(f"[reproduccion] {ruta}: {len(hilos)} camara(s), {len(eventos)} eventos"
          + (", extrinsecas ChArUco" if extr is not None else ""))
    motivo = None
    if len(hilos) > 1 and extr is None:
        motivo = "--sin-extrinsecas" if sin_extrinsecas else "la grabacion no las tiene"
    return Teleoperador(robot, fusion, escena, hilos, girar=True, metricas=metricas, reloj=reloj, eventos=eventos,
                        log_csv=log_csv, motivo_sin_extrinsecas=motivo)


def main():
    ap = argparse.ArgumentParser(description="Teleoperacion de los brazos del TRON 2 con OAK-D")
    ap.add_argument("--xml", default="tron2a/DACH_TRON2A/xml/robot_elecnor.xml")
    ap.add_argument("--webcam", type=int, default=None, help="indice de webcam (en lugar de la OAK-D)")
    ap.add_argument("--solo-mediapipe", action="store_true", help="no usar la profundidad de la OAK-D")
    ap.add_argument("--modelo", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--sin-manos", action="store_true", help="no seguir las manos ni mover las pinzas")
    ap.add_argument("--sin-giro", action="store_true", help="no orientar la palma con la mano")
    ap.add_argument("--escenario", choices=["tuberias"], default=None)
    ap.add_argument("--orca", action="store_true", help="OrcaHand en lugar de las pinzas")
    ap.add_argument("--camara2", action="store_true", help="usar dos OAK-D y fusionar las vistas")
    ap.add_argument("--sin-imu", action="store_true", help="no usar la IMU de las OAK-D")
    ap.add_argument("--proyector", type=float, default=0.7, help="intensidad 0-1 del proyector IR (0 = apagado)")
    ap.add_argument("--sin-manos-b", action="store_true", help="con --camara2: la camara B no sigue las manos")
    ap.add_argument("--sin-estereo-fino", action="store_true")
    ap.add_argument("--sin-hd", action="store_true", help="no pedir el fotograma HD para las manos")
    ap.add_argument("--sin-recorte", action="store_true", help="Hand Landmarker sobre la imagen entera (como antes)")
    ap.add_argument("--extrinsecas", default=None, metavar="ARCHIVO")
    ap.add_argument("--sin-extrinsecas", action="store_true")
    ap.add_argument("--log-csv", default=None, metavar="ARCHIVO")
    ap.add_argument("--mxid", nargs=2, metavar=("ID_A", "ID_B"), default=None)
    ap.add_argument("--grabar", default=None, metavar="ARCHIVO", help="graba la sesion (grabacion.py)")
    ap.add_argument("--grabar-video", action="store_true", help="con --grabar: tambien los fotogramas (JPEG)")
    ap.add_argument("--reproducir", default=None, metavar="ARCHIVO", help="reproduce una sesion grabada, sin camaras")
    ap.add_argument("--velocidad", type=float, default=1.0, help="con --reproducir: veces el tiempo real")
    args = ap.parse_args()
    if args.camara2 and args.webcam is not None:
        ap.error("--camara2 es para dos OAK-D, no se combina con --webcam")

    import mujoco.viewer

    if args.reproducir:
        tele = preparar_reproduccion(args.reproducir, args.xml, args.orca, velocidad=args.velocidad,
                                     sin_extrinsecas=args.sin_extrinsecas, reloj=time.monotonic, log_csv=args.log_csv)
        hilos, camaras = tele.hilos, []
    else:
        from camaras import CamaraOAK, CamaraWebcam, listar_oak
        from seguimiento_brazos import HiloSeguimiento, SeguidorBrazos
        from seguimiento_manos import SeguidorManos
        robot, escena = crear_robot(args.xml, args.orca)
        ids = [None]
        if args.camara2:
            ids = list(args.mxid) if args.mxid else [i for i, _ in listar_oak()[:2]]
            if len(ids) < 2:
                raise SystemExit(f"--camara2 necesita dos OAK-D libres y solo veo {len(ids)} (python camaras.py --listar)")
            print("Camara A:", ids[0], "| Camara B:", ids[1])
        grabadora = None
        if args.grabar:
            grabadora = grabacion.Grabadora(args.grabar, con_video=args.grabar_video,
                                            meta=dict(camaras=ids, orca=args.orca, xml=args.xml))
        camaras, hilos = [], []
        for k_cam, ident in enumerate(ids):   # una instancia COMPLETA de seguimiento por camara
            seg_k = SeguidorBrazos(args.modelo, usar_profundidad=not args.solo_mediapipe)
            sin_manos_k = args.sin_manos or (args.sin_manos_b and k_cam > 0)
            manos_k = None if sin_manos_k else SeguidorManos(recorte=not args.sin_recorte)
            cam_k = (CamaraWebcam(args.webcam) if args.webcam is not None else
                     CamaraOAK(mxid=ident, proyector=args.proyector, usar_imu=not args.sin_imu,
                               estereo_fino=not args.sin_estereo_fino, hd=not args.sin_hd))
            hilo_k = HiloSeguimiento(cam_k, seg_k, manos_k, grabadora=grabadora, etiqueta="ABCD"[k_cam])
            camaras.append(cam_k)
            hilos.append(hilo_k)
        for hilo_k in hilos:
            hilo_k.start()
        extr = None
        if args.camara2 and not args.sin_extrinsecas:
            extr = cargar_extrinsecas(ids[0], ids[1], args.extrinsecas)
            if extr is not None and grabadora is not None:
                grabadora.meta["extrinsecas"] = (np.asarray(extr[0]).tolist(), np.asarray(extr[1]).tolist())
        fusion = FusionCamaras(hilos, extrinsecas=None if extr is None else (extr[0], extr[1]))
        motivo = None
        if args.camara2 and extr is None:
            motivo = "--sin-extrinsecas" if args.sin_extrinsecas else fusion_mod.motivo_sin_extrinsecas
        tele = Teleoperador(robot, fusion, escena, hilos, girar=not (args.sin_manos or args.sin_giro),
                            grabadora=grabadora, log_csv=args.log_csv, motivo_sin_extrinsecas=motivo)

    robot, escena, estado = tele.robot, tele.escena, tele.estado
    perfil, t_perfil = {}, time.monotonic()
    t_sync, t_ventana, t_ventana2 = -1e9, -1e9, -1e9
    desde_detras = True
    n_mostrado, n_mostrado2, t_aviso_negra = -1, -1, -1e9
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    if len(hilos) > 1:
        cv2.namedWindow(VENTANA2, cv2.WINDOW_NORMAL)

    try:
        with mujoco.viewer.launch_passive(robot.m, robot.d, show_left_ui=False, show_right_ui=False) as v:
            colocar_camara(v, robot, desde_detras)
            with v.lock():
                v.opt.geomgroup[escenario_tuberias.GRUPO_ESCENARIO] = 0
            if args.escenario == "tuberias":
                escena.activar(v)
            t_ant = time.monotonic()
            while v.is_running():
                t0 = time.monotonic()
                dt = min(t0 - t_ant, 0.1)
                t_ant = t0
                if tele.fusion.error:
                    print("Error en el seguimiento:", tele.fusion.error)
                    break
                if args.reproducir and all(h.terminado() for h in hilos):
                    print("[reproduccion] fin de la sesion")
                    break

                t_s = time.perf_counter()
                snap, marcadores = tele.paso(t0, dt)
                perfil["teleop"] = perfil.get("teleop", 0.0) + time.perf_counter() - t_s

                t_s = time.perf_counter()
                if t0 - t_sync >= 1.0 / HZ_VISOR:
                    t_sync = t0
                    with v.lock():
                        dibujar_marcadores(v.user_scn, marcadores)
                    v.sync()
                perfil["mujoco visor"] = perfil.get("mujoco visor", 0.0) + time.perf_counter() - t_s
                t_s = time.perf_counter()

                # Dos ventanas, cada una con su ritmo (si las dos tocan, la mas atrasada)
                crudo0 = hilos[0].ultimo_crudo()
                n_vista = -1 if crudo0 is None else crudo0["n"]
                crudo1 = hilos[1].ultimo_crudo() if len(hilos) > 1 else None
                toca1 = snap is not None and n_vista != n_mostrado and t0 - t_ventana >= 1.0 / HZ_VENTANA
                toca2 = (crudo1 is not None and crudo1["n"] != n_mostrado2
                         and t0 - t_ventana2 >= 1.0 / HZ_VENTANA2)
                if toca1 and toca2:
                    toca1 = (t0 - t_ventana) * HZ_VENTANA >= (t0 - t_ventana2) * HZ_VENTANA2
                    toca2 = not toca1
                if toca1:
                    n_mostrado = n_vista
                    t_ventana = t0
                    estado["fps_seg"] = tele.fusion.fps
                    estado["fps_cam"] = hilos[0].fps_camara
                    estado["escena"] = escena.texto() if escena else ""
                    fu_v = snap.get("fusion")
                    base_es_a = (not fu_v) or fu_v["activas"][:1] in ([], ["A"])
                    vista = vista_fluida(hilos[0], snap) if base_es_a else snap
                    textos = textos_hud(snap, estado, dos_ventanas=len(hilos) > 1)
                    cv2.imshow(VENTANA, componer_vista(vista, estado, textos["A"]))
                elif toca2:
                    n_mostrado2 = crudo1["n"]
                    t_ventana2 = t0
                    snap_b = hilos[1].ultimo()
                    if snap_b is not None:
                        snap_b = vista_fluida(hilos[1], snap_b)
                    else:
                        snap_b = dict(bgr=crudo1["bgr"], depth=None, lm2d=None, manos=[], brazos={})
                    if crudo1["bgr"].max() == 0 and not args.reproducir and t0 - t_aviso_negra > 5.0:
                        t_aviso_negra = t0
                        print("[camara B] llegan fotogramas completamente NEGROS: revisa el cable USB / "
                              "que no la use otro programa")
                    textos = textos_hud(snap, estado)
                    cv2.imshow(VENTANA2, componer_vista_secundaria(snap_b, hilos[1].fps_camara, hilos[1].fps,
                                                                   textos["B"]))

                tecla = cv2.waitKey(1) & 0xFF
                perfil["ventanas"] = perfil.get("ventanas", 0.0) + time.perf_counter() - t_s
                perfil["ciclos"] = perfil.get("ciclos", 0) + 1
                if t0 - t_perfil > 10.0:
                    n_c = max(perfil.pop("ciclos"), 1)
                    hz = n_c / (t0 - t_perfil)
                    if hz < 45:
                        print(f"[rendimiento] bucle del robot a {hz:.0f} Hz (deberia ir a {FREC_CONTROL:.0f}). ms por ciclo: "
                              + " | ".join(f"{k} {v * 1000 / n_c:.1f}" for k, v in perfil.items())
                              + f" | camaras {' / '.join(f'{h.fps:.0f}' for h in hilos)} fps")
                    perfil, t_perfil = {}, t0
                if tecla != 255 and tele.tecla(tecla, t0, visor=v):
                    break
                if tecla == ord("v"):
                    desde_detras = not desde_detras
                    colocar_camara(v, robot, desde_detras)
                if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1 and n_mostrado >= 0:
                    break

                espera = 1.0 / FREC_CONTROL - (time.monotonic() - t0)
                if espera > 0:
                    time.sleep(espera)
    finally:
        tele.cerrar()
        for hilo_k in hilos:
            hilo_k.parar()
        for cam_k in camaras:
            cam_k.cerrar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()