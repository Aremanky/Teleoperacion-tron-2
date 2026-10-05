"""
Teleoperacion de los brazos del TRON 2 (DACH_TRON2A) en MuJoCo con una OAK-D.

Uso (desde la raiz de tron2-robot-description, con estos 4 .py copiados alli):
  python teleop_tron2.py                     # OAK-D: RGB + profundidad
  python teleop_tron2.py --solo-mediapipe    # OAK-D sin usar la profundidad
  python teleop_tron2.py --webcam 0          # sin OAK-D, con una webcam normal
  python teleop_tron2.py --modelo lite       # MediaPipe mas rapido (menos preciso)
  python teleop_tron2.py --sin-manos         # no seguir las manos (va mas rapido)
  python teleop_tron2.py --sin-giro          # manos solo para abrir/cerrar, sin girar la muneca
  python teleop_tron2.py --escenario tuberias  # arranca ya con el escenario de tuberias a la vista
  python teleop_tron2.py --orca              # OrcaHand en lugar de las pinzas (ver mano_orca.py)
  python teleop_tron2.py --camara2           # DOS OAK-D: fusiona las dos vistas (ver fusion.py)
  python teleop_tron2.py --camara2 --mxid ID_A ID_B   # idem, fijando cual es cual (python camaras.py --listar)
  python teleop_tron2.py --proyector 0       # sin el proyector IR de las OAK-D Pro (por defecto al 70 %)

Las pinzas siguen a las manos: palma abierta abre la pinza, puno cerrado la
cierra, y las posiciones intermedias se reproducen de forma proporcional.
Ademas, al girar la mano sobre el eje del antebrazo (como al girar un pomo)
gira la pinza. Doblar la muneca arriba/abajo o de lado NO se copia: la muneca
del robot se queda recta y solo gira, que es mas facil de manejar.
Cada mano manda sobre la pinza de su lado (o la contraria en modo espejo).

Teclas (con la ventana de la camara seleccionada):
  q / ESC  salir
  p        pausar / reanudar (embrague: congela el robot mientras te recolocas)
  c        calibrar: de pie, QUIETO 1.5 s, brazos relajados hacia abajo. Orienta al
           operador respecto al robot. Flujo recomendado: p (pausa) -> r (reposo) ->
           colocarse -> c -> p. Con --camara2 basta con que te vea UNA camara: la
           relacion entre camaras (que se calibra sola mientras te mueves, tambien
           en pausa) se conserva; si te ven varias se muestra cuanto discrepan.
           Si mueves una camara, se recalibra y se vuelve a situar sola
  m        modo espejo on/off (tu brazo izquierdo mueve el derecho del robot)
  o        recentrar el giro de las munecas (la postura actual de tu mano pasa a ser la neutra)
  v        ver el robot desde detras / desde delante
  r        devolver los brazos a la postura de reposo
  1        escenario de tuberias: la primera vez lo hace aparecer; despues lo reinicia

Estructura: un hilo lee la camara y ejecuta MediaPipe (a lo que de el PC, ~12-25 fps)
y el bucle principal mueve el robot a 60 Hz, suavizando los objetivos entre frames.
"""
import argparse
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from camaras import CamaraOAK, CamaraWebcam, listar_oak
import escenario_tuberias
from fusion import CalibracionCamaras, FusionCamaras
from robot_tron2 import RobotTron2, angulo_giro, ortonormalizar
from seguimiento_brazos import (COLOR_LADO, PUNTOS, VIS_MIN, HiloSeguimiento,
                                SeguidorBrazos, dibujar_esqueleto)
from seguimiento_manos import SeguidorManos, dibujar_manos
from suavizado import ObjetivoSuave, comprimir_alcance, suavizar

VENTANA = "Seguimiento OAK-D"
VENTANA2 = "Seguimiento OAK-D (camara 2)"

# R_cam (ejes del robot en el marco de la camara) lo lleva ahora FusionCamaras
# (fusion.R_cam): arranca con el de por defecto (camara horizontal, operador de
# frente; con IMU, corregido con la inclinacion real) y la tecla 'c' lo recalcula.

FREC_CONTROL = 60.0        # Hz del bucle del robot
TAU_OBJETIVO = 0.06        # (ya no se usa: ver OMEGA_OBJETIVO)
# Suavizado CONTINUO de objetivos (suavizado.py). La camara da un objetivo nuevo ~12 veces/s y el
# robot se mueve a 60 Hz: un filtro exponencial sobre esa escalera hacia que el robot acelerase y
# frenase en cada foto ("trompicones"). Un muelle criticamente amortiguado da posicion y velocidad
# continuas. Constante de tiempo ~ 2/omega: 14 -> ~0.14 s (mas alto = mas rapido pero menos suave).
OMEGA_OBJETIVO = 14.0
OMEGA_GIRO = 12.0          # idem para el giro de la muneca
ZONA_MUERTA_OBJ = 0.004    # m: el objetivo no se mueve por temblores menores que esto (0 = sin zona muerta)
ALCANCE_SUAVE, ALCANCE_MAX = 0.92, 0.97   # fraccion del alcance del brazo donde empieza a comprimirse
                           # el objetivo y donde nunca pasa: con el brazo del todo estirado la IK esta en una
                           # singularidad (tiembla y pierde precision); asi se queda un poco antes
CADUCIDAD = 0.5            # s: si no hay datos nuevos en este tiempo, el robot se queda quieto
W_CODO = 0.6               # peso del objetivo del codo frente al de la muneca
ANG_RECTO = (10.0, 30.0)   # grados: con el brazo casi recto se ignora el codo (evita giros locos)
TAU_GIRO = 0.12            # s: suavizado de la orientacion (mas alto = mas suave, mas retraso)
ESPEJO_M = np.diag([1.0, -1.0, 1.0])   # reflexion izquierda-derecha en el marco del robot
ZONA_MUERTA_CODO = (5.0, 25.0)  # grados: flexiones menores se tratan como brazo recto (se atenuan)
UMBRAL_CONGELADO = 0.08    # m: error de muneca a partir del cual se avisa de que el robot no llega
HZ_VISOR = 30.0            # el visor de MuJoCo se refresca a esto (el control del robot va a FREC_CONTROL)
HZ_VENTANA = 30.0          # ventana de la camara (antes 10: a eso se debian los "trompicones" de la imagen)
HZ_VENTANA2 = 15.0         # (antes 5)
MIN_PX_GIRO = 55           # px: con la mano mas pequena en la imagen su orientacion es ruido:
                           # el giro de la muneca se queda como estaba
SALTO_GIRO = np.radians(50)  # entre dos fotos la mano no gira tanto: es un error de MediaPipe
                             # (palma vista de canto); se ignora salvo que se repita 3 veces

COLOR_CODO = [1.0, 0.85, 0.0, 0.8]    # marcador amarillo en MuJoCo
COLOR_MUNECA = [0.1, 1.0, 0.3, 0.8]   # marcador verde en MuJoCo
OTRO_LADO = {"L": "R", "R": "L"}


# ---------------------------------------------------------------- retargeting
# (calibrar() vive ahora en fusion.py, junto a la calibracion promediada de las camaras)
def enderezar(d_b, d_a):
    """Zona muerta en la flexion del codo. Con el brazo casi estirado, el ruido del
    seguimiento se convierte en pequenas flexiones en direcciones aleatorias, y el
    robot tiene que girar mucho el hombro para seguirlas (singularidad). Por debajo
    de 5 grados el brazo se considera recto y hasta 25 se atenua suavemente."""
    theta = float(np.arccos(np.clip(np.dot(d_b, d_a), -1.0, 1.0)))
    s = np.sin(theta)
    if s < 1e-6:
        return d_a
    lo, hi = np.radians(ZONA_MUERTA_CODO)
    x = np.clip((theta - lo) / (hi - lo), 0.0, 1.0)
    theta_n = theta * x * x * (3 - 2 * x)
    return (np.sin(theta - theta_n) * d_b + np.sin(theta_n) * d_a) / s


def objetivos(robot, lado_robot, datos, R_cam, espejo):
    """Direcciones del operador -> posiciones objetivo del codo y la muneca del robot."""
    brazo = robot.brazos[lado_robot]
    d_b = R_cam @ datos["dir_brazo"]
    d_a = R_cam @ enderezar(datos["dir_brazo"], datos["dir_antebrazo"])
    if espejo:
        d_b[1] *= -1
        d_a[1] *= -1
    d_b, d_a = robot.R_base @ d_b, robot.R_base @ d_a
    codo = brazo.hombro + brazo.L_brazo * d_b
    muneca = codo + brazo.L_antebrazo * d_a
    # brazo casi estirado = singularidad de la IK: se acorta un poco (suave, sin escalones)
    muneca = comprimir_alcance(brazo.hombro, muneca, brazo.L_brazo + brazo.L_antebrazo,
                               ALCANCE_SUAVE, ALCANCE_MAX)
    return codo, muneca


def marco_mano(robot, datos, R_cam, espejo):
    """Orientacion de la mano del operador llevada al marco del robot.
    En modo espejo se refleja en el plano de simetria y se le devuelve la
    orientacion correcta invirtiendo tambien el eje transversal de la palma."""
    R = robot.R_base @ R_cam @ datos["marco"]
    if espejo:
        R = ESPEJO_M @ R @ np.diag([1.0, -1.0, 1.0])
    return ortonormalizar(R)


def peso_codo(datos):
    """Con el brazo casi estirado la 'direccion del codo' es puro ruido: se le quita peso."""
    coseno = np.clip(np.dot(datos["dir_brazo"], datos["dir_antebrazo"]), -1.0, 1.0)
    angulo = np.degrees(np.arccos(coseno))
    return W_CODO * float(np.clip((angulo - ANG_RECTO[0]) / (ANG_RECTO[1] - ANG_RECTO[0]), 0.0, 1.0))


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


def colorear_profundidad(depth):
    d = np.clip(depth.astype(np.float32), 300, 3000)
    img = cv2.applyColorMap(255 - ((d - 300) / 2700 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[depth == 0] = 0  # sin dato = negro
    return img


def componer_vista(snap, estado):
    img = snap["bgr"].copy()
    lm2d, brazos, depth = snap["lm2d"], snap["brazos"], snap["depth"]
    dibujar_esqueleto(img, lm2d)
    dibujar_manos(img, snap.get("manos") or [])
    img = cv2.flip(img, 1)  # vista espejo: tu brazo izquierdo aparece a la izquierda
    h, w = img.shape[:2]

    if depth is not None:  # miniatura de la profundidad (arriba a la derecha)
        # primero se reduce y despues se colorea: 16 veces menos pixeles que antes
        pequena = cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        mini = cv2.flip(colorear_profundidad(pequena), 1)
        img[8:8 + h // 4, w - 8 - w // 4:w - 8] = mini

    if lm2d is not None:  # etiquetas junto a las munecas
        for lado, idx in PUNTOS.items():
            u, v, vis = lm2d[idx[2]]
            if vis >= VIS_MIN:
                texto(img, "IZQ" if lado == "L" else "DER", (int(w - 1 - u) + 10, int(v)), COLOR_LADO[lado])

    for k, lado in enumerate(("L", "R")):   # barras de apertura de cada pinza
        a = estado["pinzas"].get(lado)
        x0, y0 = 10 + 130 * k, h - 46
        cv2.rectangle(img, (x0, y0), (x0 + 110, y0 + 14), (0, 0, 0), -1)
        if a is not None:
            cv2.rectangle(img, (x0 + 1, y0 + 1), (x0 + 1 + int(108 * a), y0 + 13),
                          (int(60 * (1 - a)), int(70 + 160 * a), int(235 - 175 * a)), -1)
        etiqueta = "IZQ" if lado == "L" else "DER"
        texto(img, f"pinza {etiqueta} {'--' if a is None else f'{a * 100:3.0f}%'}",
              (x0, y0 - 4), escala=0.42)

    lineas = [f"Camara {estado.get('fps_cam', 0.0):3.0f} fps | seguimiento {estado['fps_seg']:4.1f} fps | robot {estado['fps_control']:3.0f} Hz | "
              f"{'PAUSA' if estado['pausado'] else 'TELEOP'} | {'espejo' if estado['espejo'] else 'directo'} | "
              f"{'calibrado' if estado['calibrado'] else 'sin calibrar'}"]
    fu = snap.get("fusion")
    varias = bool(fu) and fu.get("n_camaras", 1) > 1
    if varias:
        lineas.append(f"Camaras: {fu['estado']}")
        des = " ".join(f"{'IZQ' if l == 'L' else 'DER'} {g:3.0f} deg" for l, g in sorted(fu["desacuerdo"].items()))
        lineas.append(f"Fusion {fu.get('modo', '')} {'+'.join(fu['activas']) or '--'}"
                      + (f" | desacuerdo entre camaras: {des}" if des else "")
                      + "".join(f" | {c}: izq/der corregido" for c in fu.get("cruzadas", [])))
    if fu:
        lineas += fu.get("avisos", [])
    for lado in ("L", "R"):
        nombre = "IZQ" if lado == "L" else "DER"
        mano = snap["aperturas"].get(lado)
        if mano:
            extra = f" | mano {mano['apertura'] * 100:3.0f}%"
            if mano.get("bruto") is not None:
                extra += f" (bruto {mano['bruto']:.2f})"
            if varias and mano.get("cam"):
                extra += f" cam {mano['cam']}"
        else:
            extra = " | mano no vista"
        lineas.append(f"Operador {nombre}: "
                      f"{brazos[lado]['fuente'] if lado in brazos else 'no detectado'}{extra}")
    for lado in ("L", "R"):
        if lado in estado["errores"]:
            ec, em, _ = estado["errores"][lado]
            giro = estado["giros"].get(lado)
            lineas.append(f"Robot {lado}: error codo {ec * 1000:3.0f} mm | muneca {em * 1000:3.0f} mm"
                          + (f" | giro muneca {np.degrees(giro):+4.0f} deg" if giro is not None else ""))
    if estado.get("escena"):
        lineas.append(estado["escena"])
    if estado["mensaje"]:
        lineas.append(estado["mensaje"])
    for k, s in enumerate(lineas):
        texto(img, s, (10, 22 + 22 * k))
    # avisos en ROJO: por que un brazo del robot se ha quedado quieto
    for k, (lado, motivo) in enumerate(sorted(estado.get("congelado", {}).items())):
        texto(img, f"Robot {'IZQ' if lado == 'L' else 'DER'} QUIETO: {motivo}",
              (10, 22 + 22 * (len(lineas) + k)), (0, 0, 255))
    texto(img, "q salir | p pausa | c calibrar | o recentrar giro | m espejo | v vista | r reposo | 1 tuberias",
          (10, h - 12), escala=0.42)
    return img


def vista_fluida(hilo, snap):
    """Copia de 'snap' con la imagen EN BRUTO mas reciente de la camara (a 30 fps) y el
    esqueleto de MediaPipe encima (el ultimo que haya, a ~12 fps). Asi la ventana va fluida
    aunque el seguimiento vaya mas lento."""
    crudo = hilo.ultimo_crudo()
    if snap is None or crudo is None:
        return snap
    v = dict(snap)
    v["bgr"], v["depth"] = crudo["bgr"], crudo["depth"]
    return v


def componer_vista_secundaria(snap, fps_cam, fps):
    """Segunda camara: imagen con el esqueleto y las manos detectadas, nada mas."""
    img = snap["bgr"].copy()
    dibujar_esqueleto(img, snap["lm2d"])
    dibujar_manos(img, snap.get("manos") or [])
    img = cv2.flip(img, 1)
    vistos = " ".join(f"{'IZQ' if l == 'L' else 'DER'}" for l in sorted(snap["brazos"])) or "nadie"
    texto(img, f"Camara B {fps_cam:3.0f} fps | seguimiento {fps:4.1f} fps | brazos vistos: {vistos}", (10, 22))
    return img


# ---------------------------------------------------------------- calibracion
def aplicar_calibracion(calib, fusion, estado):
    """Aplica el resultado de una CalibracionCamaras a la fusion. Devuelve el mensaje.
    Basta con que te vea UNA camara: la relacion entre camaras se conserva."""
    res = calib.resultado()
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
        print(f"[calibracion] discrepancia entre camaras: {info['discrepancia']:.1f} deg")
    falta = [letras[k] for k in range(len(res)) if not fusion.valida[k]]
    if falta:
        partes.append(f"{'+'.join(falta)} se calibrara sola: mueve los brazos")
    return " | ".join(partes) + f" | dispersion {disp:.1f} deg{aviso}"


# ---------------------------------------------------------------- bucle principal
def main():
    ap = argparse.ArgumentParser(description="Teleoperacion de los brazos del TRON 2 con OAK-D")
    ap.add_argument("--xml", default="tron2a/DACH_TRON2A/xml/robot_elecnor.xml")
    ap.add_argument("--webcam", type=int, default=None, help="indice de webcam (en lugar de la OAK-D)")
    ap.add_argument("--solo-mediapipe", action="store_true", help="no usar la profundidad de la OAK-D")
    ap.add_argument("--modelo", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--sin-manos", action="store_true", help="no seguir las manos ni mover las pinzas")
    ap.add_argument("--sin-giro", action="store_true", help="no orientar la pinza con la mano")
    ap.add_argument("--escenario", choices=["tuberias"], default=None,
                    help="anade un escenario de trabajo alrededor del robot")
    ap.add_argument("--orca", action="store_true", help="OrcaHand en lugar de las pinzas")
    ap.add_argument("--camara2", action="store_true", help="usar dos OAK-D y fusionar las vistas")
    ap.add_argument("--sin-imu", action="store_true",
                    help="no usar la IMU de las OAK-D (si da problemas; la fusion sigue funcionando)")
    ap.add_argument("--proyector", type=float, default=0.7,
                    help="intensidad 0-1 del proyector IR de las OAK-D Pro (0 = apagado)")
    ap.add_argument("--sin-manos-b", action="store_true",
                    help="con --camara2: la camara B no sigue las manos (solo cuerpo). Libera mucha CPU "
                         "si el seguimiento va lento; las manos se leen solo con la camara A")
    ap.add_argument("--sin-estereo-fino", action="store_true",
                    help="no activar subpixel / comprobacion izq-der / mediana en el estereo de las OAK-D")
    ap.add_argument("--mxid", nargs=2, metavar=("ID_A", "ID_B"), default=None,
                    help="ids de las dos OAK-D (camara principal, secundaria); ver python camaras.py --listar")
    args = ap.parse_args()
    if args.camara2 and args.webcam is not None:
        ap.error("--camara2 es para dos OAK-D, no se combina con --webcam")
    if args.orca:
        import mano_orca

    # El escenario de tuberias siempre se carga, pero oculto; la tecla 1 lo muestra.
    robot = RobotTron2(args.xml, escenario_tuberias.anadir_al_modelo, orca=args.orca)
    escena = escenario_tuberias.EscenarioTuberias(robot)
    robot.resumen()
    girar = not (args.sin_manos or args.sin_giro)
    ids = [None]
    if args.camara2:
        ids = list(args.mxid) if args.mxid else [i for i, _ in listar_oak()[:2]]
        if len(ids) < 2:
            raise SystemExit(f"--camara2 necesita dos OAK-D libres y solo veo {len(ids)} (python camaras.py --listar)")
        print("Camara A:", ids[0], "| Camara B:", ids[1])
    camaras, hilos = [], []
    for k_cam, ident in enumerate(ids):   # una instancia COMPLETA de seguimiento por camara (no se comparten detectores)
        seg_k = SeguidorBrazos(args.modelo, usar_profundidad=not args.solo_mediapipe)
        sin_manos_k = args.sin_manos or (args.sin_manos_b and k_cam > 0)
        manos_k = None if sin_manos_k else SeguidorManos()
        cam_k = (CamaraWebcam(args.webcam) if args.webcam is not None else
                 CamaraOAK(mxid=ident, proyector=args.proyector, usar_imu=not args.sin_imu,
                           estereo_fino=not args.sin_estereo_fino))
        hilo_k = HiloSeguimiento(cam_k, seg_k, manos_k)
        camaras.append(cam_k)
        hilos.append(hilo_k)
    for hilo_k in hilos:
        hilo_k.start()
    fusion = FusionCamaras(hilos)   # con una sola camara se comporta como antes

    R_cam = fusion.R_cam
    estado = dict(fps_seg=0.0, fps_control=0.0, pausado=False, espejo=False, calibrado=False,
                  errores={}, mensaje="", pinzas={}, giros={})
    suaves = {}           # lado del robot -> objetivos suavizados {codo, muneca, w, R}
    t_sin_brazos, t_diag = None, -1e9   # para el diagnostico en consola
    perfil, t_perfil = {}, time.monotonic()   # ms por seccion del bucle (si va lento se informa)
    t_sync, t_ventana, t_ventana2 = -1e9, -1e9, -1e9
    giros = {}            # lado del robot -> {"A0": mano neutra, "q0": giro neutro, "obj": objetivo}
    desde_detras = True
    calib = None          # CalibracionCamaras en curso (tecla c)
    t_mensaje, n_mostrado, n_mano = 0.0, -1, -1
    n_mostrado2, t_aviso_negra = -1, -1e9
    congelado_ant, t_lejos = {}, {}
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
                if fusion.error:
                    print("Error en el seguimiento:", fusion.error)
                    break

                if calib is not None:
                    calib.alimentar()
                    estado["mensaje"] = f"Calibrando: quieto, brazos colgando ({calib.restante():.1f} s)"
                    t_mensaje = t0 + 0.5
                    if calib.terminada():
                        estado["mensaje"] = aplicar_calibracion(calib, fusion, estado)
                        giros.clear()
                        t_mensaje = t0 + 6.0
                        calib = None

                t_s = time.perf_counter()
                snap = fusion.ultimo()
                perfil["fusion"] = perfil.get("fusion", 0.0) + time.perf_counter() - t_s
                if fusion.R_cam is not R_cam:   # calibracion nueva (o inclinacion de la IMU):
                    R_cam = fusion.R_cam        # el giro neutro de las munecas se vuelve a tomar
                    giros.clear()
                fresco = snap is not None and t0 - snap["t"] < CADUCIDAD
                brazos = snap["brazos"] if fresco else {}
                aperturas = snap["aperturas"] if fresco else {}
                # Sin brazos un rato: se explica en la consola, camara a camara, por que
                if brazos:
                    t_sin_brazos = None
                elif t_sin_brazos is None:
                    t_sin_brazos = t0
                elif t0 - t_sin_brazos > 2.0 and t0 - t_diag > 2.0:
                    t_diag = t0
                    print(f"[diagnostico] sin brazos desde hace {t0 - t_sin_brazos:.0f} s"
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
                            t_lejos[lado_r] = t_lejos.get(lado_r, 0.0) + dt
                            if t_lejos[lado_r] < 0.5:      # en un movimiento rapido es normal un momento
                                continue
                            motivo = f"no llega al objetivo (error muneca {err[1] * 100:.0f} cm)"
                        else:
                            t_lejos[lado_r] = 0.0
                            continue
                        estado["congelado"][lado_r] = motivo
                if estado["congelado"] != congelado_ant:
                    for lado_r, motivo in estado["congelado"].items():
                        print(f"[{time.strftime('%H:%M:%S')}] robot {lado_r} quieto: {motivo}")
                    congelado_ant = dict(estado["congelado"])

                marcadores = []
                t_s = time.perf_counter()
                if not estado["pausado"]:
                    # El objetivo se recalcula en CADA ciclo con la ultima medida (aunque la
                    # camara no haya dado foto nueva) y el muelle amortiguado lo sigue: movimiento
                    # continuo a 60 Hz en lugar de un salto por cada foto de la camara.
                    for lado_h, datos in brazos.items():
                        lado_r = OTRO_LADO[lado_h] if estado["espejo"] else lado_h
                        codo, muneca = objetivos(robot, lado_r, datos, R_cam, estado["espejo"])
                        w = peso_codo(datos)
                        if lado_r not in suaves:
                            suaves[lado_r] = dict(
                                f_codo=ObjetivoSuave(OMEGA_OBJETIVO, ZONA_MUERTA_OBJ),
                                f_muneca=ObjetivoSuave(OMEGA_OBJETIVO, ZONA_MUERTA_OBJ),
                                f_w=ObjetivoSuave(OMEGA_OBJETIVO))
                        s = suaves[lado_r]
                        s["codo"] = s["f_codo"].actualizar(codo, dt)
                        s["muneca"] = s["f_muneca"].actualizar(muneca, dt)
                        s["w"] = float(s["f_w"].actualizar(w, dt))
                    # un brazo que deja de verse mantiene su ultimo objetivo, pero el muelle
                    # termina de frenar con suavidad en lugar de cortarse en seco
                    for lado_r, s in suaves.items():
                        if lado_r not in {OTRO_LADO[l] if estado["espejo"] else l for l in brazos}:
                            s["codo"] = s["f_codo"].actualizar(s["f_codo"].ancla, dt)
                            s["muneca"] = s["f_muneca"].actualizar(s["f_muneca"].ancla, dt)
                    # las pinzas siguen a las manos; si una mano deja de verse,
                    # la pinza se queda como estaba
                    foto_nueva = fresco and snap["n"] != n_mano   # el retargeting, 1 vez por foto
                    if foto_nueva:
                        n_mano = snap["n"]
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
                        if not girar:
                            continue
                        if mano.get("tam_px", 1e9) < MIN_PX_GIRO:
                            continue          # mano demasiado pequena: no se toca el giro
                        A = marco_mano(robot, mano, R_cam, estado["espejo"])
                        g = giros.get(lado_r)
                        if g is None:   # al ver la mano por primera vez no hay salto:
                            # esa postura de la mano es la neutra y la pinza no se mueve
                            g = {"A0": A, "q0": brazo.giro(), "obj": brazo.giro(),
                                 "cam": mano.get("cam"), "A_ult": A}
                            giros[lado_r] = g
                        elif mano.get("cam") != g["cam"]:
                            # otra camara da ahora esta mano: su orientacion difiere de la
                            # anterior en el error de calibracion. Se reancla la referencia
                            # para que A @ A0^T (y el giro de la pinza) siga igual.
                            g["A0"] = g["A0"] @ g["A_ult"].T @ A
                            g["cam"] = mano.get("cam")
                        g["A_ult"] = A
                        # solo el giro de la mano alrededor del antebrazo; las
                        # inclinaciones (doblar la muneca) se descartan
                        theta = angulo_giro(A @ g["A0"].T, brazo.eje_giro())
                        th_ant = g.get("theta")
                        if th_ant is not None:
                            # continuidad: sin saltos de +180 a -180, y un salto enorme entre
                            # dos fotos se descarta (salvo que se repita: entonces es real)
                            theta = th_ant + (theta - th_ant + np.pi) % (2 * np.pi) - np.pi
                            if foto_nueva:
                                if abs(theta - th_ant) > SALTO_GIRO and g.get("saltos", 0) < 2:
                                    g["saltos"] = g.get("saltos", 0) + 1
                                    theta = th_ant
                                else:
                                    g["saltos"] = 0
                            else:
                                theta = th_ant
                        g["theta"] = theta
                        g["obj"], g["v"] = suavizar(g["obj"], g.get("v", 0.0), g["q0"] + theta,
                                                    OMEGA_GIRO, dt)
                    # las OrcaHand se mueven en cada ciclo hacia su ultimo objetivo; con el
                    # tubo cogido los dedos no cierran mas alla de su grosor
                    for lado_r, orca in robot.manos.items():
                        sujeta = escena.activa and escena.sujeta == lado_r
                        orca.mover(dt, mano_orca.TOPE_TUBO if sujeta else None)
                    # los brazos que dejan de verse mantienen su ultimo objetivo
                    for lado_r, s in suaves.items():
                        brazo = robot.brazos[lado_r]
                        estado["errores"][lado_r] = brazo.seguir(s["codo"], s["muneca"], dt, s["w"])
                        brazo.mover_muneca(giros[lado_r]["obj"] if lado_r in giros else None, dt)
                        marcadores += [(s["codo"], COLOR_CODO), (s["muneca"], COLOR_MUNECA)]

                perfil["robot+manos"] = perfil.get("robot+manos", 0.0) + time.perf_counter() - t_s
                t_s = time.perf_counter()
                robot.actualizar()
                if escena is not None:          # la pieza sigue a la pinza, encaja o cae
                    escena.actualizar(robot, estado["pinzas"], dt)
                    robot.actualizar()
                perfil["mujoco fisica"] = perfil.get("mujoco fisica", 0.0) + time.perf_counter() - t_s
                t_s = time.perf_counter()
                if t0 - t_sync >= 1.0 / HZ_VISOR:     # el visor no necesita refrescarse a 60 Hz
                    t_sync = t0
                    with v.lock():
                        dibujar_marcadores(v.user_scn, marcadores)
                    v.sync()
                perfil["mujoco visor"] = perfil.get("mujoco visor", 0.0) + time.perf_counter() - t_s
                t_s = time.perf_counter()

                estado["giros"] = {l: b.giro() for l, b in robot.brazos.items()}
                estado["fps_control"] = 0.95 * estado["fps_control"] + 0.05 / max(dt, 1e-3)
                if t0 > t_mensaje:
                    estado["mensaje"] = ""
                # Dos ventanas, cada una con su ritmo. Si las dos toca dibujarlas en este ciclo se
                # dibuja SOLO la que lleva mas retraso (asi el ciclo no se alarga y ninguna se
                # queda sin refrescar: antes la camara 2 era un "elif" de la principal y, con el
                # bucle mas lento que la camara, nunca llegaba a dibujarse -> ventana negra).
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
                    estado["fps_seg"] = fusion.fps
                    estado["fps_cam"] = hilos[0].fps_camara
                    estado["escena"] = escena.texto() if escena else ""
                    fu_v = snap.get("fusion")
                    # imagen en bruto (30 fps) solo si el resultado mostrado es de la camara A
                    base_es_a = (not fu_v) or fu_v["activas"][:1] in ([], ["A"])
                    vista = vista_fluida(hilos[0], snap) if base_es_a else snap
                    cv2.imshow(VENTANA, componer_vista(vista, estado))
                elif toca2:
                    n_mostrado2 = crudo1["n"]
                    t_ventana2 = t0
                    snap_b = hilos[1].ultimo()
                    if snap_b is not None:
                        snap_b = vista_fluida(hilos[1], snap_b)
                    else:        # MediaPipe aun no ha dado nada: al menos se ve la imagen
                        snap_b = dict(bgr=crudo1["bgr"], depth=None, lm2d=None, manos=[], brazos={})
                    if crudo1["bgr"].max() == 0 and t0 - t_aviso_negra > 5.0:
                        t_aviso_negra = t0
                        print("[camara B] llegan fotogramas completamente NEGROS (la camara entrega "
                              "imagen vacia): revisa el cable USB / que no la use otro programa")
                    cv2.imshow(VENTANA2, componer_vista_secundaria(snap_b, hilos[1].fps_camara,
                                                                    hilos[1].fps))

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
                if tecla in (ord("q"), 27):
                    break
                elif tecla == ord("p"):
                    estado["pausado"] = not estado["pausado"]
                elif tecla == ord("m"):
                    estado["espejo"] = not estado["espejo"]
                    suaves.clear()
                    giros.clear()
                    estado["errores"].clear()
                    estado["pinzas"].clear()
                elif tecla == ord("o"):
                    giros.clear()
                    estado["mensaje"] = "Giro de munecas recentrado"
                    t_mensaje = t0 + 2.0
                elif tecla == ord("1"):
                    estado["mensaje"] = ("Escenario de tuberias reiniciado" if escena.activa
                                         else "Escenario de tuberias")
                    escena.activar(v)
                    t_mensaje = t0 + 2.0
                elif tecla == ord("v"):
                    desde_detras = not desde_detras
                    colocar_camara(v, robot, desde_detras)
                elif tecla == ord("r"):
                    robot.reposo()
                    suaves.clear()
                    giros.clear()
                    estado["errores"].clear()
                    estado["pinzas"].clear()
                elif tecla == ord("c"):
                    calib = CalibracionCamaras(hilos, fusion=fusion)
                if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1 and n_mostrado >= 0:
                    break

                espera = 1.0 / FREC_CONTROL - (time.monotonic() - t0)
                if espera > 0:
                    time.sleep(espera)
    finally:
        for hilo_k in hilos:
            hilo_k.parar()
        for cam_k in camaras:
            cam_k.cerrar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()