"""
Teleoperacion de los brazos del TRON 2 (DACH_TRON2A) en MuJoCo con una OAK-D.

Uso (desde la raiz de tron2-robot-description, con estos 4 .py copiados alli):
  python teleop_tron2.py                     # OAK-D: RGB + profundidad
  python teleop_tron2.py --solo-mediapipe    # OAK-D sin usar la profundidad
  python teleop_tron2.py --webcam 0          # sin OAK-D, con una webcam normal
  python teleop_tron2.py --modelo lite       # MediaPipe mas rapido (menos preciso)
  python teleop_tron2.py --sin-manos         # no seguir las manos (va mas rapido)

Las pinzas siguen a las manos: palma abierta abre la pinza, puno cerrado la
cierra, y las posiciones intermedias se reproducen de forma proporcional.
Cada mano manda sobre la pinza de su lado (o la contraria en modo espejo).

Teclas (con la ventana de la camara seleccionada):
  q / ESC  salir
  p        pausar / reanudar (embrague: congela el robot mientras te recolocas)
  c        calibrar: de pie, mirando a la camara, brazos relajados hacia abajo
  m        modo espejo on/off (tu brazo izquierdo mueve el derecho del robot)
  v        ver el robot desde detras / desde delante
  r        devolver los brazos a la postura de reposo

Estructura: un hilo lee la camara y ejecuta MediaPipe (a lo que de el PC, ~12-25 fps)
y el bucle principal mueve el robot a 60 Hz, suavizando los objetivos entre frames.
"""
import argparse
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from camaras import CamaraOAK, CamaraWebcam
from robot_tron2 import RobotTron2
from seguimiento_brazos import (COLOR_LADO, PUNTOS, VIS_MIN, HiloSeguimiento,
                                SeguidorBrazos, dibujar_esqueleto)
from seguimiento_manos import SeguidorManos, dibujar_manos

VENTANA = "Seguimiento OAK-D"

# Ejes del robot expresados en el marco de la camara, suponiendo camara horizontal
# y operador de frente a ella (la tecla 'c' los recalcula si no es asi).
R_CAM_DEFECTO = np.array([[0.0, 0.0, -1.0],   # x robot (delante)   = -z camara (hacia la camara)
                          [1.0, 0.0, 0.0],    # y robot (izquierda) = +x camara
                          [0.0, -1.0, 0.0]])  # z robot (arriba)    = -y camara

FREC_CONTROL = 60.0        # Hz del bucle del robot
TAU_OBJETIVO = 0.06        # s: suavizado de los objetivos entre frames de la camara
CADUCIDAD = 0.5            # s: si no hay datos nuevos en este tiempo, el robot se queda quieto
W_CODO = 0.6               # peso del objetivo del codo frente al de la muneca
ANG_RECTO = (10.0, 30.0)   # grados: con el brazo casi recto se ignora el codo (evita giros locos)
ZONA_MUERTA_CODO = (5.0, 25.0)  # grados: flexiones menores se tratan como brazo recto (se atenuan)

COLOR_CODO = [1.0, 0.85, 0.0, 0.8]    # marcador amarillo en MuJoCo
COLOR_MUNECA = [0.1, 1.0, 0.3, 0.8]   # marcador verde en MuJoCo
OTRO_LADO = {"L": "R", "R": "L"}


# ---------------------------------------------------------------- retargeting
def calibrar(brazos, linea_hombros):
    """Con los brazos colgando: 'abajo' = media de las direcciones de los brazos,
    'izquierda' = linea de hombros. Corrige la inclinacion de la camara."""
    abajo = sum(brazos[l]["dir_brazo"] + brazos[l]["dir_antebrazo"] for l in ("L", "R"))
    z = -abajo / np.linalg.norm(abajo)
    y = linea_hombros - z * np.dot(linea_hombros, z)
    y /= np.linalg.norm(y)
    return np.vstack([np.cross(y, z), y, z])


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
    return codo, muneca


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
        mini = cv2.resize(cv2.flip(colorear_profundidad(depth), 1), (w // 4, h // 4))
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

    lineas = [f"Camara {estado['fps_seg']:4.1f} fps | robot {estado['fps_control']:3.0f} Hz | "
              f"{'PAUSA' if estado['pausado'] else 'TELEOP'} | {'espejo' if estado['espejo'] else 'directo'} | "
              f"{'calibrado' if estado['calibrado'] else 'sin calibrar'}"]
    for lado in ("L", "R"):
        nombre = "IZQ" if lado == "L" else "DER"
        mano = next((m for m in (snap.get("manos") or []) if m["lado"] == lado), None)
        extra = f" | mano {mano['apertura'] * 100:3.0f}% (bruto {mano['bruto']:.2f})" if mano else " | mano no vista"
        lineas.append(f"Operador {nombre}: "
                      f"{brazos[lado]['fuente'] if lado in brazos else 'no detectado'}{extra}")
    for lado in ("L", "R"):
        if lado in estado["errores"]:
            ec, em = estado["errores"][lado]
            lineas.append(f"Robot {lado}: error codo {ec * 1000:3.0f} mm | muneca {em * 1000:3.0f} mm")
    if estado["mensaje"]:
        lineas.append(estado["mensaje"])
    for k, s in enumerate(lineas):
        texto(img, s, (10, 22 + 22 * k))
    texto(img, "q salir | p pausa | c calibrar | m espejo | v vista | r reposo", (10, h - 12), escala=0.45)
    return img


# ---------------------------------------------------------------- bucle principal
def main():
    ap = argparse.ArgumentParser(description="Teleoperacion de los brazos del TRON 2 con OAK-D")
    ap.add_argument("--xml", default="tron2a/DACH_TRON2A/xml/robot_elecnor.xml")
    ap.add_argument("--webcam", type=int, default=None, help="indice de webcam (en lugar de la OAK-D)")
    ap.add_argument("--solo-mediapipe", action="store_true", help="no usar la profundidad de la OAK-D")
    ap.add_argument("--modelo", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--sin-manos", action="store_true", help="no seguir las manos ni mover las pinzas")
    args = ap.parse_args()

    robot = RobotTron2(args.xml)
    robot.resumen()
    seg = SeguidorBrazos(args.modelo, usar_profundidad=not args.solo_mediapipe)
    manos = None if args.sin_manos else SeguidorManos()
    cam = CamaraWebcam(args.webcam) if args.webcam is not None else CamaraOAK()
    hilo = HiloSeguimiento(cam, seg, manos)
    hilo.start()

    R_cam = R_CAM_DEFECTO.copy()
    estado = dict(fps_seg=0.0, fps_control=0.0, pausado=False, espejo=False, calibrado=False,
                  errores={}, mensaje="", pinzas={})
    suaves = {}           # lado del robot -> objetivos suavizados {codo, muneca, w}
    desde_detras = True
    t_mensaje, n_mostrado = 0.0, -1
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)

    try:
        with mujoco.viewer.launch_passive(robot.m, robot.d, show_left_ui=False, show_right_ui=False) as v:
            colocar_camara(v, robot, desde_detras)
            t_ant = time.monotonic()
            while v.is_running():
                t0 = time.monotonic()
                dt = min(t0 - t_ant, 0.1)
                t_ant = t0
                if hilo.error:
                    print("Error en el seguimiento:", hilo.error)
                    break

                snap = hilo.ultimo()
                fresco = snap is not None and t0 - snap["t"] < CADUCIDAD
                brazos = snap["brazos"] if fresco else {}
                aperturas = snap["aperturas"] if fresco else {}

                marcadores = []
                if not estado["pausado"]:
                    a = 1.0 - np.exp(-dt / TAU_OBJETIVO)
                    for lado_h, datos in brazos.items():
                        lado_r = OTRO_LADO[lado_h] if estado["espejo"] else lado_h
                        codo, muneca = objetivos(robot, lado_r, datos, R_cam, estado["espejo"])
                        w = peso_codo(datos)
                        if lado_r in suaves:
                            s = suaves[lado_r]
                            s["codo"] += a * (codo - s["codo"])
                            s["muneca"] += a * (muneca - s["muneca"])
                            s["w"] += a * (w - s["w"])
                        else:
                            suaves[lado_r] = dict(codo=codo, muneca=muneca, w=w)
                    # las pinzas siguen a las manos; si una mano deja de verse,
                    # la pinza se queda como estaba
                    for lado_h, apertura in aperturas.items():
                        lado_r = OTRO_LADO[lado_h] if estado["espejo"] else lado_h
                        robot.brazos[lado_r].mover_pinza(apertura, dt)
                        estado["pinzas"][lado_r] = apertura
                    # los brazos que dejan de verse mantienen su ultimo objetivo
                    for lado_r, s in suaves.items():
                        estado["errores"][lado_r] = robot.brazos[lado_r].seguir(s["codo"], s["muneca"], dt, s["w"])
                        marcadores += [(s["codo"], COLOR_CODO), (s["muneca"], COLOR_MUNECA)]

                robot.actualizar()
                with v.lock():
                    dibujar_marcadores(v.user_scn, marcadores)
                v.sync()

                estado["fps_control"] = 0.95 * estado["fps_control"] + 0.05 / max(dt, 1e-3)
                if t0 > t_mensaje:
                    estado["mensaje"] = ""
                if snap is not None and snap["n"] != n_mostrado:  # solo si hay imagen nueva
                    n_mostrado = snap["n"]
                    estado["fps_seg"] = hilo.fps
                    cv2.imshow(VENTANA, componer_vista(snap, estado))

                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                elif tecla == ord("p"):
                    estado["pausado"] = not estado["pausado"]
                elif tecla == ord("m"):
                    estado["espejo"] = not estado["espejo"]
                    suaves.clear()
                    estado["errores"].clear()
                    estado["pinzas"].clear()
                elif tecla == ord("v"):
                    desde_detras = not desde_detras
                    colocar_camara(v, robot, desde_detras)
                elif tecla == ord("r"):
                    robot.reposo()
                    suaves.clear()
                    estado["errores"].clear()
                    estado["pinzas"].clear()
                elif tecla == ord("c"):
                    if snap and "L" in snap["brazos"] and "R" in snap["brazos"] and snap["linea_hombros"] is not None:
                        R_cam = calibrar(snap["brazos"], snap["linea_hombros"])
                        estado["calibrado"] = True
                        estado["mensaje"] = "Calibrado"
                    else:
                        estado["mensaje"] = "Calibracion: tienen que verse los dos brazos"
                    t_mensaje = t0 + 2.0
                if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1 and n_mostrado >= 0:
                    break

                espera = 1.0 / FREC_CONTROL - (time.monotonic() - t0)
                if espera > 0:
                    time.sleep(espera)
    finally:
        hilo.parar()
        cam.cerrar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()