"""
Calibracion EXTRINSECA de las dos OAK-D del TRON 2: donde esta la camara B respecto a la A.
Genera camaras_extrinsecas.json (p_A = R_ab @ p_B + t_ab), que lee teleop_tron2.py --camara2
(fusion.cargar_extrinsecas). Con ella la camara B entra en la fusion desde el primer fotograma,
con una precision de ~1 grado / ~1 cm, en vez de esperar a autocalibrarse con tu cuerpo.

  python calibrar_extrinseca.py --generar-tablero      # guarda tablero_charuco.png para imprimir
  python calibrar_extrinseca.py                        # calibra con el tablero (cierra teleop_tron2.py antes)
  python calibrar_extrinseca.py --mxid ID_A ID_B       # fijando cual es A y cual B (python camaras.py --listar)

TABLERO: imprimelo a ESCALA REAL (100 %, sin "ajustar a pagina") y mide con una regla el lado de un
cuadrado: tiene que dar --lado (por defecto 40 mm). Pegalo en una superficie plana y RIGIDA (tablero de
madera, cristal...). Cuanto mas grande, mejor: en A3 usa  --lado 0.06 --marcador 0.045 al generarlo
y al calibrar.

COMO: colocalo donde LAS DOS camaras lo vean a la vez (1-2 m, inclinado 20-45 grados hacia ambas).
Pulsa ESPACIO: durante 1 s se promedian muchos fotogramas (el tablero tiene que estar QUIETO; si se
mueve, esa captura se descarta). Repite con >= 10 posiciones/inclinaciones DISTINTAS (cerca/lejos,
a los lados, girado). Pulsa C para calcular y guardar, Q para salir sin guardar.
Si las dos camaras estan separadas mas de ~110 grados no pueden ver un tablero plano a la vez; en ese
caso no uses este metodo: la fusion se calibrara sola con tu cuerpo (mas lenta y menos precisa).
"""
import argparse
import datetime
import json
import os
import sys
import threading
import time

import cv2
import numpy as np


# ═════════════ funciones puras ═════════════

def crear_tablero(cuadros=(7, 5), lado=0.04, marcador=0.03, diccionario="DICT_4X4_50"):
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, diccionario))
    board = cv2.aruco.CharucoBoard(tuple(cuadros), float(lado), float(marcador), dic)
    return board, cv2.aruco.CharucoDetector(board)


def pose_tablero(detector, board, gris, K, dist, min_esquinas=8, max_reproj_px=1.0):
    """(T 4x4 camara<-tablero, mensaje, esquinas, ids). T es None si la medida no vale."""
    corners, ids, _, _ = detector.detectBoard(gris)
    if corners is None or ids is None or len(ids) < min_esquinas:
        return None, f"esquinas {0 if ids is None else len(ids)}/{min_esquinas}", corners, ids
    obj = board.getChessboardCorners()[ids.flatten()].astype(np.float64)
    img = corners.reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, "solvePnP fallo", corners, ids
    proy, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proy.reshape(-1, 2) - img) ** 2, axis=1))))
    if err > max_reproj_px:
        return None, f"reproyeccion {err:.2f}px > {max_reproj_px}", corners, ids
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, f"{len(ids)} esquinas, err {err:.2f}px", corners, ids


def proyectar_a_so3(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def angulo_rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def promediar_transformaciones(Ts):
    """Media de transformaciones 4x4. Devuelve (T, std_giro_deg, std_traslacion_mm)."""
    R = proyectar_a_so3(np.mean([T[:3, :3] for T in Ts], axis=0))
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    ang = [angulo_rot_deg(R.T @ T[:3, :3]) for T in Ts]
    dt = [np.linalg.norm(T[:3, 3] - t) * 1000 for T in Ts]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, float(np.sqrt(np.mean(np.square(ang)))), float(np.sqrt(np.mean(np.square(dt))))


def combinar_capturas(Ts, k_mad=3.5):
    """Media robusta de varias estimaciones B->A (una por captura): descarta las que se alejan
    mas de k_mad veces la mediana de desviaciones (MAD). Devuelve (T, rms_giro, rms_t_mm, n_usadas)."""
    Ts = list(Ts)
    if len(Ts) >= 4:
        T0, _, _ = promediar_transformaciones(Ts)
        dr = np.array([angulo_rot_deg(T0[:3, :3].T @ T[:3, :3]) for T in Ts])
        dt = np.array([np.linalg.norm(T[:3, 3] - T0[:3, 3]) * 1000 for T in Ts])
        buenas = (dr <= max(0.5, k_mad * np.median(dr))) & (dt <= max(5.0, k_mad * np.median(dt)))
        if buenas.sum() >= 3:
            Ts = [T for T, b in zip(Ts, buenas) if b]
    T, g, m = promediar_transformaciones(Ts)
    return T, g, m, len(Ts)


def guardar(ruta, R, t, mxid_a, mxid_b, n, extra):
    d = {"mxid_a": mxid_a, "mxid_b": mxid_b, "R_ab": np.asarray(R).tolist(), "t_ab": np.asarray(t).tolist(),
         "convencion": "p_A = R_ab @ p_B + t_ab", "metodo": "charuco", "n_capturas": n,
         "fecha": datetime.datetime.now().isoformat(timespec="seconds"), **extra}
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


# ═════════════ camaras (DepthAI v3, las mismas que usa teleop_tron2.py) ═════════════

class _Lector(threading.Thread):
    """Lee una CamaraOAK en segundo plano y deja siempre el ultimo fotograma."""

    def __init__(self, cam):
        super().__init__(daemon=True)
        self.cam, self.ultimo, self.n, self.error, self.activo = cam, None, 0, None, True

    def run(self):
        try:
            while self.activo:
                bgr, _ = self.cam.leer()
                if bgr is not None and self.cam.K is not None:
                    self.n += 1
                    self.ultimo = (self.n, bgr, self.cam.K)
        except Exception as e:
            self.error = repr(e)


def _abrir(args):
    from camaras import CamaraOAK, listar_oak
    ids = list(args.mxid) if args.mxid else [i for i, _ in listar_oak()[:2]]
    if len(ids) < 2:
        sys.exit(f"Hacen falta 2 OAK-D libres (cierra teleop_tron2.py) y solo veo {len(ids)}: python camaras.py --listar")
    print(f"Camara A: {ids[0]} | Camara B: {ids[1]}  (resolucion {args.ancho}x{args.alto})")
    cams = [CamaraOAK(ancho=args.ancho, alto=args.alto, fps=args.fps, mxid=i, usar_imu=False,
                      proyector=0.0, estereo_fino=False) for i in ids]
    lectores = [_Lector(c) for c in cams]
    for l in lectores:
        l.start()
    t0 = time.monotonic()
    while any(l.ultimo is None for l in lectores):
        if any(l.error for l in lectores):
            sys.exit(f"Error en una camara: {[l.error for l in lectores]}")
        if time.monotonic() - t0 > 20:
            sys.exit("No llegan imagenes de las dos camaras (cable/USB3). Prueba --ancho 640 --alto 480.")
        time.sleep(0.05)
    return ids, cams, lectores


def _cerrar(cams, lectores):
    for l in lectores:
        l.activo = False
    for c in cams:
        c.cerrar()


# ═════════════ calibracion ═════════════

def calibrar(args, ids, cams, lectores):
    board, det = crear_tablero(args.cuadros, args.lado, args.marcador, args.diccionario)
    dist = np.zeros(5)                     # CamaraOAK pide la imagen YA sin distorsion (enableUndistortion)
    capturas = []                          # una estimacion B->A por captura (media de una rafaga)
    rafaga, t_fin, n_vistos = None, 0.0, [-1, -1]
    msg = ""
    print("\nESPACIO = capturar (tablero QUIETO 1 s) | C = calcular y guardar | Q = salir\n")
    while True:
        vis, poses = [], []
        for i, lec in enumerate(lectores):
            if lec.error:
                sys.exit(f"Camara {'AB'[i]}: {lec.error}")
            n, bgr, K = lec.ultimo
            img = bgr.copy()
            T, m, corners, ids_c = pose_tablero(det, board, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                                K, dist, args.min_esquinas, args.max_reproj)
            if ids_c is not None and len(ids_c) > 0:
                cv2.aruco.drawDetectedCornersCharuco(img, corners, ids_c)
            cv2.putText(img, f"{'AB'[i]}: {m}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if T is not None else (0, 0, 255), 2)
            vis.append(cv2.resize(img, None, fx=0.5, fy=0.5))
            poses.append((n, T))
        ambas = poses[0][1] is not None and poses[1][1] is not None
        if rafaga is not None:                                  # recogiendo la rafaga
            if ambas and (poses[0][0] != n_vistos[0] or poses[1][0] != n_vistos[1]):
                n_vistos = [poses[0][0], poses[1][0]]
                rafaga.append(poses[0][1] @ np.linalg.inv(poses[1][1]))   # B -> A
            if time.monotonic() >= t_fin:
                if len(rafaga) < 4:
                    msg = f"Captura descartada: solo {len(rafaga)} fotogramas con tablero en las dos camaras"
                else:
                    T, g, mm = promediar_transformaciones(rafaga)
                    if g > args.max_quieto_deg or mm > args.max_quieto_mm:
                        msg = f"Captura descartada: el tablero se movio ({g:.2f} grados, {mm:.1f} mm). Quieto!"
                    else:
                        capturas.append(T)
                        msg = f"Captura {len(capturas)} OK ({len(rafaga)} fotogramas, estabilidad {g:.2f} grados / {mm:.1f} mm)"
                print(" ", msg)
                rafaga = None
        banner = ("RECOGIENDO... quieto" if rafaga is not None else
                  f"capturas: {len(capturas)}  {'LISTO: ESPACIO' if ambas else 'el tablero tiene que verse en AMBAS'}")
        cv2.putText(vis[0], banner, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis[1], msg[:60], (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Calibracion extrinseca  (A | B)", np.hstack(vis))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            return None
        if k == ord(" ") and rafaga is None and ambas:
            rafaga, t_fin, n_vistos = [], time.monotonic() + args.duracion_rafaga, [-1, -1]
        if k == ord("c"):
            if len(capturas) < 5:
                print(f"  Faltan capturas ({len(capturas)}/5 minimo, ideal >= 10 en posiciones distintas).")
                continue
            T, g, mm, n_usadas = combinar_capturas(capturas)
            print(f"\n  {n_usadas}/{len(capturas)} capturas usadas -> dispersion: giro {g:.2f} grados, traslacion {mm:.1f} mm")
            extra = {"dispersion_giro_deg": g, "dispersion_t_mm": mm, "capturas_usadas": n_usadas,
                     "lado_cuadro_m": args.lado, "resolucion": [args.ancho, args.alto]}
            if g > 2.0 or mm > 20:
                print("  AVISO: dispersion alta. Repite con el tablero mas plano y rigido, mejor iluminado, "
                      "mas grande y con inclinaciones mas variadas. Se guarda igualmente; la fusion la usa y la vigila.")
            return T[:3, :3], T[:3, 3], n_usadas, extra


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generar-tablero", action="store_true")
    ap.add_argument("--cuadros", type=int, nargs=2, default=(7, 5), metavar=("X", "Y"))
    ap.add_argument("--lado", type=float, default=0.04, help="lado del cuadrado IMPRESO (m)")
    ap.add_argument("--marcador", type=float, default=0.03, help="lado del marcador ArUco IMPRESO (m)")
    ap.add_argument("--diccionario", default="DICT_4X4_50")
    ap.add_argument("--min-esquinas", type=int, default=10)
    ap.add_argument("--max-reproj", type=float, default=1.0, help="error de reproyeccion maximo (px)")
    ap.add_argument("--duracion-rafaga", type=float, default=1.0, help="s que se promedian en cada captura")
    ap.add_argument("--max-quieto-deg", type=float, default=0.5)
    ap.add_argument("--max-quieto-mm", type=float, default=6.0)
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--alto", type=int, default=960)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--mxid", nargs=2, metavar=("ID_A", "ID_B"), default=None)
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()

    if args.generar_tablero:
        board, _ = crear_tablero(args.cuadros, args.lado, args.marcador, args.diccionario)
        cv2.imwrite("tablero_charuco.png",
                    board.generateImage((args.cuadros[0] * 200, args.cuadros[1] * 200), marginSize=40))
        print(f"Guardado tablero_charuco.png ({args.cuadros[0]}x{args.cuadros[1]}). Imprimelo al 100 % para que "
              f"cada cuadrado mida {args.lado * 1000:.0f} mm y comprueba con una regla.")
        return

    from fusion import ruta_extrinsecas
    ids, cams, lectores = _abrir(args)
    try:
        res = calibrar(args, ids, cams, lectores)
    finally:
        cv2.destroyAllWindows()
        _cerrar(cams, lectores)
    if res is None:
        print("Cancelado: no se ha guardado nada.")
        return
    R, t, n, extra = res
    ruta = args.salida or ruta_extrinsecas()
    guardar(ruta, R, t, ids[0], ids[1], n, extra)
    print(f"\nGuardado {ruta}\n   B respecto a A: giro {angulo_rot_deg(R):.1f} grados, "
          f"distancia entre camaras {np.linalg.norm(t):.2f} m\n"
          f"   (comprueba que la distancia se parece a la que mides con un metro entre las dos camaras)\n"
          f"   Ya puedes lanzar: python teleop_tron2.py --camara2 --orca")


if __name__ == "__main__":
    main()