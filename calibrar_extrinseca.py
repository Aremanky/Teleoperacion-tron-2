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

SIN IMPRESORA: abre tablero_charuco.png a pantalla completa en un portatil/tablet/monitor, mide con
una regla 5 cuadrados seguidos y divide entre 5 -> --lado; --marcador = 0.75 * lado.

MODO AUTOMATICO (por defecto, oct-2026): no hace falta pulsar nada para capturar. Cuando las dos
camaras ven el tablero y esta QUIETO ~1 s en una postura nueva, se captura solo; a las 12 capturas
calcula y guarda. La consola dice cada 2 s por que no captura. --manual = como antes (ESPACIO).

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
    """(oct-2026) Detector algo mas tolerante que el de por defecto:
      - tryRefineMarkers: con la forma conocida del tablero recupera marcadores que no se vieron a
        la primera (oblicuo, lejos, poco contraste). En pruebas sinteticas: +30 % de esquinas y
        mismo coste.
      - refinamiento subpixel de las esquinas de los marcadores, que ahora tambien se usan para la
        pose cuando no hay esquinas ChArUco coherentes (ver detectar_esquinas)."""
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, diccionario))
    board = cv2.aruco.CharucoBoard(tuple(cuadros), float(lado), float(marcador), dic)
    cp = cv2.aruco.CharucoParameters()
    cp.tryRefineMarkers = True
    dp = cv2.aruco.DetectorParameters()
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    try:
        det = cv2.aruco.CharucoDetector(board, cp, dp, cv2.aruco.RefineParameters())
    except (cv2.error, TypeError):        # otra version de OpenCV con otra firma: el de siempre
        det = cv2.aruco.CharucoDetector(board)
    return board, det


ESTADISTICA = {}     # camara -> contadores de deteccion (diagnostico en consola)
ESQUINAS_SUFICIENTES = 10   # con menos esquinas ChArUco se suman las de los marcadores


def detectar_esquinas(detector, gris, board=None, info=None):
    """Puntos del tablero vistos en la imagen: (obj (N, 3), img (N, 1, 2), tipo) o (None, None, "").
    tipo = "esquinas" (esquinas ChArUco, las mas precisas) o "marcadores" (las 4 esquinas de cada
    marcador ArUco visto).
    (oct-2026) Con OpenCV 5 detectBoard puede devolver un numero distinto de esquinas que de ids
    (lo que rompia drawDetectedCornersCharuco). La primera version de este arreglo tiraba esas
    imagenes enteras y una camara que las daba a menudo parecia "no ver" el tablero. Ahora, si las
    esquinas ChArUco no son coherentes o son pocas, se usan las de los marcadores, que casi siempre
    estan. 'info' (dict) recibe contadores para la consola."""
    res = detector.detectBoard(gris)
    corners, ids, m_corners, m_ids = res[0], res[1], res[2], res[3]
    n_marc = 0 if m_ids is None else len(m_ids)
    obj = img = None
    tipo = ""
    incoherente = False
    if corners is not None and ids is not None and np.size(ids) > 0:
        c = np.asarray(corners, dtype=np.float32)
        i = np.asarray(ids, dtype=np.int32).reshape(-1)
        if c.size == 2 * i.size and board is not None:
            obj = board.getChessboardCorners()[i].astype(np.float64)
            img = c.reshape(-1, 1, 2)
            tipo = "esquinas"
        elif c.size == 2 * i.size:
            img, obj, tipo = c.reshape(-1, 1, 2), i, "esquinas"
        else:
            incoherente = True
    n_esq = 0 if obj is None else len(obj)
    if board is not None and n_marc > 0 and n_esq < ESQUINAS_SUFICIENTES:
        # pocas esquinas ChArUco (o incoherentes): se anaden las de los marcadores
        tabla = {int(k): np.asarray(p, dtype=np.float64) for k, p in zip(board.getIds().ravel(), board.getObjPoints())}
        o, im = ([obj], [np.asarray(img).reshape(-1, 2)]) if n_esq else ([], [])
        for mc, mi in zip(m_corners, np.asarray(m_ids).reshape(-1)):
            if int(mi) in tabla:
                o.append(tabla[int(mi)])
                im.append(np.asarray(mc, dtype=np.float32).reshape(4, 2))
        if len(o) > (1 if n_esq else 0):
            obj, img = np.vstack(o), np.vstack(im).astype(np.float32).reshape(-1, 1, 2)
            tipo = "esquinas+marcadores" if n_esq else "marcadores"
    if info is not None:
        info["marcadores"] = n_marc
        info["esquinas"] = n_esq
        info["incoherentes"] = info.get("incoherentes", 0) + int(incoherente)
        info["m_corners"], info["m_ids"] = m_corners, m_ids
    if obj is None:
        return None, None, ""
    return obj, img, tipo


def pose_tablero(detector, board, gris, K, dist, min_esquinas=8, max_reproj_px=1.0, info=None):
    """(T 4x4 camara<-tablero, mensaje, puntos imagen, tipo). T es None si la medida no vale.
    min_esquinas: puntos minimos (esquinas ChArUco, o esquinas de marcadores)."""
    obj, img, tipo = detectar_esquinas(detector, gris, board, info)
    n_marc = 0 if info is None else info.get("marcadores", 0)
    if obj is None or len(obj) < min_esquinas:
        return None, f"puntos {0 if obj is None else len(obj)}/{min_esquinas} (marcadores {n_marc})", img, tipo
    pts = img.reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, "solvePnP fallo", img, tipo
    proy, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proy.reshape(-1, 2) - pts) ** 2, axis=1))))
    if err > max_reproj_px:
        return None, f"reproyeccion {err:.2f}px > {max_reproj_px} ({tipo})", img, tipo
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, f"{len(obj)} {tipo}, err {err:.2f}px", img, tipo


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

def novedad(T_tablero, previas):
    """Cuanto se diferencia una pose del tablero (en la camara A) de las ya capturadas:
    (grados, metros) respecto a la mas parecida. Sin previas: infinito."""
    if not previas:
        return np.inf, np.inf
    difs = [(angulo_rot_deg(P[:3, :3].T @ T_tablero[:3, :3]), float(np.linalg.norm(P[:3, 3] - T_tablero[:3, 3])))
            for P in previas]
    return min(difs, key=lambda d: d[0] / 10.0 + d[1] / 0.10)


def calibrar(args, ids, cams, lectores):
    """(oct-2026) MODO AUTOMATICO por defecto: basta con ir colocando el tablero (o la pantalla)
    en posiciones distintas y dejarlo quieto ~1 s; la captura se hace sola cuando las DOS camaras
    lo ven y la postura es nueva (>= --novedad-deg o --novedad-m respecto a las anteriores). Al
    llegar a --objetivo capturas calcula y guarda solo. ESPACIO sigue capturando a mano y C
    calcula cuando quieras (con >= 5 capturas). En la consola se dice cada 2 s por que no captura."""
    board, det = crear_tablero(args.cuadros, args.lado, args.marcador, args.diccionario)
    dist = np.zeros(5)                     # CamaraOAK pide la imagen YA sin distorsion (enableUndistortion)
    capturas = []                          # una estimacion B->A por captura (media de una rafaga)
    poses_capturadas = []                  # pose del tablero en A de cada captura (para la novedad)
    rafaga, t_fin, n_vistos, T_inicio = None, 0.0, [-1, -1], None
    t_espera, t_estado = 0.0, 0.0
    msg = ""
    auto = not args.manual
    print("\n" + ("AUTOMATICO: coloca el tablero donde lo vean las dos camaras y dejalo QUIETO ~1 s; "
                  "cambialo de sitio/inclinacion y repite. " if auto else "") +
          "ESPACIO = capturar | C = calcular y guardar | Q = salir\n")
    print(f"Tablero: {args.cuadros[0]}x{args.cuadros[1]}, cuadrado {args.lado * 1000:.1f} mm, marcador "
          f"{args.marcador * 1000:.1f} mm (tienen que ser los que MIDES en el tablero o la pantalla)\n")
    while True:
        vis, poses, estados = [], [], []
        for i, lec in enumerate(lectores):
            if lec.error:
                sys.exit(f"Camara {'AB'[i]}: {lec.error}")
            n, bgr, K = lec.ultimo
            img = bgr.copy()
            info = ESTADISTICA.setdefault(i, {})
            T, m, pts, tipo = pose_tablero(det, board, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                           K, dist, args.min_esquinas, args.max_reproj, info)
            try:                       # solo es el dibujo: nunca debe parar la calibracion
                if info.get("m_ids") is not None and len(info["m_ids"]) > 0:
                    cv2.aruco.drawDetectedMarkers(img, info["m_corners"], info["m_ids"])
                if pts is not None:
                    for p in pts.reshape(-1, 2):
                        cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 0) if tipo == "esquinas" else (255, 160, 0), -1)
            except cv2.error:
                pass
            cv2.putText(img, f"{'AB'[i]}: {m}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if T is not None else (0, 0, 255), 2)
            vis.append(cv2.resize(img, None, fx=0.5, fy=0.5))
            poses.append((n, T))
            estados.append(f"{'AB'[i]}: {m}")
        ambas = poses[0][1] is not None and poses[1][1] is not None
        ahora = time.monotonic()
        if rafaga is not None:                                  # recogiendo la rafaga
            if ambas and (poses[0][0] != n_vistos[0] or poses[1][0] != n_vistos[1]):
                n_vistos = [poses[0][0], poses[1][0]]
                rafaga.append(poses[0][1] @ np.linalg.inv(poses[1][1]))   # B -> A
            if ahora >= t_fin:
                if len(rafaga) < 4:
                    msg = f"Captura descartada: solo {len(rafaga)} fotogramas con tablero en las dos camaras"
                else:
                    T, g, mm = promediar_transformaciones(rafaga)
                    if g > args.max_quieto_deg or mm > args.max_quieto_mm:
                        msg = f"Captura descartada: el tablero se movio ({g:.2f} grados, {mm:.1f} mm). Quieto!"
                    else:
                        capturas.append(T)
                        poses_capturadas.append(T_inicio)
                        msg = (f"Captura {len(capturas)} OK ({len(rafaga)} fotogramas, estabilidad {g:.2f} grados / "
                               f"{mm:.1f} mm). Cambia el tablero de sitio o de inclinacion")
                print(" ", msg)
                rafaga, t_espera = None, ahora + 0.8
        elif auto and ambas and ahora >= t_espera:              # captura automatica
            dg, dm = novedad(poses[0][1], poses_capturadas)
            if dg >= args.novedad_deg or dm >= args.novedad_m:
                rafaga, t_fin, n_vistos, T_inicio = [], ahora + args.duracion_rafaga, [-1, -1], poses[0][1]
        if ahora >= t_estado and rafaga is None:                # por que no captura (consola)
            t_estado = ahora + 2.0
            if not ambas:
                print("  esperando:", " | ".join(estados))
                inc = [f"{'AB'[k]} {v['incoherentes']}" for k, v in ESTADISTICA.items() if v.get("incoherentes")]
                if inc:
                    print("   (salidas incoherentes de OpenCV, ya recuperadas con los marcadores:", ", ".join(inc), ")")
            elif auto:
                dg, dm = novedad(poses[0][1], poses_capturadas)
                print(f"  las dos lo ven; postura parecida a una ya capturada ({dg:.0f} grados, {dm * 100:.0f} cm): "
                      f"muevelo o inclinalo mas")
        if auto and len(capturas) >= args.objetivo:
            print(f"\n  {len(capturas)} capturas: calculando.")
            k = ord("c")
        else:
            banner = ("RECOGIENDO... quieto" if rafaga is not None else
                      f"capturas: {len(capturas)}/{args.objetivo if auto else 10}  "
                      f"{('LISTO' + ('' if auto else ': ESPACIO')) if ambas else 'el tablero tiene que verse en AMBAS'}")
            cv2.putText(vis[0], banner, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(vis[1], msg[:60], (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Calibracion extrinseca  (A | B)", np.hstack(vis))
            k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            return None
        if k == ord(" ") and rafaga is None and ambas:
            rafaga, t_fin, n_vistos, T_inicio = [], ahora + args.duracion_rafaga, [-1, -1], poses[0][1]
        elif k == ord(" ") and not ambas:
            print("  ESPACIO ignorado: el tablero no se detecta en las dos camaras ->", " | ".join(estados))
        if k == ord("c"):
            if len(capturas) < 5:
                print(f"  Faltan capturas ({len(capturas)}/5 minimo, ideal >= 10 en posiciones distintas). "
                      f"C solo CALCULA: las capturas se hacen " + ("solas al dejar el tablero quieto." if auto
                                                                   else "con ESPACIO."))
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
    ap.add_argument("--max-reproj", type=float, default=1.5,
                    help="error de reproyeccion maximo (px); en pantalla 1.5-2 (antes 1.0)")
    ap.add_argument("--manual", action="store_true", help="capturar solo con ESPACIO (sin modo automatico)")
    ap.add_argument("--objetivo", type=int, default=12, help="modo automatico: capturas antes de calcular solo")
    ap.add_argument("--novedad-deg", type=float, default=12.0,
                    help="modo automatico: giro minimo del tablero respecto a las capturas anteriores")
    ap.add_argument("--novedad-m", type=float, default=0.12,
                    help="modo automatico: o desplazamiento minimo (m)")
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