"""
Calibración EXTRÍNSECA entre las dos OAK-D: dónde está la cámara B respecto a la A.
Genera camaras_extrinsecas.json (p_A = R_ab · p_B + t_ab), que lee Skill_Shield_Nivel_6.py.

  python calibrar_extrinseca.py --generar-tablero       # guarda tablero_charuco.png para imprimir
  python calibrar_extrinseca.py                         # calibra con el tablero ChArUco (recomendado)
  python calibrar_extrinseca.py --metodo cuerpo         # sin tablero: mueve la muñeca delante de las dos

ChArUco: imprime el tablero a ESCALA REAL (comprueba con una regla el lado del cuadrado),
pégalo en una superficie plana y rígida, colócalo donde LAS DOS cámaras lo vean a la vez
(ideal: ~1,5-2 m, inclinado hacia ambas) y pulsa ESPACIO con varias posiciones/inclinaciones
distintas (≥ 12). Pulsa C para calcular y guardar, Q para salir.
Con las cámaras a más de ~100° una a otra no pueden ver un tablero plano a la vez: usa --metodo cuerpo.
"""
import argparse
import datetime
import json
import sys

import cv2
import numpy as np

# ═════════════ funciones puras (probadas con datos sintéticos) ═════════════

def crear_tablero(cuadros=(7, 5), lado=0.04, marcador=0.03, diccionario="DICT_4X4_50"):
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, diccionario))
    board = cv2.aruco.CharucoBoard(tuple(cuadros), float(lado), float(marcador), dic)
    return board, cv2.aruco.CharucoDetector(board)


def pose_tablero(detector, board, gris, K, dist, min_esquinas=8, max_reproj_px=1.5):
    """T (4x4) cámara←tablero con solvePnP sobre las esquinas ChArUco, o (None, motivo)."""
    corners, ids, _, _ = detector.detectBoard(gris)
    if corners is None or ids is None or len(ids) < min_esquinas:
        return None, f"esquinas {0 if ids is None else len(ids)}/{min_esquinas}", corners, ids
    obj = board.getChessboardCorners()[ids.flatten()].astype(np.float64)
    img = corners.reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, "solvePnP falló", corners, ids
    proy, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proy.reshape(-1, 2) - img) ** 2, axis=1))))
    if err > max_reproj_px:
        return None, f"reproyección {err:.2f}px > {max_reproj_px}", corners, ids
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rvec)[0]; T[:3, 3] = tvec.ravel()
    return T, f"{len(ids)} esquinas, err {err:.2f}px", corners, ids


def proyectar_a_so3(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R


def promediar_transformaciones(Ts):
    """Media de transformaciones 4x4. Devuelve (T, std_giro_deg, std_traslacion_mm)."""
    R = proyectar_a_so3(np.mean([T[:3, :3] for T in Ts], axis=0))
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    ang = [np.degrees(np.arccos(np.clip((np.trace(R.T @ T[:3, :3]) - 1) / 2, -1, 1))) for T in Ts]
    dt  = [np.linalg.norm(T[:3, 3] - t) * 1000 for T in Ts]
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T, float(np.std(ang)), float(np.std(dt))


def kabsch(PB, PA):
    """R, t que minimizan |R·PB + t − PA|² (puntos Nx3 correspondientes)."""
    cb, ca = PB.mean(0), PA.mean(0)
    H = (PB - cb).T @ (PA - ca)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, ca - R @ cb


def kabsch_recortado(PB, PA, descartar=0.2, iteraciones=3):
    """Kabsch descartando en cada vuelta el 'descartar' % de puntos con más residuo."""
    idx = np.arange(len(PB))
    for _ in range(iteraciones):
        R, t = kabsch(PB[idx], PA[idx])
        res = np.linalg.norm((PB @ R.T + t) - PA, axis=1)
        corte = np.quantile(res[idx], 1 - descartar)
        idx = np.where(res <= corte)[0]
    R, t = kabsch(PB[idx], PA[idx])
    res = np.linalg.norm((PB[idx] @ R.T + t) - PA[idx], axis=1)
    return R, t, float(np.sqrt(np.mean(res ** 2))), len(idx)


def guardar(ruta, R, t, mxid_a, mxid_b, metodo, n, extra):
    d = {"mxid_a": mxid_a, "mxid_b": mxid_b, "R_ab": R.tolist(), "t_ab": np.asarray(t).tolist(),
         "convencion": "p_A = R_ab @ p_B + t_ab", "metodo": metodo, "n_muestras": n,
         "fecha": datetime.datetime.now().isoformat(timespec="seconds"), **extra}
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


# ═════════════ parte con cámaras ═════════════

def _abrir_las_dos(args):
    from config import ANCHO_CAM, ALTO_CAM
    from skill_shield.vision.camara_oak import crear_pipeline_oak
    from skill_shield.vision.multicamara import (abrir_dispositivo, comprobar_usb, activar_ir,
                                                 intrinsecos_color, listar_camaras, mxid_de)
    libres = listar_camaras()
    print("OAK-D disponibles:", libres)
    if len(libres) < 2 and not (args.mxid_a and args.mxid_b):
        sys.exit("Hacen falta 2 OAK-D conectadas (y libres: cierra Skill_Shield_Nivel_6.py).")
    pa = crear_pipeline_oak(con_hd=False); pb = crear_pipeline_oak(con_hd=False)
    dev_a = abrir_dispositivo(pa, args.mxid_a, excluir=[args.mxid_b])
    dev_b = abrir_dispositivo(pb, args.mxid_b, excluir=[mxid_de(dev_a)])
    cams = []
    for nombre, dev in (("A", dev_a), ("B", dev_b)):
        comprobar_usb(dev, nombre)
        if args.metodo == "cuerpo":
            activar_ir(dev, nombre)
        K, dist = intrinsecos_color(dev, ANCHO_CAM, ALTO_CAM)
        cams.append({"nombre": nombre, "dev": dev, "mxid": mxid_de(dev), "K": K, "dist": dist,
                     "qc": dev.getOutputQueue("color", maxSize=2, blocking=False),
                     "qd": dev.getOutputQueue("depth", maxSize=2, blocking=False)})
    print(f"A = {cams[0]['mxid']}   B = {cams[1]['mxid']}")
    return cams


def _ultimo(q, previo):
    p = q.tryGet()
    while p is not None:
        previo = p
        p = q.tryGet()
    return previo


def calibrar_charuco(args, cams):
    board, det = crear_tablero(args.cuadros, args.lado, args.marcador, args.diccionario)
    ultimo = [None, None]
    pares, ahora_ok = [], False
    print("\nESPACIO = capturar pareja | C = calcular y guardar | Q = salir\n")
    while True:
        vis, poses = [], []
        for i, c in enumerate(cams):
            ultimo[i] = _ultimo(c["qc"], ultimo[i])
            if ultimo[i] is None:
                continue
            img = ultimo[i].getCvFrame()
            T, msg, corners, ids = pose_tablero(det, board, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                                c["K"], c["dist"], args.min_esquinas, args.max_reproj)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(img, corners, ids)
            col = (0, 255, 0) if T is not None else (0, 0, 255)
            cv2.putText(img, f"{c['nombre']}: {msg}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            vis.append(img); poses.append(T)
        if len(vis) == 2:
            ahora_ok = poses[0] is not None and poses[1] is not None
            cv2.putText(vis[0], f"parejas: {len(pares)}  {'LISTO: ESPACIO' if ahora_ok else ''}",
                        (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("Calibracion extrinseca  (A | B)", np.hstack(vis))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            return None
        if k == ord(' ') and len(vis) == 2 and ahora_ok:
            TA, TB = poses
            pares.append(TA @ np.linalg.inv(TB))          # B → A
            print(f"  pareja {len(pares)} guardada")
        if k == ord('c'):
            if len(pares) < 5:
                print("  Faltan parejas (mínimo 5, ideal ≥ 12)."); continue
            T, std_g, std_t = promediar_transformaciones(pares)
            print(f"\n  {len(pares)} parejas → dispersión: giro {std_g:.2f}°  traslación {std_t:.1f} mm")
            if std_g > 2.0 or std_t > 30:
                print("  ⚠️ Dispersión alta: repite con el tablero más plano/rígido, mejor iluminado y más inclinaciones.")
            return T[:3, :3], T[:3, 3], len(pares), {"dispersion_giro_deg": std_g, "dispersion_t_mm": std_t}


def calibrar_cuerpo(args, cams):
    import mediapipe as mp
    from config import ANCHO_CAM, ALTO_CAM, PS_MUNECA_DER, PS_MUNECA_IZQ, PROF_Z_MIN, PROF_Z_MAX
    from skill_shield.brazo.angulos_humano import profundidad_robusta
    from skill_shield.vision.profundidad_3d import pixel_a_3d
    idx = PS_MUNECA_DER if args.brazo == "derecho" else PS_MUNECA_IZQ
    poses = [mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.6,
                                    min_tracking_confidence=0.6) for _ in cams]
    ultimo_c, ultimo_d = [None, None], [None, None]
    PA, PB = [], []
    capturando = False
    print("\nESPACIO = empezar/parar a grabar | C = calcular y guardar | Q = salir")
    print("Colócate donde te vean las dos y mueve la muñeca por TODO el volumen de trabajo "
          "(arriba/abajo, cerca/lejos, izquierda/derecha); ≥ 300 puntos.\n")
    while True:
        pts, imgs = [], []
        for i, c in enumerate(cams):
            ultimo_c[i] = _ultimo(c["qc"], ultimo_c[i]); ultimo_d[i] = _ultimo(c["qd"], ultimo_d[i])
            if ultimo_c[i] is None or ultimo_d[i] is None:
                pts.append(None); continue
            img = ultimo_c[i].getCvFrame(); depth = ultimo_d[i].getFrame()
            r = poses[i].process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            p3 = None
            if r.pose_landmarks:
                l = r.pose_landmarks.landmark[idx]
                if l.visibility > 0.8 and 0 <= l.x < 1 and 0 <= l.y < 1:
                    z = profundidad_robusta(depth, int(l.x * depth.shape[1]), int(l.y * depth.shape[0]))
                    if z is not None and PROF_Z_MIN < z < PROF_Z_MAX:
                        p3 = np.array(pixel_a_3d(l.x * ANCHO_CAM, l.y * ALTO_CAM, z, ANCHO_CAM, ALTO_CAM,
                                                 float(c["K"][0][0])))
                mp.solutions.drawing_utils.draw_landmarks(img, r.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            pts.append(p3); imgs.append(img)
        if len(imgs) == 2:
            if capturando and pts[0] is not None and pts[1] is not None:
                PA.append(pts[0]); PB.append(pts[1])
            cv2.putText(imgs[0], f"{'GRABANDO' if capturando else 'pausa'}  puntos: {len(PA)}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Calibracion extrinseca  (A | B)", np.hstack(imgs))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            return None
        if k == ord(' '):
            capturando = not capturando
        if k == ord('c'):
            if len(PA) < 100:
                print("  Faltan puntos (mínimo 100, ideal ≥ 300)."); continue
            R, t, rms, n = kabsch_recortado(np.array(PB), np.array(PA))
            print(f"\n  {n}/{len(PA)} puntos usados → error RMS {rms * 1000:.1f} mm")
            if rms > 0.04:
                print("  ⚠️ Error alto (>40 mm): amplía el volumen recorrido o mejora la iluminación; "
                      "con ChArUco sale más preciso.")
            return R, t, n, {"error_rms_m": rms}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metodo", choices=("charuco", "cuerpo"), default="charuco")
    ap.add_argument("--generar-tablero", action="store_true")
    ap.add_argument("--cuadros", type=int, nargs=2, default=(7, 5), metavar=("X", "Y"))
    ap.add_argument("--lado", type=float, default=0.04, help="lado del cuadrado IMPRESO (m)")
    ap.add_argument("--marcador", type=float, default=0.03, help="lado del marcador ArUco impreso (m)")
    ap.add_argument("--diccionario", default="DICT_4X4_50")
    ap.add_argument("--min-esquinas", type=int, default=8)
    ap.add_argument("--max-reproj", type=float, default=1.5)
    ap.add_argument("--brazo", choices=("derecho", "izquierdo"), default="derecho")
    ap.add_argument("--mxid-a"); ap.add_argument("--mxid-b")
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()

    if args.generar_tablero:
        board, _ = crear_tablero(args.cuadros, args.lado, args.marcador, args.diccionario)
        cv2.imwrite("tablero_charuco.png", board.generateImage((args.cuadros[0] * 200, args.cuadros[1] * 200), marginSize=40))
        print(f"Guardado tablero_charuco.png ({args.cuadros[0]}x{args.cuadros[1]}). Imprímelo para que cada "
              f"cuadrado mida {args.lado * 1000:.0f} mm y mídelo con una regla.")
        return

    from skill_shield.vision.multicamara import _ruta_extrinsecas
    cams = _abrir_las_dos(args)
    try:
        res = (calibrar_charuco if args.metodo == "charuco" else calibrar_cuerpo)(args, cams)
    finally:
        cv2.destroyAllWindows()
        for c in cams:
            c["dev"].close()
    if res is None:
        print("Cancelado: no se ha guardado nada."); return
    R, t, n, extra = res
    ruta = args.salida or _ruta_extrinsecas()
    guardar(ruta, R, t, cams[0]["mxid"], cams[1]["mxid"], args.metodo, n, extra)
    sep = np.linalg.norm(t)
    giro = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"\n✅ Guardado {ruta}\n   B respecto a A: giro {giro:.1f}°, distancia entre cámaras {sep:.2f} m")


if __name__ == "__main__":
    main()
