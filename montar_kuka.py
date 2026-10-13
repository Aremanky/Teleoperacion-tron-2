"""
Monta el KUKA KR210 L150 del proyecto (kuka_kr210_support) como SOPORTE FIJO del TRON 2, en
lugar de la columna del DACH_TRON2A: los cuerpos base_2_Link (columna) y base_3_Link (pie)
de robot_elecnor.xml.

No se toca el XML original. RobotTron2 llama a montar(root) cuando genera robot_fixed.xml,
igual que ya se anaden el escenario de tuberias y las OrcaHand.

MALLAS (oct-2026)
  MuJoCo lee el URDF sin problema, pero NO lee las mallas visuales .dae (Collada). La primera
  version usaba las .stl de colision, que son una version simplificada (por eso el KUKA salia
  "a bloques" y de un solo color). Ahora las .dae se convierten UNA vez a .obj, separadas por
  color (cuerpo naranja, motores negros...), y se guardan en
  kuka_kr210_support/meshes/kr210l150/visual_mujoco/ junto con colores.json.
  Necesita: pip install trimesh pycollada   (solo para la conversion; luego no hace falta)
  Al convertir se comprueba cada .dae contra su .stl de colision (unidades y eje "arriba"
  de Collada) y se corrige si hace falta: lo imprime para poder revisarlo.

BRIDA
  En kr210l150.urdf el marco tool0 NO esta girado como en el macro (.xacro): coincide con
  link_6, cuyo eje x es el que sale de la brida. Por eso la brida se define aqui como link_6
  + BRIDA_OFFSET y "mirar hacia arriba" es poner el eje x de link_6 en la vertical del torso.

COMO QUEDA MONTADO
  - Al arrancar, el torso (base_Link) queda EXACTAMENTE donde estaba (0, 0, 1.21): mesa,
    tuberias y calibracion de camaras siguen valiendo. Despues se puede desplazar (KUKA MOVIL).
  - El KUKA es un arbol aparte bajo <worldbody>, CONGELADO en una postura (sin juntas, sin
    masa, sin colisiones). La postura sale de una IK: la brida queda en ANCLAJE_TORSO, debajo
    del torso, mirando hacia arriba, con la base del KUKA en BASE_KUKA_POS.

KUKA MOVIL (oct-2026, ControlKuka)
  En marcha, el teleop puede DESPLAZAR al robot con el KUKA (flechas y +/-): el torso se
  traslada, nunca gira, y los eslabones del KUKA se recolocan con la misma IK de arriba,
  partiendo de la postura anterior. Sigue sin haber juntas: se escriben m.body_pos y
  m.body_quat de los cuerpos del KUKA y del torso, asi que nq no cambia y las grabaciones
  siguen valiendo. Se rechaza un movimiento si la IK no llega, si se sale de LIMITES_DESPLAZ
  o si el KUKA o la cabeza del robot pasarian del techo del escenario (ControlKuka.techo).

Prueba rapida (genera el modelo y abre el visor):
    python montar_kuka.py
"""
import json
import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
PAQUETE_KUKA = os.path.join(RAIZ, "kuka_kr210_support")
URDF_KUKA = os.path.join(PAQUETE_KUKA, "urdf", "kr210l150.urdf")
CARPETA_MUJOCO = os.path.join(PAQUETE_KUKA, "meshes", "kr210l150", "visual_mujoco")

# ------------------------------- montaje ---------------------------------------
CUERPOS_SOPORTE = ("base_2_Link",)     # se quitan (base_3_Link cuelga de base_2_Link)
CUERPO_TORSO = "base_Link"
# Punto del torso (en sus coordenadas) donde apoya la brida: donde empezaba la columna
ANCLAJE_TORSO = np.array([0.00056, -0.00009, -0.07285])
# Base del KUKA en el mundo (x delante del robot, y a su izquierda, z arriba).
# (oct-2026) -1.40 -> -2.00: con la brida hacia arriba bajo el torso y la base tan cerca, el KUKA
# quedaba tan plegado que su muneca (a5) tenia que doblar ~130 grados y su limite es 125: la brida
# no llegaba (17 mm, 4 grados de error) y el KUKA movil no podia moverse. Mas lejos, el antebrazo
# va mas horizontal y a5 queda dentro de su rango.
BASE_KUKA_POS = np.array([-2.00, 0.0, 0.0])
BASE_KUKA_YAW_DEG = 0.0                # 0 = el KUKA mira hacia +x (hacia donde mira el TRON)
# Postura de partida de la IK (a1..a6, grados): brazo hacia delante y muneca hacia arriba
SEMILLA_DEG = (0.0, 0.0, -20.0, 0.0, -70.0, 0.0)
LINK_BRIDA = "link_6"
BRIDA_OFFSET = np.array([0.0375, 0.0, -0.00023924])   # link_6 -> cara de la brida (del URDF)
GIRO_BRIDA_DEG = 0.0                   # giro del torso sobre el eje de la brida (cambia la postura)
TOL_POS = 0.005                        # m: error de la brida a partir del cual se avisa
# Ajuste del contacto brida-torso con las MALLAS (no con los origenes de los cuerpos, que
# dejaban un hueco): la cara de la brida es el punto mas alejado de la malla de link_6 y el
# apoyo es el punto mas bajo de la malla del torso dentro del circulo de la brida.
AJUSTE_CONTACTO = True
MALLA_TORSO = os.path.join(RAIZ, "tron2a", "DACH_TRON2A", "meshes", "base_Link.STL")
SOLAPE = 0.003                         # m que la brida "entra" en el torso para que no se vea junta
# Colores: "dae" = los que traen las mallas visuales (naranja + negro)
#          "urdf" = el material del URDF (gris 0.753, el que se ve en RViz)
#          o un rgba fijo, p.ej. "0.75 0.75 0.75 1"
COLORES = "dae"
PREFIJO = "kuka_"                      # para no chocar con nombres del TRON (base_link/base_Link)
COLOR_POR_DEFECTO = "0.95 0.42 0.0 1"  # si una malla no trae color

# ------------------------------- KUKA movil ------------------------------------
PASO_KUKA = 0.02                       # m que se desplaza el torso por pulsacion
# Recorrido permitido del torso respecto a su sitio de partida (m, ejes del mundo):
# (x atras, x delante), (y derecha, y izquierda), (z abajo, z arriba)
LIMITES_DESPLAZ = ((-0.60, 0.80), (-0.80, 0.80), (-0.50, 1.20))
ITER_IK_MOVIL = 80                     # iteraciones de la IK al mover (parte de la postura actual)
SALTO_MAX_JUNTA = 0.5                  # rad: un paso que gire mas una junta del KUKA se rechaza
HOLGURA_TECHO = 0.03                   # m que el KUKA y la cabeza del robot dejan bajo el techo

_MONTAJE = None   # lo rellena montar(): lo que necesita ControlKuka para mover el KUKA en marcha


# ------------------------------- utilidades ------------------------------------
def _f(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def _mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def _rot(eje, grados):
    a = np.radians(grados)
    c, s = np.cos(a), np.sin(a)
    if eje == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if eje == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _rpy(texto):
    r, p, y = (float(x) for x in (texto or "0 0 0").split())
    return _rot("z", np.degrees(y)) @ _rot("y", np.degrees(p)) @ _rot("x", np.degrees(r))


def _pose_cuerpo(el):
    """Pose (pos, R) de un <body> respecto a su padre. Solo pos/quat."""
    for a in ("euler", "axisangle", "xyaxes", "zaxis"):
        if el.get(a):
            raise ValueError(f"{el.get('name')}: usa '{a}', montar_kuka solo entiende pos/quat")
    p = np.array([float(x) for x in el.get("pos", "0 0 0").split()])
    R = _mat([float(x) for x in el.get("quat").split()]) if el.get("quat") else np.eye(3)
    return p, R


def _leer_vertices(ruta):
    """Vertices (N x 3) de un .obj o un .stl (binario o de texto)."""
    if ruta.lower().endswith(".obj"):
        with open(ruta, errors="ignore") as f:
            return np.array([[float(x) for x in l.split()[1:4]] for l in f if l.startswith("v ")])
    with open(ruta, "rb") as f:
        datos = f.read()
    n = int.from_bytes(datos[80:84], "little") if len(datos) >= 84 else -1
    if len(datos) == 84 + 50 * n:
        tipo = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        return np.frombuffer(datos, dtype=tipo, count=n, offset=84)["v"].reshape(-1, 3).astype(float)
    t = datos.decode(errors="ignore").split()
    return np.array([[float(t[i + 1]), float(t[i + 2]), float(t[i + 3])]
                     for i, x in enumerate(t) if x == "vertex"])


def _ruta_paquete(nombre):
    """package://kuka_kr210_support/... -> ruta en disco."""
    pre = "package://kuka_kr210_support/"
    return os.path.join(PAQUETE_KUKA, nombre[len(pre):]) if nombre.startswith(pre) else nombre


# ------------------------------- .dae -> .obj ----------------------------------
def _color(malla):
    """rgba (0..1) de una malla de trimesh, o None."""
    v = malla.visual
    for obtener in (lambda: v.material.main_color, lambda: v.main_color):
        try:
            c = np.asarray(obtener(), dtype=float).ravel()
            if c.size >= 3:
                c = c[:4] / (255.0 if c.max() > 1.0 else 1.0)
                return np.r_[c[:3], 1.0]
        except Exception:
            pass
    return None


def _convertir_visuales(raiz_urdf):
    """Convierte cada .dae visual a uno o varios .obj (uno por color) en CARPETA_MUJOCO.
    Devuelve {nombre_dae: [(ruta_obj, rgba), ...]}. Si ya estan convertidos, solo los lee."""
    manifiesto = os.path.join(CARPETA_MUJOCO, "colores.json")
    if os.path.isfile(manifiesto):
        with open(manifiesto, encoding="utf-8") as f:
            datos = json.load(f)
        if all(os.path.isfile(os.path.join(CARPETA_MUJOCO, o)) for v in datos.values() for o, _ in v):
            return {k: [(os.path.join(CARPETA_MUJOCO, o), c) for o, c in v] for k, v in datos.items()}
    try:
        import trimesh
    except ImportError as e:
        raise ImportError("Para convertir las mallas .dae del KUKA hace falta (una sola vez): "
                          "pip install trimesh pycollada") from e
    os.makedirs(CARPETA_MUJOCO, exist_ok=True)
    datos = {}
    for link in raiz_urdf.findall("link"):
        vis, col = link.find("visual"), link.find("collision")
        if vis is None or vis.find("geometry/mesh") is None:
            continue
        dae = _ruta_paquete(vis.find("geometry/mesh").get("filename"))
        base = os.path.splitext(os.path.basename(dae))[0]
        escena = trimesh.load(dae, force="scene")
        piezas = []                      # (vertices, caras, rgba) en el marco del .dae
        for nodo in escena.graph.nodes_geometry:
            T, nombre_g = escena.graph[nodo]
            g = escena.geometry[nombre_g]
            if not hasattr(g, "faces") or len(g.faces) == 0:
                continue
            v = (np.c_[g.vertices, np.ones(len(g.vertices))] @ np.asarray(T).T)[:, :3]
            piezas.append((v, np.asarray(g.faces), _color(g)))
        if not piezas:
            raise RuntimeError(f"{dae}: no tiene mallas")

        # --- correccion de unidades / eje arriba, comparando con la .stl de colision ---
        corr, escala = np.eye(3), 1.0
        if col is not None and col.find("geometry/mesh") is not None:
            stl = trimesh.load(_ruta_paquete(col.find("geometry/mesh").get("filename")), force="mesh")
            o_col = col.find("origin")
            R_col = _rpy(o_col.get("rpy") if o_col is not None else None)
            ref = stl.vertices @ R_col.T
            o_vis = vis.find("origin")
            R_vis = _rpy(o_vis.get("rpy") if o_vis is not None else None)
            todos = np.vstack([p[0] for p in piezas])
            mejor = np.inf
            for e in (1.0, 0.001, 0.01, 0.0254):
                for C in (np.eye(3), _rot("x", 90), _rot("x", -90), _rot("x", 180)):
                    w = (todos * e) @ C.T @ R_vis.T
                    dif = np.abs(w.min(0) - ref.min(0)).sum() + np.abs(w.max(0) - ref.max(0)).sum()
                    if dif < mejor - 1e-9:
                        mejor, corr, escala = dif, C, e
            nota = "" if escala == 1.0 and np.allclose(corr, np.eye(3)) else "  <- CORREGIDA"
            print(f"[kuka] {base}.dae: escala {escala:g}, giro de ajuste "
                  f"{np.round(np.degrees(np.arccos(np.clip((np.trace(corr) - 1) / 2, -1, 1))))} deg, "
                  f"diferencia con la colision {mejor * 100:.1f} cm{nota}")
            if mejor > 0.10:
                print(f"[kuka] AVISO: {base}.dae no cuadra con su colision (>10 cm): revisa la malla")

        # --- un .obj por color ---
        grupos = {}
        for v, caras, c in piezas:
            clave = tuple(np.round(c, 3)) if c is not None else None
            grupos.setdefault(clave, []).append(((v * escala) @ corr.T, caras))
        datos[base] = []
        for k, (clave, lista) in enumerate(grupos.items()):
            v = np.vstack([x[0] for x in lista])
            desplaz = np.cumsum([0] + [len(x[0]) for x in lista[:-1]])
            caras = np.vstack([x[1] + d for x, d in zip(lista, desplaz)])
            nombre_obj = f"{base}_{k}.obj"
            trimesh.Trimesh(v, caras, process=False).export(os.path.join(CARPETA_MUJOCO, nombre_obj))
            rgba = _f(clave) if clave is not None else COLOR_POR_DEFECTO
            datos[base].append((nombre_obj, rgba))
        print(f"[kuka] {base}.dae -> {len(grupos)} malla(s): {[c for _, c in datos[base]]}")
    with open(manifiesto, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=1)
    return {k: [(os.path.join(CARPETA_MUJOCO, o), c) for o, c in v] for k, v in datos.items()}


# ------------------------------- KUKA ------------------------------------------
def _cargar_kuka():
    """Carga el URDF del proyecto en MuJoCo con las mallas VISUALES convertidas.
    Devuelve (modelo, MJCF guardado, {nombre de malla: rgba})."""
    if not os.path.isfile(URDF_KUKA):
        raise FileNotFoundError(f"No encuentro el URDF del KUKA: {URDF_KUKA}")
    raiz = ET.parse(URDF_KUKA).getroot()
    objs = _convertir_visuales(raiz)
    colores, objs_link = {}, {}
    for link in raiz.findall("link"):
        for col in link.findall("collision"):     # solo queremos lo visual
            link.remove(col)
        vis = link.find("visual")
        if vis is None or vis.find("geometry/mesh") is None:
            continue
        base = os.path.splitext(os.path.basename(vis.find("geometry/mesh").get("filename")))[0]
        color_urdf = vis.find("material/color")
        color_urdf = color_urdf.get("rgba") if color_urdf is not None else COLOR_POR_DEFECTO
        o = vis.find("origin")
        objs_link[link.get("name")] = ([r for r, _ in objs[base]], _rpy(o.get("rpy") if o is not None else None),
                                       np.array([float(x) for x in (o.get("xyz") if o is not None else "0 0 0").split()]))
        link.remove(vis)
        for ruta, rgba in objs[base]:             # una <visual> por color, misma <origin>
            nueva = ET.fromstring(ET.tostring(vis))
            nueva.find("geometry/mesh").set("filename", ruta)
            for mat in nueva.findall("material"):
                nueva.remove(mat)
            link.append(nueva)
            colores[os.path.splitext(os.path.basename(ruta))[0]] = (
                rgba if COLORES == "dae" else color_urdf if COLORES == "urdf" else COLORES)
    # fusestatic=false: si no, MuJoCo funde los cuerpos sin junta y cambian los nombres
    ext = ET.Element("mujoco")
    ET.SubElement(ext, "compiler", {"strippath": "false", "discardvisual": "false",
                                    "fusestatic": "false"})
    raiz.insert(0, ext)
    carpeta = tempfile.mkdtemp(prefix="kuka_")
    ruta_urdf = os.path.join(carpeta, "kuka.urdf")
    ET.ElementTree(raiz).write(ruta_urdf)
    m = mujoco.MjModel.from_xml_path(ruta_urdf)
    ruta_mjcf = os.path.join(carpeta, "kuka.xml")
    mujoco.mj_saveLastXML(ruta_mjcf, m)
    if not np.all(m.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE)):
        raise RuntimeError("El KUKA tiene juntas que no son bisagras: revisa el URDF")
    return m, ET.parse(ruta_mjcf).getroot(), colores, objs_link


def _contacto(objs_link, anclaje_torso):
    """(offset de la brida en link_6, anclaje en el torso) ajustados con las mallas para que
    la brida toque el torso sin hueco."""
    rutas, R_vis, p_vis = objs_link[LINK_BRIDA]
    v6 = np.vstack([_leer_vertices(r) for r in rutas]) @ R_vis.T + p_vis   # en el marco de link_6
    offset = BRIDA_OFFSET.copy()
    offset[0] = float(v6[:, 0].max())                                      # cara de la brida (eje x)
    radio = float(np.percentile(np.linalg.norm(v6[:, 1:] - offset[1:], axis=1), 95))
    anclaje = np.array(anclaje_torso, dtype=float)
    if os.path.isfile(MALLA_TORSO):
        vt = _leer_vertices(MALLA_TORSO)
        cerca = np.linalg.norm(vt[:, :2] - anclaje[:2], axis=1) < radio
        if cerca.any():
            anclaje[2] = float(vt[cerca, 2].min()) + SOLAPE
        else:
            print(f"[kuka] aviso: la malla del torso no tiene puntos sobre la brida; uso ANCLAJE_TORSO")
    else:
        print(f"[kuka] aviso: no encuentro {MALLA_TORSO}; uso ANCLAJE_TORSO")
    print(f"[kuka] contacto: cara de la brida a {offset[0] * 1000:.1f} mm de link_6 "
          f"(URDF: {BRIDA_OFFSET[0] * 1000:.1f}), radio {radio * 1000:.0f} mm | "
          f"apoyo en el torso z = {anclaje[2] * 1000:+.1f} mm (antes {anclaje_torso[2] * 1000:+.1f})")
    return offset, anclaje


def _postura(m, b, offset, p_obj, R_obj, semillas=None, iteraciones=400):
    """IK (minimos cuadrados amortiguados, varias semillas): lleva el punto 'offset' del
    cuerpo b a p_obj con la orientacion R_obj (marco de la base del KUKA).
    semillas: posturas de partida (None = SEMILLA_DEG + 40 al azar, para el montaje; al mover
    en marcha se pasa solo la postura actual, que esta muy cerca de la solucion).
    Devuelve (q, error pos [m], error ang [rad])."""
    d = mujoco.MjData(m)
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    lim = m.jnt_limited.astype(bool)
    if semillas is None:
        rng = np.random.default_rng(0)
        semillas = [np.radians(SEMILLA_DEG[:m.nq])]
        semillas += [np.where(lim, rng.uniform(lo, hi), rng.uniform(-np.pi, np.pi, m.nq)) for _ in range(40)]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mejor = (np.inf, None, np.inf, np.inf)
    for q in semillas:
        q = np.array(q, dtype=float)
        for _ in range(iteraciones):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.xmat[b].reshape(3, 3)
            p = d.xpos[b] + R @ offset
            ep = p_obj - p
            qe = _quat(R_obj @ R.T)
            if qe[0] < 0:
                qe = -qe
            er = np.zeros(3)
            mujoco.mju_quat2Vel(er, qe, 1.0)
            if np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-4:
                break
            mujoco.mj_jac(m, d, jacp, jacr, p, b)
            J = np.vstack([jacp, jacr])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[ep, er])
            q = q + np.clip(dq, -0.2, 0.2)
            q[lim] = np.clip(q[lim], lo[lim], hi[lim])
        coste = np.linalg.norm(ep) + 0.1 * np.linalg.norm(er)
        if coste < mejor[0]:
            mejor = (coste, q.copy(), float(np.linalg.norm(ep)), float(np.linalg.norm(er)))
    return mejor[1], mejor[2], mejor[3]


# ------------------------------- montaje en el XML ------------------------------
def montar(root):
    """Quita la columna del TRON 2 y pone el KUKA congelado sosteniendo el torso."""
    mundo = root.find("worldbody")
    torso = next((b for b in mundo.findall("body") if b.get("name") == CUERPO_TORSO), None)
    if torso is None:
        raise RuntimeError(f"No encuentro <body name=\"{CUERPO_TORSO}\"> colgando de <worldbody>")

    # 1) fuera el soporte original
    quitados = []
    for padre in root.iter():
        for h in list(padre):
            if h.tag == "body" and h.get("name") in CUERPOS_SOPORTE:
                padre.remove(h)
                quitados.append(h.get("name"))
    if not quitados:
        print(f"[kuka] aviso: no he encontrado {CUERPOS_SOPORTE} en el modelo")

    # 2) donde tiene que quedar la brida (mundo): debajo del torso, con su eje de salida
    #    (x de link_6) hacia arriba del torso
    m, mjcf, colores, objs_link = _cargar_kuka()
    offset, anclaje_local = BRIDA_OFFSET, ANCLAJE_TORSO
    if AJUSTE_CONTACTO:
        offset, anclaje_local = _contacto(objs_link, ANCLAJE_TORSO)
    p_t, R_t = _pose_cuerpo(torso)
    anclaje = p_t + R_t @ anclaje_local
    R_brida = R_t @ _rot("z", GIRO_BRIDA_DEG) @ _rot("y", -90)   # columna x = z del torso
    R_b, p_b = _rot("z", BASE_KUKA_YAW_DEG), np.asarray(BASE_KUKA_POS, dtype=float)

    # 3) postura del KUKA
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, LINK_BRIDA)
    if b < 0:
        raise RuntimeError(f"El URDF del KUKA no tiene el cuerpo {LINK_BRIDA}")
    q, ep, er = _postura(m, b, offset, R_b.T @ (anclaje - p_b), R_b.T @ R_brida)
    print(f"[kuka] soporte quitado: {quitados} | base KUKA en {np.round(p_b, 3)} | "
          f"brida en {np.round(anclaje, 3)}")
    print(f"[kuka] postura a1..a6 (deg): {np.round(np.degrees(q), 1)} | "
          f"error brida {ep * 1000:.1f} mm, {np.degrees(er):.2f} deg")
    if ep > TOL_POS or np.degrees(er) > 1.0:
        print("[kuka] AVISO: el KUKA no llega bien al torso desde BASE_KUKA_POS. Si alguna junta "
              "esta en su limite (a5 = +-125: KUKA demasiado plegado), ALEJA la base (x mas negativa).")
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    global _MONTAJE                     # para ControlKuka (mover el KUKA en marcha)
    _MONTAJE = dict(m=m, b=b, offset=np.array(offset, dtype=float), R_b=R_b, p_b=p_b,
                    R_brida=R_brida, anclaje=np.array(anclaje, dtype=float), q=np.array(q, dtype=float),
                    ep=float(ep), er=float(er))

    # 4) mallas del KUKA
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    nombres_malla = {}
    for malla in mjcf.iter("mesh"):
        if malla.get("file"):
            atr = {k: v for k, v in malla.attrib.items() if k != "class"}
            original = atr.get("name", os.path.splitext(os.path.basename(atr["file"]))[0])
            atr["name"] = PREFIJO + original
            nombres_malla[original] = os.path.splitext(os.path.basename(atr["file"]))[0]
            ET.SubElement(asset, "mesh", atr)

    # 5) cuerpos congelados (sin juntas): cada uno con la pose que tiene en esa postura
    soporte = ET.SubElement(mundo, "body", {"name": PREFIJO + "soporte", "pos": _f(p_b), "quat": _f(_quat(R_b))})

    def copiar_geoms(origen, destino):
        for g in origen.findall("geom"):
            atr = {k: v for k, v in g.attrib.items()
                   if k not in ("class", "material", "contype", "conaffinity", "group", "rgba", "density", "mass")}
            if "name" in atr:
                atr["name"] = PREFIJO + atr["name"]
            rgba = COLOR_POR_DEFECTO
            if "mesh" in atr:
                rgba = colores.get(nombres_malla.get(atr["mesh"], atr["mesh"]), COLOR_POR_DEFECTO)
                atr["mesh"] = PREFIJO + atr["mesh"]
                atr["type"] = "mesh"           # el tipo por defecto de este modelo es "sphere"
                atr.pop("size", None)
            atr.update(contype="0", conaffinity="0", group="1", density="0", rgba=rgba)
            ET.SubElement(destino, "geom", atr)

    mundo_k = mjcf.find("worldbody")
    copiar_geoms(mundo_k, soporte)          # por si el URDF dejo geoms en el mundo
    for cuerpo in mundo_k.iter("body"):
        nombre = cuerpo.get("name")
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nombre)
        destino = ET.SubElement(soporte, "body", {"name": PREFIJO + nombre, "pos": _f(d.xpos[i]),
                                                  "quat": _f(d.xquat[i])})
        copiar_geoms(cuerpo, destino)


class ControlKuka:
    """Mueve el KUKA en marcha: traslada el torso del TRON 2 (sin girarlo nunca) y recoloca los
    eslabones del KUKA con una IK. No anade juntas (nq no cambia): el KUKA y el torso son
    cuerpos sin juntas y se mueven escribiendo m.body_pos / m.body_quat.
    techo: altura (m) que no deben pasar el KUKA ni la cabeza del robot (la fija el escenario)."""

    def __init__(self, m, d):
        if _MONTAJE is None:
            raise RuntimeError("ControlKuka necesita que montar() haya montado antes el KUKA")
        M = _MONTAJE
        self.m, self.d = m, d
        self.mk = M["m"]
        self.dk = mujoco.MjData(self.mk)
        self.b, self.offset = M["b"], M["offset"]
        self.R_b, self.p_b, self.R_brida = M["R_b"], M["p_b"], M["R_brida"]
        self.anclaje0, self.q = M["anclaje"].copy(), M["q"].copy()
        self.ep, self.er = M["ep"], M["er"]    # error de la brida en la postura actual (m, rad)
        self.desplaz = np.zeros(3)          # cuanto se ha trasladado el torso desde el arranque
        self.techo = None
        CUERPO = mujoco.mjtObj.mjOBJ_BODY
        self.torso = mujoco.mj_name2id(m, CUERPO, CUERPO_TORSO)
        # pares (cuerpo del modelo del KUKA, cuerpo del modelo completo)
        self.pares = []
        for i in range(1, self.mk.nbody):
            j = mujoco.mj_name2id(m, CUERPO, PREFIJO + (mujoco.mj_id2name(self.mk, CUERPO, i) or ""))
            if j >= 0:
                self.pares.append((i, j))
        self.disponible = self.torso >= 0 and int(m.body_parentid[self.torso]) == 0 and bool(self.pares)
        if not self.disponible:
            print("[kuka] AVISO: no encuentro los cuerpos del KUKA o del torso en el modelo "
                  "compilado: el KUKA no se podra mover")
            return
        self.pos_torso0 = m.body_pos[self.torso].copy()
        self._vigilados = self._preparar_vigilados()

    # ---------------- comprobaciones ----------------
    def _preparar_vigilados(self):
        """Geoms que no deben pasar del techo: los del KUKA, el torso y la cabeza del robot.
        Con mallas, una muestra de sus vertices (coordenadas del geom); si no, su radio."""
        m = self.m
        CUERPO = mujoco.mjtObj.mjOBJ_BODY
        cuerpos = {j for _, j in self.pares} | {self.torso}
        cuerpos |= {b for b in range(m.nbody) if "head" in (mujoco.mj_id2name(m, CUERPO, b) or "").lower()}
        vig = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) not in cuerpos:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_dataid[g] >= 0:
                mid = int(m.geom_dataid[g])
                a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                vig.append((g, m.mesh_vert[a:a + n][::max(1, n // 400)].copy(), 0.0))
            else:
                vig.append((g, np.zeros((1, 3)), float(m.geom_rbound[g])))
        return vig

    def z_max(self):
        """Altura del punto mas alto del KUKA, el torso y la cabeza (con la cinematica actual)."""
        z = -np.inf
        for g, v, r in self._vigilados:
            R = self.d.geom_xmat[g].reshape(3, 3)
            z = max(z, float((v @ R.T)[:, 2].max() + self.d.geom_xpos[g][2] + r))
        return z

    # ---------------- movimiento ----------------
    def _aplicar(self, q, desplaz):
        """Coloca los eslabones del KUKA en la postura q y el torso desplazado 'desplaz'."""
        self.dk.qpos[:] = q
        mujoco.mj_kinematics(self.mk, self.dk)
        for i, j in self.pares:      # el padre de los cuerpos del KUKA es kuka_soporte = su base
            self.m.body_pos[j] = self.dk.xpos[i]
            self.m.body_quat[j] = self.dk.xquat[i]
        self.m.body_pos[self.torso] = self.pos_torso0 + desplaz
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def mover(self, delta):
        """Desplaza el torso 'delta' (m, ejes del mundo) sin girarlo. Devuelve (ok, mensaje)."""
        if not self.disponible:
            return False, "KUKA: no se puede mover en este modelo"
        lim = np.array(LIMITES_DESPLAZ, dtype=float)
        nuevo = np.clip(self.desplaz + np.asarray(delta, dtype=float), lim[:, 0], lim[:, 1])
        if np.allclose(nuevo, self.desplaz):
            return False, "KUKA: fin del recorrido permitido (LIMITES_DESPLAZ)"
        p_obj = self.R_b.T @ (self.anclaje0 + nuevo - self.p_b)
        q, ep, er = _postura(self.mk, self.b, self.offset, p_obj, self.R_b.T @ self.R_brida,
                             semillas=[self.q], iteraciones=ITER_IK_MOVIL)
        # Se acepta si la brida llega (TOL_POS, 1 grado) o, si ya arrancaba con algo de error,
        # si no empeora: asi se puede mover aunque el montaje no fuera perfecto, y se frena justo
        # donde el KUKA empieza a no llegar (p.ej. muneca en su limite al bajar).
        tol_p = max(TOL_POS, self.ep + 0.001)
        tol_a = max(np.radians(1.0), self.er + np.radians(0.2))
        if ep > tol_p or er > tol_a:
            return False, (f"KUKA: no llega ahi (error de la brida {ep * 1000:.0f} mm, "
                           f"{np.degrees(er):.1f} deg)")
        if np.abs(q - self.q).max() > SALTO_MAX_JUNTA:
            return False, "KUKA: ese paso le obliga a cambiar de postura de golpe"
        self._aplicar(q, nuevo)
        if self.techo is not None and self.z_max() > self.techo - HOLGURA_TECHO:
            self._aplicar(self.q, self.desplaz)          # se deshace
            return False, "KUKA: tocaria el techo"
        self.q, self.desplaz, self.ep, self.er = q, nuevo, ep, er
        return True, self.texto()

    def ir_a(self, desplaz):
        """Lleva el torso al desplazamiento dado en pasos de PASO_KUKA (reproduccion de sesiones)."""
        objetivo = np.asarray(desplaz, dtype=float)
        for _ in range(500):
            falta = objetivo - self.desplaz
            n = float(np.linalg.norm(falta))
            if n < 1e-6:
                return True
            ok, _ = self.mover(falta if n <= PASO_KUKA else falta * (PASO_KUKA / n))
            if not ok:
                return False
        return False

    def texto(self):
        x, y, z = self.desplaz * 100
        return (f"KUKA: torso desplazado x {x:+.0f} | y {y:+.0f} | z {z:+.0f} cm  (flechas y +/-)"
                + (f" | brida a {self.ep * 1000:.0f} mm" if self.ep > TOL_POS else ""))


def con_kuka(escenario=None):
    """Envuelve una funcion extra(root) (p.ej. el escenario) para montar antes el KUKA."""
    def extra(root):
        montar(root)
        if escenario is not None:
            escenario(root)
    return extra


if __name__ == "__main__":
    import sys

    import mujoco.viewer

    from robot_tron2 import quitar_base_flotante

    xml = sys.argv[1] if len(sys.argv) > 1 else "tron2a/DACH_TRON2A/xml/robot_elecnor.xml"
    ruta = quitar_base_flotante(xml, montar)
    modelo = mujoco.MjModel.from_xml_path(ruta)
    datos = mujoco.MjData(modelo)
    mujoco.mj_forward(modelo, datos)
    print(f"[kuka] modelo listo: {ruta} (nq={modelo.nq}, como antes)")
    mujoco.viewer.launch(modelo, datos)
