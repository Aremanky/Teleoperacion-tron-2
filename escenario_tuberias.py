"""
Escenario de montaje de tuberias de PVC para el TRON 2.

  - Techo con una red de tuberias ya instalada, colgada con varillas y abrazaderas.
  - Una de esas tuberias termina en un extremo libre al alcance del robot, marcado
    con un anillo: es el PUNTO DE CONEXION.
  - En una mesa delante del robot hay una pieza suelta: un tramo con codo de 90
    grados y una copa (manguito) en cada extremo.

La tarea: coger la pieza con una pinza (cerrar la mano cerca de ella), llevarla
hasta el punto de conexion y encajar uno de sus extremos en el tubo del techo.
Cuando un extremo llega cerca y bien alineado, la pieza encaja sola, se suelta
de la pinza y se queda INSTALADA. Si la sueltas antes, cae a la mesa; si cae al
suelo, al cabo de un segundo reaparece en su sitio de la mesa.

La simulacion es cinematica (no hay fisica), asi que el agarre tambien lo es:
al cerrar la pinza junto a la pieza, esta pasa a moverse solidaria con la pinza.

Todas las medidas estan en metros, en el marco del mundo del robot:
x hacia delante del robot, y a su izquierda, z hacia arriba.
"""
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

# =============================== GEOMETRIA ===================================
R_TUBO = 0.020          # PVC de 40 mm
R_COPA = 0.0245         # copa / manguito de los accesorios
LARGO_COPA = 0.045
SOLAPE = 0.030          # cuanto entra el tubo del techo dentro de la copa al encajar
L1, L2 = 0.25, 0.22     # largo de los dos tramos de la pieza suelta, desde el codo

Z_TECHO = 2.25

# Extremo libre del tubo del techo: punto y direccion hacia fuera del tubo
CONEXION_PUNTO = np.array([0.42, 0.05, 1.55])
CONEXION_EJE = np.array([0.0, -1.0, 0.0])

# Red ya instalada: (desde, hasta, radio). La primera acaba en el punto de conexion.
TUBOS_TECHO = [
    ((0.42, 0.05, 1.55), (0.42, 1.60, 1.55), R_TUBO),
    ((0.42, 1.60, 1.55), (2.40, 1.60, 1.55), R_TUBO),
    ((0.85, -2.00, 1.85), (0.85, 2.00, 1.85), R_TUBO),
    ((0.30, -1.20, 1.95), (2.40, -1.20, 1.95), 0.032),
]
CODOS_TECHO = [((0.42, 1.60, 1.55), R_TUBO)]
SEPARACION_SOPORTES = 0.60

MESA_CENTRO = (0.8, -0.22)     # centrada: la alcanzan los DOS brazos (antes 0.46, -0.22)
MESA_MEDIO = (0.20, 0.24)        # semiancho en x, y
MESA_ALTO = 0.95                 # altura del tablero

# ---------------- PIEZA DE LA MESA ----------------------------------------------
# Que pieza hay en la mesa. Se usa la primera opcion que no sea None:
#   RUTA_XML_PIEZA   -> un cuerpo de un .xml de MuJoCo, con sus mallas y sus colores
#                       (p.ej. el taladro del proyecto OrcaHand). RECOMENDADO.
#   RUTA_MALLA_PIEZA -> un .obj / .stl suelto (se escala, se centra y se apoya solo)
#   las dos a None   -> el codo de PVC de siempre
RUTA_XML_PIEZA = "taladro/escena_taladro.xml"
CUERPO_XML_PIEZA = "taladro"                 # cuerpo del .xml que es la pieza (con sus hijos)
GEOM_AGARRE_XML = "taladro_col_empunadura"   # capsula por donde se coge (None = su eje mas largo)

RUTA_MALLA_PIEZA = None          # p.ej. "modelos/mi_pieza.stl"
ESCALA_MALLA = "auto"            # "auto": se escala para que su lado mas largo mida TAMANO_PIEZA
                                 # o un numero: 0.001 si el archivo esta en mm, 1.0 si en metros
TAMANO_PIEZA = 0.25              # m (solo con ESCALA_MALLA = "auto")
COLOR_PIEZA = "0.95 0.45 0.10 1"

GIRO_PIEZA = (0.0, 0.0, 0.0)     # grados alrededor de x, y, z: como queda en la mesa
                                 # (taladro: (0, 0, 0) = de pie con la broca hacia delante;
                                 #  (0, 0, 180) = broca hacia el robot)
PIEZA_ENCAJA = None              # None = solo el codo de PVC se "instala" en el tubo del techo

# Pieza suelta: origen en el codo. Tramo 1 por +y local, tramo 2 por -z local.
EXTREMOS = [  # (punto local, direccion hacia fuera)
    (np.array([0.0, L1, 0.0]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, -L2]), np.array([0.0, 0.0, -1.0])),
]
# Sobre la mesa, tumbada: tramo 1 hacia +y del mundo, tramo 2 hacia +x
POS_INICIAL = np.array([0.34, -0.12, MESA_ALTO + R_COPA])   # antes y = -0.36: fuera del alcance del brazo izquierdo
ROT_INICIAL = np.array([[0.0, 0.0, -1.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0]])


def _leer_vertices(ruta):
    """Vertices (N x 3) de un .obj o un .stl (binario o de texto), en sus unidades."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No encuentro el modelo de la pieza: {os.path.abspath(ruta)}")
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".obj":
        with open(ruta, "r", errors="ignore") as f:
            v = [[float(x) for x in linea.split()[1:4]] for linea in f if linea.startswith("v ")]
        return np.array(v, dtype=float)
    if ext == ".stl":
        with open(ruta, "rb") as f:
            datos = f.read()
        n = int.from_bytes(datos[80:84], "little") if len(datos) >= 84 else -1
        if len(datos) == 84 + 50 * n:                       # STL binario
            tipo = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(datos, dtype=tipo, count=n, offset=84)["v"].reshape(-1, 3).astype(float)
        t = datos.decode(errors="ignore").split()           # STL de texto
        return np.array([[float(t[i + 1]), float(t[i + 2]), float(t[i + 3])]
                         for i, x in enumerate(t) if x == "vertex"], dtype=float)
    raise ValueError(f"Formato no soportado para la pieza: {ext} (usa .obj o .stl)")


def _rot_xyz(grados):
    ax, ay, az = np.radians(grados)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# =============================== REGLAS ======================================
D_AGARRE = 0.05          # m: distancia maxima de la pinza al eje del tubo para cogerlo
CIERRE = 0.35            # la mano cuenta como cerrada por debajo de esta apertura
SUELTA = 0.60            # y como abierta (suelta la pieza) por encima de esta
APERTURA_TUBO = 0.45     # la pinza no cierra mas alla del grosor del tubo
TOL_POS = 0.06           # m: tolerancia de posicion para encajar
TOL_ANG = 35.0           # grados: tolerancia de alineacion para encajar
D_AVISO = 0.15           # m: a esta distancia el anillo de conexion se pone amarillo
GRAVEDAD = 9.81
ESPERA_SUELO = 1.0       # s que se queda en el suelo antes de reaparecer en la mesa

# Grupo de visualizacion de MuJoCo donde van todos los geoms del escenario.
# El escenario siempre esta en el modelo, pero oculto hasta que se activa
# (tecla 1 en la ventana de la camara), que enciende este grupo en el visor.
GRUPO_ESCENARIO = 5

# =============================== COLORES =====================================
PVC = "0.62 0.64 0.66 1"
PVC_COPA = "0.50 0.53 0.56 1"
TECHO = "0.78 0.78 0.76 1"
VARILLA = "0.35 0.36 0.38 1"
ABRAZADERA = "0.94 0.49 0.15 1"          # naranja Elecnor
MESA_TABLERO = "0.00 0.23 0.56 1"        # azul Elecnor
MESA_PATAS = "0.55 0.57 0.60 1"
ANILLO_LEJOS = np.array([0.94, 0.49, 0.15, 0.55])
ANILLO_CERCA = np.array([1.00, 0.85, 0.10, 0.75])
ANILLO_HECHO = np.array([0.20, 0.85, 0.30, 0.75])
DESTELLO = np.array([0.25, 0.90, 0.35, 1.0])


# =============================== UTILIDADES ==================================
def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def rot_entre(a, b):
    """Rotacion minima que lleva el vector unitario a hasta b."""
    a, b = _unit(a), _unit(b)
    eje = np.cross(a, b)
    s, c = np.linalg.norm(eje), float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        eje = np.cross(a, [1.0, 0.0, 0.0])        # 180 grados: cualquier eje perpendicular
        if np.linalg.norm(eje) < 1e-6:
            eje = np.cross(a, [0.0, 1.0, 0.0])
        s, c = 0.0, -1.0
        eje = _unit(eje)
        K = np.array([[0, -eje[2], eje[1]], [eje[2], 0, -eje[0]], [-eje[1], eje[0], 0]])
        return np.eye(3) + 2 * K @ K
    eje = eje / s
    K = np.array([[0, -eje[2], eje[1]], [eje[2], 0, -eje[0]], [-eje[1], eje[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def mat_a_quat(R):
    """Matriz de rotacion -> cuaternion (w, x, y, z), como lo usa MuJoCo."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _punto_segmento(p, a, b):
    """Punto de [a, b] mas cercano a p."""
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return a + t * ab


def _f(v):
    return " ".join(f"{x:.4f}" for x in v)


# =============================== PIEZA PROPIA ================================
def _quat_a_mat(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _vec(texto, defecto):
    return np.array([float(x) for x in texto.split()]) if texto else np.array(defecto, dtype=float)


def _esquinas(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _buscar_malla(fichero, base, carpeta_xml):
    """Las rutas de un .xml pueden estar mal (p.ej. '/piezas/cuerpo.obj' o un .obj fuera
    de su carpeta): se prueba tal cual, relativa al .xml y, si no, se busca por su nombre."""
    for c in (os.path.join(base, fichero.lstrip("/\\")), fichero):
        if os.path.isfile(c):
            return os.path.abspath(c)
    nombre = os.path.basename(fichero)
    for raiz, _, archivos in os.walk(carpeta_xml):
        if nombre in archivos:
            return os.path.abspath(os.path.join(raiz, nombre))
    raise FileNotFoundError(f"No encuentro la malla '{fichero}' de {RUTA_XML_PIEZA} "
                            f"(buscada dentro de {carpeta_xml})")


def _leer_pieza_xml(ruta, nombre_cuerpo, nombre_agarre):
    """Lee un cuerpo de un .xml de MuJoCo y lo 'aplana' en un solo cuerpo: sus geoms
    VISUALES (contype=0) con la pose relativa al cuerpo, sus mallas con la ruta ya
    corregida, la caja que ocupa y el segmento por donde se coge.
    Las articulaciones, inercias y geoms de colision se descartan: aqui no hay fisica."""
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No encuentro {os.path.abspath(ruta)}")
    root = ET.parse(ruta).getroot()
    carpeta = os.path.dirname(os.path.abspath(ruta))
    comp = root.find("compiler")
    base = os.path.join(carpeta, comp.get("meshdir", "") if comp is not None else "")
    mallas = {}
    for m in root.iter("mesh"):
        if m.get("file"):
            atr = {k: v for k, v in m.attrib.items() if k != "class"}
            nombre = atr.pop("name", os.path.splitext(os.path.basename(atr["file"]))[0])
            atr["file"] = _buscar_malla(atr["file"], base, carpeta)
            mallas[nombre] = atr

    cuerpo = next((b for b in root.iter("body") if b.get("name") == nombre_cuerpo), None)
    if cuerpo is None:
        raise RuntimeError(f"{ruta} no tiene ningun <body name=\"{nombre_cuerpo}\">")

    geoms, agarre = [], None

    def pose(el):
        for a in ("euler", "axisangle", "xyaxes", "zaxis"):
            if el.get(a):
                print(f"[aviso] {el.tag} '{el.get('name', '')}' usa '{a}': no soportado, se ignora su giro")
        R = _quat_a_mat(_vec(el.get("quat"), (1, 0, 0, 0))) if el.get("quat") else np.eye(3)
        return _vec(el.get("pos"), (0, 0, 0)), R

    def recorrer(b, P, R):
        nonlocal agarre
        for h in b:
            if h.tag == "geom":
                gp, gR = pose(h)
                Pg, Rg = P + R @ gp, R @ gR
                ft = _vec(h.get("fromto"), ()) if h.get("fromto") else None
                if ft is not None:
                    ft = np.r_[P + R @ ft[:3], P + R @ ft[3:]]
                if h.get("name") == nombre_agarre:
                    if ft is not None:
                        agarre = (ft[:3], ft[3:])
                    else:   # capsula/cilindro a lo largo de su z local
                        tam = _vec(h.get("size"), ())
                        mitad = float(tam[1]) if len(tam) > 1 else 0.0
                        agarre = (Pg - Rg[:, 2] * mitad, Pg + Rg[:, 2] * mitad)
                if h.get("contype") == "0":                     # visual
                    geoms.append(dict(tipo=h.get("type", "mesh" if h.get("mesh") else "sphere"),
                                      malla=h.get("mesh"), size=h.get("size"), rgba=h.get("rgba"),
                                      fromto=ft, P=Pg, R=Rg))
            elif h.tag == "body":
                bp, bR = pose(h)
                recorrer(h, P + R @ bp, R @ bR)

    recorrer(cuerpo, np.zeros(3), np.eye(3))
    if not geoms:
        raise RuntimeError(f"El cuerpo '{nombre_cuerpo}' no tiene geoms visuales (contype=\"0\")")

    # caja que ocupa la pieza (con los vertices reales de las mallas)
    puntos = []
    for g in geoms:
        if g["tipo"] == "mesh" and g["malla"] in mallas:
            atr = mallas[g["malla"]]
            v = _leer_vertices(atr["file"]) * _vec(atr.get("scale"), (1, 1, 1))
            puntos.append(v @ g["R"].T + g["P"])
        else:
            r = float(np.max(_vec(g["size"], (0.01,))))
            centro = [g["fromto"][:3], g["fromto"][3:]] if g["fromto"] is not None else [g["P"]]
            puntos += [np.array([c - r, c + r]) for c in centro]
    puntos = np.vstack(puntos)
    return mallas, geoms, puntos.min(axis=0), puntos.max(axis=0), agarre


def _apoyar_en_mesa(esquinas, R):
    """Posicion para que la pieza (girada con R) quede apoyada y centrada en la mesa."""
    centro = R @ esquinas.mean(axis=0)
    z = MESA_ALTO - min(float((R @ c)[2]) for c in esquinas)
    return np.array([MESA_CENTRO[0] - 0.06 - centro[0], MESA_CENTRO[1] - centro[1], z])


PIEZA_XML = None          # (mallas, geoms) si la pieza sale de un .xml
ESQUINAS_PIEZA = None     # 8 esquinas de la caja de la pieza (en sus coordenadas)
SEGMENTO_AGARRE = None    # (a, b): por donde se coge (en sus coordenadas)
ESCALA_REAL, MALLA_POS = 1.0, np.zeros(3)
NOMBRE_PIEZA = "Tuberia"

if RUTA_XML_PIEZA:
    _mallas, _geoms, _lo, _hi, SEGMENTO_AGARRE = _leer_pieza_xml(
        RUTA_XML_PIEZA, CUERPO_XML_PIEZA, GEOM_AGARRE_XML)
    PIEZA_XML = (_mallas, _geoms)
    NOMBRE_PIEZA = CUERPO_XML_PIEZA.replace("_", " ").capitalize()
    ESQUINAS_PIEZA = _esquinas(_lo, _hi)
    print(f"Pieza {NOMBRE_PIEZA} ({RUTA_XML_PIEZA}): {len(_geoms)} geoms, {len(_mallas)} mallas | "
          f"mide {np.round((_hi - _lo) * 100, 1)} cm"
          + ("" if SEGMENTO_AGARRE is not None else f" | sin '{GEOM_AGARRE_XML}': se coge por su eje mas largo"))
elif RUTA_MALLA_PIEZA:
    _v = _leer_vertices(RUTA_MALLA_PIEZA)
    if len(_v) == 0:
        raise ValueError(f"{RUTA_MALLA_PIEZA} no tiene vertices")
    _lo, _hi = _v.min(axis=0), _v.max(axis=0)
    _medidas = _hi - _lo
    ESCALA_REAL = (TAMANO_PIEZA / max(float(_medidas.max()), 1e-12) if ESCALA_MALLA == "auto"
                   else float(ESCALA_MALLA))
    MALLA_POS = -(_lo + _hi) / 2 * ESCALA_REAL        # el centro de la caja va al origen de la pieza
    _semi = _medidas * ESCALA_REAL / 2
    ESQUINAS_PIEZA = _esquinas(-_semi, _semi)
    NOMBRE_PIEZA = "Pieza"
    print(f"Pieza propia {RUTA_MALLA_PIEZA}: {len(_v)} vertices | en el archivo mide "
          f"{np.round(_medidas, 4)} (sus unidades) | escala {ESCALA_REAL:.5g} -> "
          f"{np.round(2 * _semi * 100, 1)} cm")

if ESQUINAS_PIEZA is not None:
    if SEGMENTO_AGARRE is None:                       # se coge por su eje mas largo
        _c = ESQUINAS_PIEZA.mean(axis=0)
        _semi = (ESQUINAS_PIEZA.max(axis=0) - ESQUINAS_PIEZA.min(axis=0)) / 2
        _u = np.eye(3)[int(np.argmax(_semi))] * _semi.max()
        SEGMENTO_AGARRE = (_c - _u, _c + _u)
    _a, _b = (np.asarray(x, dtype=float) for x in SEGMENTO_AGARRE)
    SEGMENTO_AGARRE = (_a, _b)
    EXTREMOS = [(_b, _unit(_b - _a)), (_a, _unit(_a - _b))]
    ROT_INICIAL = _rot_xyz(GIRO_PIEZA)
    POS_INICIAL = _apoyar_en_mesa(ESQUINAS_PIEZA, ROT_INICIAL)
if PIEZA_ENCAJA is None:
    PIEZA_ENCAJA = ESQUINAS_PIEZA is None


# =============================== XML =========================================
def anadir_al_modelo(root):
    """Anade el escenario al arbol XML del robot (se llama al cargar el modelo)."""
    mundo = root.find("worldbody")
    base = {"contype": "0", "conaffinity": "0", "group": str(GRUPO_ESCENARIO)}

    def geom(padre, **kw):
        atr = dict(base)
        atr.update({k: (v if isinstance(v, str) else _f(np.atleast_1d(v))) for k, v in kw.items()})
        return ET.SubElement(padre, "geom", atr)

    # techo (solo por delante del robot, para no tapar la camara del visor)
    geom(mundo, name="techo", type="box", pos=(1.30, 0.0, Z_TECHO + 0.03),
         size=(1.15, 2.20, 0.03), rgba=TECHO)

    # red instalada, con varillas roscadas y abrazaderas naranjas
    for i, (a, b, r) in enumerate(TUBOS_TECHO):
        a, b = np.array(a), np.array(b)
        geom(mundo, name=f"tubo_techo_{i}", type="cylinder", fromto=np.r_[a, b], size=r, rgba=PVC)
        largo = np.linalg.norm(b - a)
        eje = (b - a) / largo
        n = max(1, int(largo // SEPARACION_SOPORTES))
        for k in range(n):
            c = a + eje * (largo * (k + 0.5) / n)
            geom(mundo, type="cylinder", fromto=np.r_[c + [0, 0, r], [c[0], c[1], Z_TECHO]],
                 size=0.004, rgba=VARILLA)
            geom(mundo, type="cylinder", fromto=np.r_[c - eje * 0.012, c + eje * 0.012],
                 size=r + 0.006, rgba=ABRAZADERA)
    for c, r in CODOS_TECHO:
        geom(mundo, type="sphere", pos=c, size=r + 0.006, rgba=PVC_COPA)

    # anillo que marca el punto de conexion
    p = CONEXION_PUNTO - CONEXION_EJE * 0.02
    geom(mundo, name="punto_conexion", type="cylinder",
         fromto=np.r_[p - CONEXION_EJE * 0.006, p + CONEXION_EJE * 0.006],
         size=R_TUBO + 0.014, rgba=_f(ANILLO_LEJOS))

    # mesa
    cx, cy = MESA_CENTRO
    mx, my = MESA_MEDIO
    geom(mundo, name="mesa", type="box", pos=(cx, cy, MESA_ALTO - 0.015),
         size=(mx, my, 0.015), rgba=MESA_TABLERO)
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * (mx - 0.03), cy + sy * (my - 0.03)
            geom(mundo, type="cylinder", fromto=(x, y, 0.0, x, y, MESA_ALTO - 0.03),
                 size=0.015, rgba=MESA_PATAS)

    # pieza suelta: cuerpo mocap (se coloca a mano desde Python en cada ciclo)
    pieza = ET.SubElement(mundo, "body", {"name": "tubo_suelto", "mocap": "true",
                                          "pos": _f(POS_INICIAL), "quat": _f(mat_a_quat(ROT_INICIAL))})
    if PIEZA_XML is not None:      # pieza de un .xml (p.ej. el taladro): sus geoms visuales
        mallas, geoms_xml = PIEZA_XML
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        for nombre, atr in mallas.items():
            ET.SubElement(asset, "mesh", dict(atr, name="pieza_" + nombre))
        for k, g in enumerate(geoms_xml):
            # el nombre debe empezar por "tubo_suelto_" (asi la logica la reconoce)
            kw = dict(name=f"tubo_suelto_{k}", type=g["tipo"], rgba=g["rgba"] or COLOR_PIEZA, mass="0")
            if g["malla"]:
                kw["mesh"] = "pieza_" + g["malla"]
            if g["size"]:
                kw["size"] = g["size"]
            if g["fromto"] is not None:
                kw["fromto"] = g["fromto"]
            else:
                kw["pos"], kw["quat"] = g["P"], mat_a_quat(g["R"])
            geom(pieza, **kw)
        return
    if RUTA_MALLA_PIEZA:     # modelo propio: una malla en lugar de los cilindros
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "mesh", {"name": "pieza_propia",
                                      "file": os.path.abspath(RUTA_MALLA_PIEZA),
                                      "scale": _f([ESCALA_REAL] * 3)})
        # el nombre debe empezar por "tubo_suelto_" (asi se pinta de verde al encajar)
        geom(pieza, name="tubo_suelto_malla", type="mesh", mesh="pieza_propia",
             pos=MALLA_POS, rgba=COLOR_PIEZA)
        return
    geom(pieza, name="tubo_suelto_codo", type="sphere", pos=(0, 0, 0), size=R_COPA + 0.003, rgba=PVC_COPA)
    for k, (punto, eje) in enumerate(EXTREMOS):
        largo = np.linalg.norm(punto)
        geom(pieza, name=f"tubo_suelto_tramo{k}", type="cylinder",
             fromto=np.r_[[0, 0, 0], eje * (largo - LARGO_COPA)], size=R_TUBO, rgba=PVC)
        geom(pieza, name=f"tubo_suelto_copa{k}", type="cylinder",
             fromto=np.r_[eje * (largo - LARGO_COPA), eje * largo], size=R_COPA, rgba=PVC_COPA)
        geom(pieza, name=f"tubo_suelto_copacodo{k}", type="cylinder",
             fromto=np.r_[[0, 0, 0], eje * 0.035], size=R_COPA, rgba=PVC_COPA)


# =============================== LOGICA ======================================
class EscenarioTuberias:
    def __init__(self, robot):
        m, d = robot.m, robot.d
        self.m, self.d = m, d
        cuerpo = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tubo_suelto")
        if cuerpo < 0:
            raise RuntimeError("El modelo no tiene el escenario de tuberias cargado")
        self.id_mocap = int(m.body_mocapid[cuerpo])
        self.g_anillo = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "punto_conexion")
        self.g_pieza = [g for g in range(m.ngeom)
                        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("tubo_suelto_")]
        self.rgba_pieza = m.geom_rgba[self.g_pieza].copy()
        self._informar_malla()
        self.activa = False             # oculto y sin efecto hasta que se active
        self.reiniciar()

    def _informar_malla(self):
        """Con un modelo propio, imprime la caja que ocupa tal y como la ha cargado
        MuJoCo, en las coordenadas de la pieza (comprobacion del centrado y la escala)."""
        g = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "tubo_suelto_malla")
        if g < 0:
            return
        mid = int(self.m.geom_dataid[g])
        a, n = int(self.m.mesh_vertadr[mid]), int(self.m.mesh_vertnum[mid])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.m.geom_quat[g])
        v = self.m.mesh_vert[a:a + n] @ R.reshape(3, 3).T + self.m.geom_pos[g]
        print(f"\nPieza propia ({RUTA_MALLA_PIEZA}): ocupa en sus coordenadas (m)\n"
              f"   x [{v[:, 0].min():+.3f}, {v[:, 0].max():+.3f}]  "
              f"y [{v[:, 1].min():+.3f}, {v[:, 1].max():+.3f}]  "
              f"z [{v[:, 2].min():+.3f}, {v[:, 2].max():+.3f}]\n"
              f"   (comprobacion: deberia salir centrada en 0 y con medidas de centimetros)\n")

    def activar(self, visor=None):
        """Muestra el escenario (o lo reinicia si ya estaba a la vista)."""
        self.activa = True
        self.reiniciar()
        if visor is not None:
            with visor.lock():
                visor.opt.geomgroup[GRUPO_ESCENARIO] = 1

    # ---------------------------------------------------------------- estado
    def reiniciar(self):
        self.p, self.R = POS_INICIAL.copy(), ROT_INICIAL.copy()
        self.estado = "mesa"            # mesa | sujeta | cayendo | suelo | instalada
        self.sujeta = None              # lado del robot que la lleva
        self.rel_p = self.rel_R = None
        self.vz = 0.0
        self.t_suelo = 0.0
        self.armado = {"L": True, "R": True}
        self.dist = self.ang = None
        self.t_destello = 0.0
        self.m.geom_rgba[self.g_pieza] = self.rgba_pieza
        self._pintar_anillo(ANILLO_LEJOS)
        self._escribir()

    def _volver_a_la_mesa(self):
        """Reaparece en su sitio de la mesa, sin tocar lo ya instalado."""
        self.p, self.R = POS_INICIAL.copy(), ROT_INICIAL.copy()
        self.estado, self.sujeta, self.vz = "mesa", None, 0.0
        self._escribir()

    def apertura_minima(self, lado):
        """La pinza que sujeta la pieza no se cierra mas alla del grosor del tubo."""
        return APERTURA_TUBO if self.activa and self.sujeta == lado else 0.0

    # ---------------------------------------------------------------- geometria
    def _segmentos(self):
        if SEGMENTO_AGARRE is not None:
            a, b = SEGMENTO_AGARRE
            return [(self.p + self.R @ a, self.p + self.R @ b)]
        return [(self.p, self.p + self.R @ punto) for punto, _ in EXTREMOS]

    def _mas_cercano(self, q):
        mejores = [_punto_segmento(q, a, b) for a, b in self._segmentos()]
        c = min(mejores, key=lambda c: np.linalg.norm(q - c))
        return c, float(np.linalg.norm(q - c))

    def _mejor_extremo(self):
        """Extremo de la pieza mas cercano a encajar: (indice, distancia, angulo)."""
        mejor = None
        for k, (punto, eje) in enumerate(EXTREMOS):
            pe = self.p + self.R @ punto
            ang = np.degrees(np.arccos(np.clip(np.dot(self.R @ eje, -CONEXION_EJE), -1, 1)))
            dist = float(np.linalg.norm(pe - CONEXION_PUNTO))
            puntuacion = dist + 0.002 * ang
            if mejor is None or puntuacion < mejor[3]:
                mejor = (k, dist, float(ang), puntuacion)
        return mejor[:3]

    def _encajar(self, k):
        """Coloca la pieza exactamente encajada por su extremo k (conserva el giro
        alrededor del eje del tubo, como pasa con un manguito de PVC real)."""
        punto, eje = EXTREMOS[k]
        self.R = rot_entre(self.R @ eje, -CONEXION_EJE) @ self.R
        destino = CONEXION_PUNTO - CONEXION_EJE * SOLAPE
        self.p = destino - self.R @ punto

    def _altura_apoyo(self):
        """Altura de la mesa si la pieza esta encima de ella; si no, el suelo."""
        if ESQUINAS_PIEZA is not None:
            c = self.p + self.R @ ESQUINAS_PIEZA.mean(axis=0)
            dentro = (abs(c[0] - MESA_CENTRO[0]) < MESA_MEDIO[0] and abs(c[1] - MESA_CENTRO[1]) < MESA_MEDIO[1])
            return MESA_ALTO if dentro else 0.0
        c = self.p + self.R @ (sum(p for p, _ in EXTREMOS) / (len(EXTREMOS) + 1))
        dentro = (abs(c[0] - MESA_CENTRO[0]) < MESA_MEDIO[0] and abs(c[1] - MESA_CENTRO[1]) < MESA_MEDIO[1])
        return MESA_ALTO if dentro else 0.0

    def _z_minima(self):
        if ESQUINAS_PIEZA is not None:   # pieza propia: su caja, girada como este la pieza
            return float(min((self.p + self.R @ c)[2] for c in ESQUINAS_PIEZA))
        puntos = [self.p] + [self.p + self.R @ p for p, _ in EXTREMOS]
        return min(q[2] for q in puntos) - R_COPA

    # ---------------------------------------------------------------- ciclo
    def actualizar(self, robot, aperturas, dt):
        """Llamar en cada ciclo, con las posiciones del robot ya actualizadas.
        aperturas: {lado del robot: apertura 0..1 pedida por la mano del operador}."""
        if not self.activa:
            return
        for lado, a in aperturas.items():
            if a > SUELTA:
                self.armado[lado] = True

        if self.estado == "instalada":
            if self.t_destello > 0:
                self.t_destello -= dt
                if self.t_destello <= 0:
                    self.m.geom_rgba[self.g_pieza] = self.rgba_pieza
            return

        if self.sujeta is None:
            # coger: mano que se cierra (despues de haber estado abierta) junto a la pieza
            for lado, a in aperturas.items():
                if not self.armado.get(lado) or a >= CIERRE:
                    continue
                brazo = robot.brazos[lado]
                g = brazo.punto_agarre()
                c, dist = self._mas_cercano(g)
                self.armado[lado] = False   # cerrar lejos gasta el intento: hay que abrir y volver a cerrar
                if dist < D_AGARRE:
                    self.p = self.p + (g - c)       # centra el tubo entre las mordazas
                    Rg = brazo.orientacion()
                    self.rel_p, self.rel_R = Rg.T @ (self.p - g), Rg.T @ self.R
                    self.sujeta, self.estado, self.vz = lado, "sujeta", 0.0
                    break
            if self.estado == "cayendo":
                self.vz -= GRAVEDAD * dt
                self.p = self.p + np.array([0.0, 0.0, self.vz * dt])
                apoyo = self._altura_apoyo()
                if self._z_minima() <= apoyo:
                    self.p[2] += apoyo - self._z_minima()
                    self.vz = 0.0
                    self.estado = "mesa" if apoyo > 0 else "suelo"
                    self.t_suelo = ESPERA_SUELO
            elif self.estado == "suelo":        # tras un momento, vuelve a la mesa
                self.t_suelo -= dt
                if self.t_suelo <= 0:
                    self._volver_a_la_mesa()
                    return
        else:
            a = aperturas.get(self.sujeta)
            if a is not None and a > SUELTA:         # la mano se abre: se suelta
                self.sujeta, self.estado = None, "cayendo"
            else:                                    # va solidaria con la pinza
                brazo = robot.brazos[self.sujeta]
                Rg = brazo.orientacion()
                self.p = brazo.punto_agarre() + Rg @ self.rel_p
                self.R = Rg @ self.rel_R

        if self.estado == "sujeta" and PIEZA_ENCAJA:
            k, self.dist, self.ang = self._mejor_extremo()
            if self.dist < TOL_POS and self.ang < TOL_ANG:
                self._encajar(k)
                self.estado, self.sujeta = "instalada", None
                self.m.geom_rgba[self.g_pieza] = DESTELLO
                self.t_destello = 1.2
                self._pintar_anillo(ANILLO_HECHO)
            else:
                self._pintar_anillo(ANILLO_CERCA if self.dist < D_AVISO else ANILLO_LEJOS)
        else:
            self.dist = self.ang = None
            self._pintar_anillo(ANILLO_LEJOS)
        self._escribir()

    def _pintar_anillo(self, rgba):
        if self.g_anillo >= 0:
            self.m.geom_rgba[self.g_anillo] = rgba

    def _escribir(self):
        self.d.mocap_pos[self.id_mocap] = self.p
        self.d.mocap_quat[self.id_mocap] = mat_a_quat(self.R)

    # ---------------------------------------------------------------- texto
    def texto(self):
        if not self.activa:
            return ""
        nombres = {"L": "IZQ", "R": "DER"}
        if self.estado == "instalada":
            return f"{NOMBRE_PIEZA}: INSTALADA  (1 = empezar de nuevo)"
        if self.estado == "sujeta":
            s = f"{NOMBRE_PIEZA}: en la mano {nombres[self.sujeta]}"
            if self.dist is not None:
                s += f" | conexion a {self.dist * 100:4.1f} cm, {self.ang:3.0f} deg"
                s += f" (encaja < {TOL_POS * 100:.0f} cm y < {TOL_ANG:.0f} deg)"
            return s
        if self.estado == "cayendo":
            return f"{NOMBRE_PIEZA}: cayendo"
        if self.estado == "suelo":
            return f"{NOMBRE_PIEZA}: se ha caido al suelo, vuelve a la mesa..."
        return f"{NOMBRE_PIEZA}: en la mesa  (cierra la mano junto a ella para cogerla)"