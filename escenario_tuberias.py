"""
Escenario de montaje de tuberias de PVC para el TRON 2.

  - Techo con una red de tuberias ya instalada, colgada con varillas y abrazaderas.
  - La red va por DETRAS del robot; solo un tramo llega por delante y termina en un
    extremo libre al alcance del robot, marcado con un anillo: es el PUNTO DE CONEXION.
  - En un banco de trabajo A LA DERECHA del robot hay una pieza suelta. Hay DOS, y el
    escenario se abre con una u otra (en el teleop, teclas 1 y 2, que hacen de interruptor):
      tecla 1 -> la HERRAMIENTA (el taladro, de RUTA_XML_PIEZA / RUTA_MALLA_PIEZA)
      tecla 2 -> el TUBO para colocar: un tramo con codo de 90 grados y una copa (manguito)
                 en cada extremo, que se instala en el punto de conexion
    Las dos estan siempre en el modelo (un cuerpo mocap cada una); la que no esta en juego
    se aparca bajo el suelo.

La tarea: coger la pieza con una pinza (cerrar la mano cerca de ella), llevarla
hasta el punto de conexion y encajar uno de sus extremos en el tubo del techo.
Cuando un extremo llega cerca y bien alineado, la pieza encaja sola, se suelta
de la pinza y se queda INSTALADA. Si la sueltas antes, cae a la mesa; si cae al
suelo, al cabo de un segundo reaparece en su sitio de la mesa.

La simulacion es cinematica (no hay fisica), asi que el agarre tambien lo es:
al cerrar la pinza junto a la pieza, esta pasa a moverse solidaria con la pinza.

Todas las medidas estan en metros, en el marco del mundo del robot:
x hacia delante del robot, y a su izquierda, z hacia arriba.

CAMBIOS (oct-2026, mesa):
  - La mesa pasa de estar DELANTE del robot a su DERECHA (y negativa), con el lado largo
    paralelo al robot. La pieza se deja en la mitad de la mesa mas cercana al robot
    (PIEZA_EN_MESA) para que la alcance el brazo derecho.
  - Mesa nueva: banco de trabajo con tablero de roble, tapete azul Elecnor, bastidor de
    acero antracita con faldones, franja naranja Elecnor, balda inferior y pies niveladores.
  - Al arrancar se imprime a que distancia queda la pieza del hombro derecho, comparada
    con el alcance del brazo (aviso si queda demasiado lejos).
  - Mesa de estilo nordico (roble con veta, patas de nogal con contera de laton). El tablero
    y la almohadilla de fieltro son UNA malla cada uno: con piezas solapadas en el mismo
    plano MuJoCo no sabia cual pintar delante y la textura parpadeaba y salia cortada.
  - Cada geom de la mesa lleva rgba explicito: el <default> del robot es semitransparente.
  - La red de tuberias va por DETRAS del robot (solo el tramo de conexion queda delante).
  - Techo continuo (un plano que solo se ve desde abajo, para no tapar la camara del visor).
    Con la columna, a Z_TECHO = 1.85 m, al alcance para taladrarlo; con el KUKA, a
    Z_TECHO_KUKA = 2.60 m, y es el KUKA el que sube al robot hasta el (flechas y +/-).

CAMBIOS (oct-2026, tuberias con detalle; ver tuberias_pvc.py):
  - Los tubos ya no son cilindros macizos: son tubos HUECOS de PVC (pared de 3 mm, interior
    oscuro, bisel en los extremos) con el marcaje impreso a lo largo. El extremo del punto de
    conexion se ve abierto.
  - Codos de verdad con sus copas (antes, esferas), copas de union en los tramos largos y los
    extremos libres de la red suben al techo con un codo y un embellecedor.
  - Abrazaderas isofonicas (banda naranja Elecnor con goma, orejetas y tornillos), varillas
    roscadas con tuercas y etiquetas "EVACUACION" con el sentido del flujo.
  - La pieza de PVC de la mesa (si se usa en lugar del taladro) se construye igual.
  - Todo es estetico: TUBOS_TECHO, CONEXION_PUNTO, EXTREMOS... y la logica no cambian. Las
    subidas al techo se guardan en RED_EXTRA y entran en la comprobacion del KUKA.
  - Para volver a los cilindros de antes: TRON2_TUBERIAS_DETALLE=0

CAMBIOS (oct-2026, interruptores):
  - El modelo lleva las DOS piezas: la herramienta (cuerpo "herramienta_suelta", geoms
    "herramienta_suelta_*") y el tubo (cuerpo "tubo_suelto", geoms "tubo_suelto_*"). Antes
    habia una sola y cambiarla obligaba a editar RUTA_XML_PIEZA.
  - EscenarioTuberias.alternar(pieza, visor): interruptor. Si el escenario ya esta abierto con
    esa pieza, lo quita (oculta el grupo y libera la pieza y el techo del KUKA); si no, lo abre
    con esa pieza (o cambia de pieza). activar(visor, pieza) y desactivar(visor) por separado.
  - PIEZA_ENCAJA vale ahora solo para la herramienta (None = no se instala); el tubo siempre.
"""
import os
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import mujoco
import numpy as np

try:                    # tuberias con detalle (huecas, codos, abrazaderas...): tuberias_pvc.py
    import tuberias_pvc
except ImportError as _e:
    print(f"[aviso] sin tuberias_pvc ({_e}): tuberias lisas")
    tuberias_pvc = None
TUBERIAS_DETALLADAS = tuberias_pvc is not None and tuberias_pvc.ACTIVA

# =============================== GEOMETRIA ===================================
R_TUBO = 0.020          # PVC de 40 mm
R_COPA = 0.0245         # copa / manguito de los accesorios
LARGO_COPA = 0.045
SOLAPE = 0.030          # cuanto entra el tubo del techo dentro de la copa al encajar
L1, L2 = 0.25, 0.22     # largo de los dos tramos de la pieza suelta, desde el codo

# Altura del techo segun el soporte del robot:
#  - columna (soporte por defecto): AL ALCANCE del robot, para taladrarlo con el brazo
#    levantado. Al arrancar se imprime hasta que altura llega la mano: si no llega, baja Z_TECHO.
#  - KUKA: alto, para que el KUKA no lo atraviese; para taladrarlo, el KUKA sube al robot
#    (flechas y +/- en el teleop). El KUKA no deja subir mas alla del techo.
Z_TECHO = 1.85
Z_TECHO_KUKA = 2.60
z_techo = Z_TECHO          # el que se usa: lo fija anadir_al_modelo() segun el soporte
# Extension del techo (x0, x1, y0, y1): UNA placa continua, tambien por encima del robot y
# del KUKA. Es un plano que mira hacia abajo: desde abajo se ve solido, pero desde arriba
# MuJoCo no lo dibuja, asi que no tapa la camara del visor.
TECHO_EXTENSION = (-3.20, 1.20, -2.20, 2.20)

# Extremo libre del tubo del techo: punto y direccion hacia fuera del tubo
CONEXION_PUNTO = np.array([0.42, 0.05, 1.55])
CONEXION_EJE = np.array([0.0, -1.0, 0.0])

# Red ya instalada: (desde, hasta, radio). La primera acaba en el punto de conexion.
# La red va POR DETRAS del robot (x negativa). Solo el tramo de conexion queda delante, al
# alcance de la mano: sale hacia la izquierda del robot y desde ahi corre hacia atras.
# Detras del robot esta el KUKA (base en x = -2.00, y = 0, ver montar_kuka.py): las tuberias
# dejan libre ese pasillo (|y| < 0.6) salvo la transversal, que cruza por detras de su base.
TUBOS_TECHO = [
    ((0.42, 0.05, 1.55), (0.42, 0.80, 1.55), R_TUBO),      # tramo de conexion (al alcance)
    ((0.42, 0.80, 1.55), (-3.00, 0.80, 1.55), R_TUBO),     # sigue hacia atras por la izquierda
    ((-2.85, -2.00, 1.68), (-2.85, 2.00, 1.68), R_TUBO),   # transversal, detras del KUKA
    ((-0.15, -1.20, 1.74), (-3.00, -1.20, 1.74), 0.032),   # gruesa, por la derecha
]
CODOS_TECHO = [((0.42, 0.80, 1.55), R_TUBO)]
SEPARACION_SOPORTES = 0.60

HOLGURA_KUKA = 0.05     # m: aviso al arrancar si una tuberia pasa mas cerca del KUKA
RED_EXTRA = []          # tramos que anade la red con detalle (subidas al techo): (a, b, r).
                        # Los rellena anadir_al_modelo(); entran en la comprobacion del KUKA.

# ---------------- MESA (banco de trabajo a la DERECHA del robot) -----------------
MESA_CENTRO = (0.15, -0.62)      # x, y del centro (antes 0.8, -0.22: delante del robot)
MESA_MEDIO = (0.36, 0.22)        # semiancho en x (a lo largo del robot) y en y
MESA_ALTO = 0.95                 # altura de la superficie de apoyo (encima del tapete)
# Donde se deja la pieza, respecto al centro de la mesa. +y = hacia el robot: asi
# queda en la mitad cercana, al alcance del brazo DERECHO.
PIEZA_EN_MESA = (0.0, 0.10)

# Medidas de la mesa (solo estetica: la logica usa MESA_CENTRO / MESA_MEDIO / MESA_ALTO)
# Estilo nordico: tablero de roble con veta y esquinas redondeadas, almohadilla de
# fieltro antracita con pespunte naranja Elecnor, patas de nogal inclinadas con contera
# de laton y una balda de listones.
# El tablero y el fieltro son cada uno UNA malla (losa de esquinas redondeadas con sus
# coordenadas de textura): nada de piezas solapadas en el mismo plano, que parpadeaban.
GROSOR_TABLERO = 0.036           # tablero de roble
RADIO_ESQUINA = 0.055            # esquinas redondeadas del tablero (vistas desde arriba)
GROSOR_TAPETE = 0.003            # almohadilla de fieltro encima del tablero
MARGEN_TAPETE = 0.045            # borde de madera que queda a la vista alrededor del tapete
PUNTADA = dict(margen=0.012,     # pespunte: distancia al borde del fieltro (m)
               largo=0.006,      # largo de cada puntada
               hueco=0.0035,     # hueco entre puntadas
               ancho=0.0014)     # grosor del hilo
RADIO_PATA = 0.019               # patas torneadas
RETRANQUEO_PATA = 0.075          # cuanto se meten las patas (arriba) respecto al borde
ABERTURA_PATA = (0.055, 0.040)   # cuanto se abren hacia fuera en el suelo (x, y)
LARGO_CONTERA = 0.055            # contera de laton al pie de cada pata
ALTO_FALDON = 0.060              # faldon de nogal bajo el tablero
Z_BALDA = 0.24                   # altura de la balda de listones
N_LISTONES = 5

CARPETA_TEXTURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decoracion")

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
                                 #  (0, 0, 180) = broca hacia atras;
                                 #  (0, 0, 90) = broca hacia el robot, que esta a +y)
PIEZA_ENCAJA = None              # la herramienta, se "instala" en el tubo del techo? (None = no;
                                 # el tubo para colocar se instala siempre)

# El TUBO para colocar (tecla 2): origen en el codo. Tramo 1 por +y local, tramo 2 por -z local.
EXTREMOS_TUBO = [  # (punto local, direccion hacia fuera)
    (np.array([0.0, L1, 0.0]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, -L2]), np.array([0.0, 0.0, -1.0])),
]
# Sobre la mesa, tumbado: tramo 1 hacia +y del mundo (hacia el robot), tramo 2 hacia +x
POS_INICIAL_TUBO = np.array([MESA_CENTRO[0] + PIEZA_EN_MESA[0] - 0.10,
                             MESA_CENTRO[1] + PIEZA_EN_MESA[1] - 0.12,
                             MESA_ALTO + R_COPA])
ROT_INICIAL_TUBO = np.array([[0.0, 0.0, -1.0],
                             [0.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0]])
APARCAMIENTO = np.array([0.0, 0.0, -30.0])   # donde se aparca la pieza que no esta en juego


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
# (teclas 1 y 2 en la ventana de la camara), que encienden y apagan este grupo en el visor.
GRUPO_ESCENARIO = 5

# =============================== COLORES =====================================
PVC = "0.62 0.64 0.66 1"
PVC_COPA = "0.50 0.53 0.56 1"
TECHO = "0.78 0.78 0.76 1"
VARILLA = "0.35 0.36 0.38 1"
ABRAZADERA = "0.94 0.49 0.15 1"          # naranja Elecnor
ANILLO_LEJOS = np.array([0.94, 0.49, 0.15, 0.55])
ANILLO_CERCA = np.array([1.00, 0.85, 0.10, 0.75])
ANILLO_HECHO = np.array([0.20, 0.85, 0.30, 0.75])
DESTELLO = np.array([0.25, 0.90, 0.35, 1.0])

# Materiales de la mesa: (rgba, specular, shininess, reflectance, textura o None)
# Las texturas se generan solas la primera vez (ver _generar_texturas) en CARPETA_TEXTURAS.
# Si no se pueden generar, el material se queda con su color liso.
MATERIALES_MESA = {
    "mesa_roble":   ("0.80 0.60 0.40 1", 0.30, 0.50, 0.03, "madera_roble.png"),   # barnizado
    "mesa_tapete":  ("0.21 0.22 0.23 1", 0.04, 0.05, 0.00, "fieltro"),            # fieltro, mate
    "mesa_nogal":   ("0.33 0.21 0.13 1", 0.30, 0.45, 0.02, None),
    "mesa_laton":   ("0.80 0.62 0.32 1", 0.90, 0.90, 0.15, None),
    "mesa_naranja": ("0.94 0.49 0.15 1", 0.40, 0.50, 0.00, None),                 # naranja Elecnor
}
LARGO_VETA = 0.9                 # m de tablero que ocupa la imagen de la veta (alto = 1/4)
COLORES_ROBLE = ((0.86, 0.68, 0.47), (0.66, 0.46, 0.28))       # veta clara / oscura (RGB 0..1)
COLOR_FIELTRO = (0.20, 0.21, 0.22)                             # antracita
COLOR_PUNTADA = (0.94, 0.49, 0.15)                             # naranja Elecnor


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
    """Posicion para que la pieza (girada con R) quede apoyada en la mesa, en el
    punto PIEZA_EN_MESA (respecto al centro del tablero)."""
    centro = R @ esquinas.mean(axis=0)
    z = MESA_ALTO - min(float((R @ c)[2]) for c in esquinas)
    return np.array([MESA_CENTRO[0] + PIEZA_EN_MESA[0] - centro[0],
                     MESA_CENTRO[1] + PIEZA_EN_MESA[1] - centro[1], z])


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

# Las piezas del escenario. Cada una: clave, tecla, nombre, cuerpo mocap y prefijo de sus geoms
# (la logica los busca asi), extremos [(punto, direccion hacia fuera)], segmento de agarre (o
# None: se coge por sus tramos), caja que ocupa (o None), pose inicial en la mesa y si encaja.
PIEZAS = {"tubo": SimpleNamespace(
    clave="tubo", tecla="2", nombre="Tubo", cuerpo="tubo_suelto", prefijo="tubo_suelto_",
    extremos=EXTREMOS_TUBO, segmento=None, esquinas=None,
    pos0=POS_INICIAL_TUBO, rot0=ROT_INICIAL_TUBO, encaja=True)}
if ESQUINAS_PIEZA is not None:                        # la herramienta (taladro o malla propia)
    if SEGMENTO_AGARRE is None:                       # se coge por su eje mas largo
        _c = ESQUINAS_PIEZA.mean(axis=0)
        _semi = (ESQUINAS_PIEZA.max(axis=0) - ESQUINAS_PIEZA.min(axis=0)) / 2
        _u = np.eye(3)[int(np.argmax(_semi))] * _semi.max()
        SEGMENTO_AGARRE = (_c - _u, _c + _u)
    _a, _b = (np.asarray(x, dtype=float) for x in SEGMENTO_AGARRE)
    SEGMENTO_AGARRE = (_a, _b)
    _rot = _rot_xyz(GIRO_PIEZA)
    PIEZAS["herramienta"] = SimpleNamespace(
        clave="herramienta", tecla="1", nombre=NOMBRE_PIEZA, cuerpo="herramienta_suelta",
        prefijo="herramienta_suelta_", extremos=[(_b, _unit(_b - _a)), (_a, _unit(_a - _b))],
        segmento=SEGMENTO_AGARRE, esquinas=ESQUINAS_PIEZA,
        pos0=_apoyar_en_mesa(ESQUINAS_PIEZA, _rot), rot0=_rot, encaja=bool(PIEZA_ENCAJA))
PIEZA_INICIAL = "herramienta" if "herramienta" in PIEZAS else "tubo"
# Compatibilidad (lo de antes, de la pieza por defecto)
EXTREMOS = PIEZAS[PIEZA_INICIAL].extremos
POS_INICIAL, ROT_INICIAL = PIEZAS[PIEZA_INICIAL].pos0, PIEZAS[PIEZA_INICIAL].rot0
PIEZA_ENCAJA = PIEZAS[PIEZA_INICIAL].encaja


# =============================== XML =========================================
def _medidas_tapete():
    """Semiancho en x, en y y radio de esquina de la almohadilla de fieltro."""
    sx, sy = MESA_MEDIO[0] - MARGEN_TAPETE, MESA_MEDIO[1] - MARGEN_TAPETE
    return sx, sy, max(RADIO_ESQUINA - MARGEN_TAPETE, 0.015)


def _contorno_redondeado(sx, sy, r, n_arco=12):
    """Puntos (N x 2) del contorno de un rectangulo de semilados sx, sy con esquinas de
    radio r, en sentido antihorario visto desde arriba (empieza en la esquina +x +y)."""
    r = min(r, sx, sy)
    pts = []
    for (ex, ey), a0 in (((1, 1), 0.0), ((-1, 1), 90.0), ((-1, -1), 180.0), ((1, -1), 270.0)):
        for a in np.radians(np.linspace(a0, a0 + 90.0, n_arco + 1)):
            pts.append((ex * (sx - r) + r * np.cos(a), ey * (sy - r) + r * np.sin(a)))
    return np.array(pts)


def _malla_losa(sx, sy, h, r, uv_cara, uv_canto, n_arco=12):
    """Malla cerrada de una losa de esquinas redondeadas, centrada en el origen, de
    semilados sx, sy y semigrosor h. Caras de arriba, abajo y canto con vertices propios
    (aristas vivas, esquinas suaves). uv_cara(x, y) y uv_canto(s, z) dan las coordenadas
    de textura (s = recorrido a lo largo del canto). Devuelve los atributos de <mesh>."""
    anillo = _contorno_redondeado(sx, sy, r, n_arco)
    n = len(anillo)
    V, N, T, F = [], [], [], []

    def cara(z, nz):
        base = len(V)
        V.append((0.0, 0.0, z)); N.append((0, 0, nz)); T.append(uv_cara(0.0, 0.0))
        for x, y in anillo:
            V.append((x, y, z)); N.append((0, 0, nz)); T.append(uv_cara(x, y))
        for i in range(n):
            a, b = base + 1 + i, base + 1 + (i + 1) % n
            F.append((base, a, b) if nz > 0 else (base, b, a))

    cara(h, 1)
    cara(-h, -1)
    # canto: el primer punto se repite al final para que la textura no de un salto
    c = np.vstack([anillo, anillo[:1]])
    centros = np.vstack([np.sign(c[:, 0]) * (sx - r), np.sign(c[:, 1]) * (sy - r)]).T
    normal = c - centros
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    s_acum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    base = len(V)
    for (x, y), (nx, ny), s_ in zip(c, normal, s_acum):
        V += [(x, y, h), (x, y, -h)]
        N += [(nx, ny, 0), (nx, ny, 0)]
        T += [uv_canto(s_, h), uv_canto(s_, -h)]
    for i in range(len(c) - 1):
        ta, ba, tb, bb = base + 2 * i, base + 2 * i + 1, base + 2 * i + 2, base + 2 * i + 3
        F += [(ba, bb, tb), (ba, tb, ta)]
    plano = lambda filas, fmt: " ".join(fmt.format(*f) for f in filas)
    return {"vertex": plano(V, "{:.5f} {:.5f} {:.5f}"), "normal": plano(N, "{:.4f} {:.4f} {:.4f}"),
            "texcoord": plano(T, "{:.5f} {:.5f}"), "face": plano(F, "{} {} {}")}


def _ruido_suave(rng, alto, ancho, celdas):
    """Ruido suave (alto x ancho) en [-1, 1]: rejilla aleatoria interpolada."""
    import cv2
    base = rng.uniform(-1, 1, (celdas[0], celdas[1])).astype(np.float32)
    return cv2.resize(base, (ancho, alto), interpolation=cv2.INTER_CUBIC)


def _generar_texturas():
    """Crea (solo si faltan) las texturas de la mesa en CARPETA_TEXTURAS.
    Devuelve {nombre de fichero: ruta absoluta} de las que existen al final."""
    rutas = {}
    try:
        import cv2
        os.makedirs(CARPETA_TEXTURAS, exist_ok=True)
        rng = np.random.default_rng(7)                      # siempre la misma veta

        ruta = os.path.join(CARPETA_TEXTURAS, "madera_roble.png")
        if not os.path.isfile(ruta):                        # veta de roble, a lo largo de u (x)
            alto, ancho = 512, 2048
            v = np.linspace(0, 1, alto, dtype=np.float32)[:, None]
            u = np.linspace(0, 1, ancho, dtype=np.float32)[None, :]
            ondula = 0.035 * _ruido_suave(rng, alto, ancho, (6, 4)) + 0.010 * _ruido_suave(rng, alto, ancho, (24, 12))
            anillos = 0.5 + 0.5 * np.sin(2 * np.pi * (v + ondula) * 38 + 2.0 * np.sin(u * 3.1))
            anillos = anillos ** 3                                           # lineas finas y oscuras
            fibra = 0.5 + 0.5 * _ruido_suave(rng, alto, ancho, (alto // 2, 8))   # fibras alargadas
            poros = rng.random((alto, ancho)).astype(np.float32) > 0.985
            t = np.clip(0.55 * anillos + 0.30 * fibra + 0.25 * poros, 0, 1)
            claro, oscuro = np.array(COLORES_ROBLE[0]), np.array(COLORES_ROBLE[1])
            img = claro * (1 - t[..., None]) + oscuro * t[..., None]
            cv2.imwrite(ruta, (np.clip(img, 0, 1)[..., ::-1] * 255).astype(np.uint8))
        rutas["madera_roble.png"] = ruta

        # fieltro: la imagen cubre la almohadilla entera (una sola vez), con el pespunte dibujado
        # siguiendo su contorno redondeado. El nombre lleva las medidas: si cambian, se regenera.
        sx, sy, r = _medidas_tapete()
        ruta = os.path.join(CARPETA_TEXTURAS, f"fieltro_{round(2000 * sx)}x{round(2000 * sy)}"
                                              f"_r{round(1000 * r)}.png")
        if not os.path.isfile(ruta):
            ppm = 2000                                       # pixeles por metro
            ancho, alto = round(2 * sx * ppm), round(2 * sy * ppm)
            fondo = np.array(COLOR_FIELTRO)
            grano = cv2.GaussianBlur(rng.normal(0, 1, (alto, ancho)).astype(np.float32), (0, 0), 0.8)
            manchas = _ruido_suave(rng, alto, ancho, (12, 20))
            img = fondo * (1 + 0.07 * grano[..., None] + 0.035 * manchas[..., None])
            img = (np.clip(img, 0, 1)[..., ::-1] * 255).astype(np.uint8)        # BGR

            m = PUNTADA["margen"]
            camino = _contorno_redondeado(sx - m, sy - m, max(r - m, 0.004), n_arco=40)
            camino = np.vstack([camino, camino[:1]])
            tramos = np.linalg.norm(np.diff(camino, axis=0), axis=1)
            s_acum = np.r_[0.0, np.cumsum(tramos)]
            periodo = PUNTADA["largo"] + PUNTADA["hueco"]
            fino = np.linspace(0, s_acum[-1], int(s_acum[-1] / 0.0005))
            pts = np.c_[np.interp(fino, s_acum, camino[:, 0]), np.interp(fino, s_acum, camino[:, 1])]
            a_px = lambda q: (int(round((q[0] + sx) * ppm * 16)), int(round((sy - q[1]) * ppm * 16)))
            grosor = max(1, round(PUNTADA["ancho"] * ppm))
            hilo = tuple(int(255 * c) for c in COLOR_PUNTADA[::-1])
            sombra = tuple(int(255 * c * 0.35) for c in COLOR_FIELTRO[::-1])
            for desplaz, color in ((1.5, sombra), (0.0, hilo)):        # sombra del hilo y el hilo
                for k in range(len(fino) - 1):
                    if (fino[k] % periodo) < PUNTADA["largo"]:
                        a, b = a_px(pts[k]), a_px(pts[k + 1])
                        d = int(desplaz * 16)
                        cv2.line(img, (a[0] + d, a[1] + d), (b[0] + d, b[1] + d), color, grosor,
                                 cv2.LINE_AA, shift=4)
            cv2.imwrite(ruta, img)
        rutas["fieltro"] = ruta
    except Exception as e:                                  # sin cv2 o sin permiso de escritura
        print(f"[aviso] no se pudieron generar las texturas de la mesa ({e}): uso colores lisos")
    return rutas


def _anadir_mesa(root, geom):
    """Mesa de estilo nordico: tablero de roble con esquinas redondeadas, almohadilla de
    fieltro antracita con pespunte naranja Elecnor, faldon y patas inclinadas de nogal con contera de laton, y una balda
    de listones. La superficie de apoyo queda justo a MESA_ALTO."""
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    texturas = _generar_texturas()

    # El modelo del robot tiene un <default> que da a todos los geoms un rgba semitransparente,
    # y un rgba heredado TAPA el color del material (la mesa salia translucida y gris).
    # Por eso cada geom de la mesa lleva su rgba explicito: el del material, opaco, o blanco
    # si el material tiene textura (la textura se multiplica por ese rgba).
    color_geom = {n: ("1 1 1 1" if tex in texturas else rgba)
                  for n, (rgba, _, _, _, tex) in MATERIALES_MESA.items()}
    geom_escena = geom

    def geom(padre, **kw):
        if "material" in kw and "rgba" not in kw:
            kw["rgba"] = color_geom[kw["material"]]
        return geom_escena(padre, **kw)
    for nombre, (rgba, spec, brillo, refl, tex) in MATERIALES_MESA.items():
        atr = {"name": nombre, "rgba": rgba, "specular": str(spec),
               "shininess": str(brillo), "reflectance": str(refl)}
        if tex in texturas:
            ET.SubElement(asset, "texture", {"name": "tex_" + nombre, "type": "2d",
                                             "file": texturas[tex]})
            # las losas son mallas con sus propias coordenadas de textura (ver _malla_losa);
            # en las piezas sueltas (listones) la textura se repite por metro
            atr.update(texture="tex_" + nombre, rgba="1 1 1 1", texuniform="true",
                       texrepeat=f"{1 / LARGO_VETA:.4f} {4 / LARGO_VETA:.4f}")
        ET.SubElement(asset, "material", atr)

    mundo = root.find("worldbody")
    cx, cy = MESA_CENTRO
    mx, my = MESA_MEDIO

    def losa(nombre, z, sx, sy, semigrosor, r, material, uv_cara, uv_canto):
        """Losa de esquinas redondeadas: UNA malla (sin piezas solapadas que parpadeen)."""
        ET.SubElement(asset, "mesh", dict(name="malla_" + nombre,
                                          **_malla_losa(sx, sy, semigrosor, r, uv_cara, uv_canto)))
        geom(mundo, name=nombre, type="mesh", mesh="malla_" + nombre, pos=(cx, cy, z), material=material)

    # tablero: la veta corre a lo largo de x; en el canto, a lo largo del borde
    z_tapete = MESA_ALTO - GROSOR_TAPETE / 2
    z_tablero = MESA_ALTO - GROSOR_TAPETE - GROSOR_TABLERO / 2
    z_bajo = MESA_ALTO - GROSOR_TAPETE - GROSOR_TABLERO
    L = LARGO_VETA
    losa("mesa", z_tablero, mx, my, GROSOR_TABLERO / 2, RADIO_ESQUINA, "mesa_roble",
         uv_cara=lambda x, y: (0.5 + x / L, 0.5 + 4 * y / L),
         uv_canto=lambda s_, z: (s_ / L, 0.3 + 4 * z / L))
    # fieltro: la imagen (con el pespunte) cubre la almohadilla exactamente una vez
    tx, ty, tr = _medidas_tapete()
    losa("mesa_tapete", z_tapete, tx, ty, GROSOR_TAPETE / 2, tr, "mesa_tapete",
         uv_cara=lambda x, y: (0.5 + x / (2 * tx), 0.5 + y / (2 * ty)),
         uv_canto=lambda s_, z: (0.01, 0.01))

    # faldon de nogal, retranqueado bajo el tablero
    px, py = mx - RETRANQUEO_PATA, my - RETRANQUEO_PATA
    z_f, e = z_bajo - ALTO_FALDON / 2, 0.011
    for sy in (-1, 1):
        geom(mundo, type="box", pos=(cx, cy + sy * py, z_f), size=(px, e, ALTO_FALDON / 2), material="mesa_nogal")
    for sx in (-1, 1):
        geom(mundo, type="box", pos=(cx + sx * px, cy, z_f), size=(e, py, ALTO_FALDON / 2), material="mesa_nogal")
    # filete naranja Elecnor en el faldon que mira al robot (+y)
    geom(mundo, name="mesa_franja", type="box", pos=(cx, cy + py + e + 0.0008, z_f),
         size=(px - 0.05, 0.0008, 0.006), material="mesa_naranja")

    # patas inclinadas hacia fuera, con contera de laton
    ax, ay = ABERTURA_PATA
    patas = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            arriba = np.array([cx + sx * px, cy + sy * py, z_bajo])
            abajo = np.array([cx + sx * (px + ax), cy + sy * (py + ay), RADIO_PATA])
            patas[sx, sy] = (arriba, abajo)
            eje = _unit(arriba - abajo)
            geom(mundo, type="capsule", fromto=np.r_[abajo + eje * LARGO_CONTERA, arriba],
                 size=RADIO_PATA, material="mesa_nogal")
            geom(mundo, type="capsule", fromto=np.r_[abajo, abajo + eje * LARGO_CONTERA],
                 size=RADIO_PATA * 1.04, material="mesa_laton")

    def en_pata(sx, sy, z):
        arriba, abajo = patas[sx, sy]
        t = (z - abajo[2]) / (arriba[2] - abajo[2])
        return abajo + t * (arriba - abajo)

    # balda: dos travesanos (de delante a atras) y listones apoyados encima
    r_trav = 0.013
    for sx in (-1, 1):
        geom(mundo, type="capsule", fromto=np.r_[en_pata(sx, -1, Z_BALDA), en_pata(sx, 1, Z_BALDA)],
             size=r_trav, material="mesa_nogal")
    x_in = en_pata(1, 1, Z_BALDA)[0] - cx + r_trav             # semilargo de los listones
    y_in = en_pata(1, 1, Z_BALDA)[1] - cy - 0.02
    ancho = 2 * y_in / N_LISTONES
    for k in range(N_LISTONES):
        y = cy - y_in + ancho * (k + 0.5)
        geom(mundo, name=f"mesa_liston_{k}", type="box",
             pos=(cx, y, Z_BALDA + r_trav + 0.009), size=(x_in, ancho / 2 - 0.006, 0.009),
             material="mesa_roble")


def anadir_al_modelo(root):
    """Anade el escenario al arbol XML del robot (se llama al cargar el modelo)."""
    global z_techo, RED_EXTRA
    mundo = root.find("worldbody")
    # con el KUKA (montar_kuka.py lo monta antes que el escenario) el techo va alto
    hay_kuka = any(b.get("name") == "kuka_soporte" for b in root.iter("body"))
    z_techo = Z_TECHO_KUKA if hay_kuka else Z_TECHO
    base = {"contype": "0", "conaffinity": "0", "group": str(GRUPO_ESCENARIO)}

    def geom(padre, **kw):
        atr = dict(base)
        atr.update({k: (v if isinstance(v, str) else _f(np.atleast_1d(v))) for k, v in kw.items()})
        return ET.SubElement(padre, "geom", atr)

    # techo: un plano girado 180 grados (su cara visible mira hacia abajo, al robot).
    # MuJoCo solo dibuja los planos por su cara visible: desde arriba no tapa la camara.
    x0, x1, y0, y1 = TECHO_EXTENSION
    geom(mundo, name="techo", type="plane", pos=((x0 + x1) / 2, (y0 + y1) / 2, z_techo),
         quat="0 1 0 0", size=((x1 - x0) / 2, (y1 - y0) / 2, 0.1), rgba=TECHO)

    # red instalada, con varillas roscadas y abrazaderas naranjas
    RED_EXTRA = []
    if TUBERIAS_DETALLADAS:        # tubos huecos, codos, abrazaderas isofonicas... (tuberias_pvc.py)
        RED_EXTRA = tuberias_pvc.anadir_red(root, TUBOS_TECHO, CODOS_TECHO, z_techo, CONEXION_PUNTO,
                                            GRUPO_ESCENARIO, SEPARACION_SOPORTES)
    else:                          # cilindros lisos (como antes)
        for i, (a, b, r) in enumerate(TUBOS_TECHO):
            a, b = np.array(a), np.array(b)
            geom(mundo, name=f"tubo_techo_{i}", type="cylinder", fromto=np.r_[a, b], size=r, rgba=PVC)
            largo = np.linalg.norm(b - a)
            eje = (b - a) / largo
            n = max(1, int(largo // SEPARACION_SOPORTES))
            for k in range(n):
                c = a + eje * (largo * (k + 0.5) / n)
                geom(mundo, type="cylinder", fromto=np.r_[c + [0, 0, r], [c[0], c[1], z_techo]],
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

    # mesa (banco de trabajo a la derecha del robot)
    _anadir_mesa(root, geom)

    # piezas sueltas: un cuerpo mocap por pieza (se colocan a mano desde Python en cada ciclo).
    # Las dos estan siempre; la que no esta en juego se aparca bajo el suelo (EscenarioTuberias).
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    def cuerpo(pz):
        return ET.SubElement(mundo, "body", {"name": pz.cuerpo, "mocap": "true",
                                             "pos": _f(pz.pos0), "quat": _f(mat_a_quat(pz.rot0))})

    # la herramienta (tecla 1): el taladro de un .xml o una malla propia
    if "herramienta" in PIEZAS:
        pz = PIEZAS["herramienta"]
        pieza = cuerpo(pz)
        if PIEZA_XML is not None:      # pieza de un .xml (p.ej. el taladro): sus geoms visuales
            mallas, geoms_xml = PIEZA_XML
            for nombre, atr in mallas.items():
                ET.SubElement(asset, "mesh", dict(atr, name="pieza_" + nombre))
            for k, g in enumerate(geoms_xml):
                # el nombre debe empezar por el prefijo de la pieza (asi la logica la reconoce)
                kw = dict(name=f"{pz.prefijo}{k}", type=g["tipo"], rgba=g["rgba"] or COLOR_PIEZA, mass="0")
                if g["malla"]:
                    kw["mesh"] = "pieza_" + g["malla"]
                if g["size"]:
                    kw["size"] = g["size"]
                if g["fromto"] is not None:
                    kw["fromto"] = g["fromto"]
                else:
                    kw["pos"], kw["quat"] = g["P"], mat_a_quat(g["R"])
                geom(pieza, **kw)
        else:                          # modelo propio: una malla
            ET.SubElement(asset, "mesh", {"name": "pieza_propia",
                                          "file": os.path.abspath(RUTA_MALLA_PIEZA),
                                          "scale": _f([ESCALA_REAL] * 3)})
            # el nombre debe empezar por el prefijo de la pieza (asi se pinta de verde al encajar)
            geom(pieza, name=f"{pz.prefijo}malla", type="mesh", mesh="pieza_propia",
                 pos=MALLA_POS, rgba=COLOR_PIEZA)

    # el tubo para colocar (tecla 2): codo de PVC con un tramo y un manguito a cada lado
    pz = PIEZAS["tubo"]
    pieza = cuerpo(pz)
    if TUBERIAS_DETALLADAS:     # hueco, como la red del techo (tuberias_pvc.py)
        tuberias_pvc.pieza_codo(root, pieza, R_TUBO, R_COPA, LARGO_COPA, pz.extremos, GRUPO_ESCENARIO,
                                prefijo=pz.prefijo)
        return
    geom(pieza, name=f"{pz.prefijo}codo", type="sphere", pos=(0, 0, 0), size=R_COPA + 0.003, rgba=PVC_COPA)
    for k, (punto, eje) in enumerate(pz.extremos):
        largo = np.linalg.norm(punto)
        geom(pieza, name=f"{pz.prefijo}tramo{k}", type="cylinder",
             fromto=np.r_[[0, 0, 0], eje * (largo - LARGO_COPA)], size=R_TUBO, rgba=PVC)
        geom(pieza, name=f"{pz.prefijo}copa{k}", type="cylinder",
             fromto=np.r_[eje * (largo - LARGO_COPA), eje * largo], size=R_COPA, rgba=PVC_COPA)
        geom(pieza, name=f"{pz.prefijo}copacodo{k}", type="cylinder",
             fromto=np.r_[[0, 0, 0], eje * 0.035], size=R_COPA, rgba=PVC_COPA)


# =============================== LOGICA ======================================
class EscenarioTuberias:
    def __init__(self, robot):
        m, d = robot.m, robot.d
        self.m, self.d = m, d
        # cada pieza del modelo con su mocap, sus geoms y sus colores (para el destello)
        self.piezas = {}
        for clave, pz in PIEZAS.items():
            cuerpo = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, pz.cuerpo)
            if cuerpo < 0:
                raise RuntimeError(f"El modelo no tiene el escenario de tuberias cargado (falta '{pz.cuerpo}')")
            geoms = [g for g in range(m.ngeom)
                     if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(pz.prefijo)]
            self.piezas[clave] = SimpleNamespace(**vars(pz), id_mocap=int(m.body_mocapid[cuerpo]),
                                                 geoms=geoms, rgba=m.geom_rgba[geoms].copy())
        self.pz = self.piezas[PIEZA_INICIAL]      # la pieza en juego
        self.g_anillo = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "punto_conexion")
        self._informar_malla()
        self._comprobar_alcance(robot)
        self._comprobar_kuka()
        # el KUKA no puede subir al robot mas alla del techo (solo con el escenario a la vista)
        self.kuka = getattr(robot, "kuka", None)
        self.activa = False             # oculto y sin efecto hasta que se active
        self.reiniciar()

    @property
    def pieza(self):
        """Clave de la pieza en juego: "herramienta" o "tubo"."""
        return self.pz.clave

    def _comprobar_alcance(self, robot, lado="R"):
        """Imprime a que distancia queda el punto de agarre de cada pieza del hombro del
        brazo que la va a coger, comparada con el alcance hombro-muneca de ese brazo.
        Es orientativo: la mano anade unos centimetros mas de alcance."""
        try:
            brazo = robot.brazos[lado]
            hombro = np.asarray(brazo.hombro, dtype=float)
            alcance = float(brazo.L_brazo + brazo.L_antebrazo)
        except (AttributeError, KeyError, TypeError):
            return
        nombre = {"L": "izquierdo", "R": "derecho"}.get(lado, lado)
        for pz in self.piezas.values():
            if pz.segmento is not None:
                objetivo = pz.pos0 + pz.rot0 @ ((pz.segmento[0] + pz.segmento[1]) / 2)
            else:
                objetivo = pz.pos0.copy()
            dist = float(np.linalg.norm(objetivo - hombro))
            r = dist / max(alcance, 1e-9)
            print(f"Mesa en ({MESA_CENTRO[0]:+.2f}, {MESA_CENTRO[1]:+.2f}) m | {pz.nombre} (tecla {pz.tecla}) a "
                  f"{dist * 100:.0f} cm del hombro {nombre} (alcance hombro-muneca {alcance * 100:.0f} cm, "
                  f"{r * 100:.0f} %)")
            if r > 0.92:
                print(f"[aviso] {pz.nombre} queda casi al limite del brazo {nombre}: acerca la mesa "
                      f"(sube MESA_CENTRO[1] hacia 0) o sube PIEZA_EN_MESA[1]")
        # techo: hasta donde sube la muneca con el brazo hacia arriba (el seguimiento comprime
        # el brazo a partir del 92 % de su alcance), mas la mano
        muneca_max = float(hombro[2]) + 0.92 * alcance
        if getattr(robot, "kuka", None) is not None:
            print(f"Techo a {z_techo:.2f} m (soporte KUKA): para llegar, sube al robot con + "
                  f"(ahora la muneca {nombre} llega a ~{muneca_max:.2f} m)")
            return
        print(f"Techo a {z_techo:.2f} m | la muneca {nombre} sube hasta ~{muneca_max:.2f} m "
              f"(hombro a {hombro[2]:.2f} m)")
        if muneca_max < z_techo - 0.25:
            print(f"[aviso] el techo queda alto para taladrarlo: con la mano y el taladro "
                  f"(~25 cm) no se llega. Baja Z_TECHO a ~{muneca_max + 0.15:.2f} m")

    def _comprobar_kuka(self, prefijo="kuka_"):
        """Avisa si alguna tuberia del techo pasa a menos de HOLGURA_KUKA del KUKA (con los
        vertices reales de sus mallas). Si no hay KUKA en el modelo, no hace nada."""
        m = self.m
        geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                 and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "")
                 .startswith(prefijo)]
        if not geoms:
            return
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        puntos = []
        for g in geoms:
            mid = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            puntos.append(v[::4] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        P = np.vstack(puntos)
        problemas = []
        for i, (a, b, r) in enumerate(list(TUBOS_TECHO) + list(RED_EXTRA)):   # + subidas al techo
            a, b = np.array(a, float), np.array(b, float)
            ab = b - a
            t = np.clip((P - a) @ ab / (ab @ ab), 0.0, 1.0)
            dist = float(np.min(np.linalg.norm(P - (a + t[:, None] * ab), axis=1))) - r
            if dist < HOLGURA_KUKA:
                problemas.append(f"tuberia {i} a {dist * 100:.0f} cm")
        techo_z = float(P[:, 2].max())
        if techo_z > z_techo - 0.01:
            problemas.append(f"el KUKA atraviesa el techo (llega a z = {techo_z:.2f} m)")
        if problemas:
            print(f"[aviso] el KUKA choca con el escenario: {', '.join(problemas)}")
        else:
            print(f"Red del techo libre del KUKA (holgura >= {HOLGURA_KUKA * 100:.0f} cm; "
                  f"el KUKA llega hasta z = {techo_z:.2f} m)")

    def _informar_malla(self):
        """Con un modelo propio, imprime la caja que ocupa tal y como la ha cargado
        MuJoCo, en las coordenadas de la pieza (comprobacion del centrado y la escala)."""
        pz = self.piezas.get("herramienta")
        if pz is None:
            return
        g = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"{pz.prefijo}malla")
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

    # ---------------------------------------------------------------- interruptores
    def _clave(self, pieza):
        """Clave valida: sin herramienta configurada (RUTA_XML_PIEZA y RUTA_MALLA_PIEZA a None),
        las dos teclas usan el tubo."""
        if pieza is None:
            return self.pz.clave
        return pieza if pieza in self.piezas else "tubo"

    def alternar(self, pieza, visor=None):
        """Interruptor (teclas 1 y 2 del teleop). Si el escenario esta abierto con esa pieza,
        lo quita; si no, lo abre con ella (o cambia a ella). Devuelve el mensaje para el HUD."""
        clave = self._clave(pieza)
        if self.activa and self.pz.clave == clave:
            self.desactivar(visor)
            return "Escenario de tuberias: quitado"
        cambio = self.activa
        self.activar(visor, clave)
        return f"Escenario de tuberias {'-> ' if cambio else ''}con {self.pz.nombre.lower()}"

    def activar(self, visor=None, pieza=None):
        """Muestra el escenario con la pieza dada ("herramienta" o "tubo"; None = la de antes)
        y lo deja como al principio."""
        self.pz = self.piezas[self._clave(pieza)]
        self.activa = True
        self.reiniciar()
        if self.kuka is not None and getattr(self.kuka, "disponible", False):
            self.kuka.techo = z_techo
            if self.kuka.z_max() > z_techo:
                print(f"[aviso] el KUKA ya esta por encima del techo ({z_techo:.2f} m): bajalo con -")
        if visor is not None:
            with visor.lock():
                visor.opt.geomgroup[GRUPO_ESCENARIO] = 1

    def desactivar(self, visor=None):
        """Quita el escenario: lo oculta, suelta la pieza (sin efecto en las manos) y el KUKA
        deja de tener techo."""
        self.activa = False
        self.reiniciar()
        if self.kuka is not None and getattr(self.kuka, "disponible", False):
            self.kuka.techo = None
        if visor is not None:
            with visor.lock():
                visor.opt.geomgroup[GRUPO_ESCENARIO] = 0

    # ---------------------------------------------------------------- estado
    def reiniciar(self):
        """La pieza en juego vuelve a la mesa; las demas, al aparcamiento (bajo el suelo)."""
        for pz in self.piezas.values():
            self.m.geom_rgba[pz.geoms] = pz.rgba
            if pz is not self.pz:
                self.d.mocap_pos[pz.id_mocap] = APARCAMIENTO
                self.d.mocap_quat[pz.id_mocap] = (1.0, 0.0, 0.0, 0.0)
        self.p, self.R = self.pz.pos0.copy(), self.pz.rot0.copy()
        self.estado = "mesa"            # mesa | sujeta | cayendo | suelo | instalada
        self.sujeta = None              # lado del robot que la lleva
        self.rel_p = self.rel_R = None
        self.vz = 0.0
        self.t_suelo = 0.0
        self.armado = {"L": True, "R": True}
        self.dist = self.ang = None
        self.t_destello = 0.0
        self._pintar_anillo(ANILLO_LEJOS)
        self._escribir()

    def _volver_a_la_mesa(self):
        """Reaparece en su sitio de la mesa, sin tocar lo ya instalado."""
        self.p, self.R = self.pz.pos0.copy(), self.pz.rot0.copy()
        self.estado, self.sujeta, self.vz = "mesa", None, 0.0
        self._escribir()

    def apertura_minima(self, lado):
        """La pinza que sujeta la pieza no se cierra mas alla del grosor del tubo."""
        return APERTURA_TUBO if self.activa and self.sujeta == lado else 0.0

    # ---------------------------------------------------------------- geometria
    def _segmentos(self):
        if self.pz.segmento is not None:
            a, b = self.pz.segmento
            return [(self.p + self.R @ a, self.p + self.R @ b)]
        return [(self.p, self.p + self.R @ punto) for punto, _ in self.pz.extremos]

    def _mas_cercano(self, q):
        mejores = [_punto_segmento(q, a, b) for a, b in self._segmentos()]
        c = min(mejores, key=lambda c: np.linalg.norm(q - c))
        return c, float(np.linalg.norm(q - c))

    def _mejor_extremo(self):
        """Extremo de la pieza mas cercano a encajar: (indice, distancia, angulo)."""
        mejor = None
        for k, (punto, eje) in enumerate(self.pz.extremos):
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
        punto, eje = self.pz.extremos[k]
        self.R = rot_entre(self.R @ eje, -CONEXION_EJE) @ self.R
        destino = CONEXION_PUNTO - CONEXION_EJE * SOLAPE
        self.p = destino - self.R @ punto

    def _altura_apoyo(self):
        """Altura de la mesa si la pieza esta encima de ella; si no, el suelo."""
        if self.pz.esquinas is not None:
            c = self.p + self.R @ self.pz.esquinas.mean(axis=0)
        else:
            c = self.p + self.R @ (sum(p for p, _ in self.pz.extremos) / (len(self.pz.extremos) + 1))
        dentro = (abs(c[0] - MESA_CENTRO[0]) < MESA_MEDIO[0] and abs(c[1] - MESA_CENTRO[1]) < MESA_MEDIO[1])
        return MESA_ALTO if dentro else 0.0

    def _z_minima(self):
        if self.pz.esquinas is not None:   # su caja, girada como este la pieza
            return float(min((self.p + self.R @ c)[2] for c in self.pz.esquinas))
        puntos = [self.p] + [self.p + self.R @ p for p, _ in self.pz.extremos]
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
                    self.m.geom_rgba[self.pz.geoms] = self.pz.rgba
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

        if self.estado == "sujeta" and self.pz.encaja:
            k, self.dist, self.ang = self._mejor_extremo()
            if self.dist < TOL_POS and self.ang < TOL_ANG:
                self._encajar(k)
                self.estado, self.sujeta = "instalada", None
                self.m.geom_rgba[self.pz.geoms] = DESTELLO
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
        self.d.mocap_pos[self.pz.id_mocap] = self.p
        self.d.mocap_quat[self.pz.id_mocap] = mat_a_quat(self.R)

    # ---------------------------------------------------------------- texto
    def texto(self):
        if not self.activa:
            return ""
        nombres = {"L": "IZQ", "R": "DER"}
        nombre, tecla = self.pz.nombre, self.pz.tecla
        if self.estado == "instalada":
            return f"{nombre}: INSTALADO  ({tecla} dos veces = empezar de nuevo)"
        if self.estado == "sujeta":
            s = f"{nombre}: en la mano {nombres[self.sujeta]}"
            if self.dist is not None:
                s += f" | conexion a {self.dist * 100:4.1f} cm, {self.ang:3.0f} deg"
                s += f" (encaja < {TOL_POS * 100:.0f} cm y < {TOL_ANG:.0f} deg)"
            return s
        if self.estado == "cayendo":
            return f"{nombre}: cayendo"
        if self.estado == "suelo":
            return f"{nombre}: se ha caido al suelo, vuelve a la mesa..."
        return f"{nombre}: en la mesa  (cierra la mano junto a el para cogerlo | {tecla} = quitar)"
