"""
OrcaHand v2 en las munecas del TRON 2 (modo cinematico, igual que el resto del TRON 2).

Que hay aqui:
  quitar_pinzas(root)     -> borra del XML las pinzas originales del TRON 2 (y todo lo que las nombra)
  montar_manos(ruta_xml)  -> carga el TRON 2 y le "atornilla" una OrcaHand en cada muneca (MjSpec)
  ManoOrca                -> usa el MISMO retargeting del proyecto OrcaHand (retarget_a_ctrl) y
                             escribe el resultado directamente en qpos (sin fisica, como los brazos)

Archivos a copiar del proyecto OrcaHand a la carpeta del TRON 2:
  config.py, biomecanica.py, retargeting.py y skill_shield/comun/suavizador.py
  + la carpeta orcahand_description-main/v2/models (mjcf y assets)
En retargeting.py cambiar las dos importaciones de skill_shield por:
  from suavizador import SuavizadorND
  from biomecanica import calcular_abertura, calcular_distancia, calcular_flexion, calcular_flexion_muneca

Prueba rapida del montaje (sin camara, las manos abren y cierran solas):
  python mano_orca.py tron2a/DACH_TRON2A/xml/robot_elecnor.xml
"""
import os
import shutil
import sys
import time
import types
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from config import LIMITES_CTRL, N_ACT, OFFSET_MANO, POSICION_STANDBY, TALADRO_DEMO_MANO_AGARRE
from retargeting import retarget_a_ctrl
from suavizador import SuavizadorND

# ============================== AJUSTES ======================================
ORCA_MJCF = "orcahand_description-main/v2/models/mjcf"
NOMBRE = {"L": "left", "R": "right"}           # orcahand_<nombre>_body.xml (+ orcahand_<nombre>.mjcf)
PREFIJO = {"L": "orcaL_", "R": "orcaR_"}      # sin "_L_"/"_R_" para no confundir a la IK del brazo
CUERPO_MUNECA = "wrist_roll_{lado}_Link"      # donde se atornilla la mano

# Posicion y giro de la mano respecto a la muneca: pos en metros (ejes del cuerpo de la
# muneca), quat (w, x, y, z).
#   quat=None -> se calcula al arrancar: la torre de la Orca queda en la misma linea que el
#                antebrazo del TRON 2 y la palma mira hacia ORIENTACION_MANO["palma"].
#   pos=None  -> se calcula al arrancar: la base de la torre queda centrada al final de la muneca.
# Al arrancar se imprimen los valores calculados y lo bien que encaja; puedes copiarlos
# aqui para fijarlos y retocarlos a mano (p.ej. pos para meter o sacar la mano unos mm).
MONTAJE = {"L": dict(pos=None, quat=None),
           "R": dict(pos=None, quat=None)}

# Orientacion deseada de la mano con el brazo en reposo, en ejes del mundo del robot
# (x hacia delante del robot, y a su izquierda, z arriba):
#   dedos="antebrazo" -> la torre de la Orca sigue la linea del antebrazo del TRON 2
#                        (la mano parece parte del brazo)
#   dedos=[0, 0, -1]  -> los dedos apuntan exactamente hacia abajo en el mundo
# La palma mira lo mas parecido posible a 'palma' sin dejar de cumplir lo de los dedos.
ORIENTACION_MANO = dict(dedos="antebrazo",
                        palma=[-1.0, 0.0, 0.0])   # palma hacia atras

VEL_DEDOS = 8.0          # rad/s maximos de cada junta (quita tirones entre fotogramas)
# Si en la mano IZQUIERDA alguna junta se mueve al reves (suele pasar con las abducciones),
# pon -1.0 en su indice (mismo orden que LIMITES_CTRL de config.py)
SIGNO_IZQ = np.ones(N_ACT)
# Juntas de flexion (las que cierran la mano): mcp/pip de los 4 dedos + mcp/ip del pulgar
DEDOS_FLEXION = [2, 3, 5, 6, 8, 9, 11, 12, 15, 16]
TOPE_TUBO = 1.0          # rad: con el tubo cogido los dedos no cierran mas (no lo atraviesan)

# Posturas para el modo sencillo (sin landmarks de los dedos): se mezclan con la apertura
MANO_ABIERTA = np.array(POSICION_STANDBY, dtype=float)
MANO_CERRADA = MANO_ABIERTA.copy()
for _i, _v in TALADRO_DEMO_MANO_AGARRE.items():   # agarre ya calibrado en el proyecto OrcaHand
    MANO_CERRADA[_i] = _v

SECCIONES_CON_REFERENCIAS = ("equality", "actuator", "contact", "sensor", "tendon")


# ============================== MODELO =======================================
def quitar_pinzas(root):
    """Borra los cuerpos grasper_* del TRON 2 y todo lo que los nombra (equality,
    actuadores, exclusiones de contacto, sensores, tendones). Los keyframes se quitan
    porque su qpos ya no tendria el tamano correcto."""
    nombres = set()
    objetivos = [(p, h) for p in root.iter() for h in p
                 if h.tag == "body" and h.get("name", "").startswith("grasper_")]
    for padre, hijo in objetivos:
        for e in hijo.iter():
            if e.get("name"):
                nombres.add(e.get("name"))
        if hijo in list(padre):
            padre.remove(hijo)
    for seccion in [s for tag in SECCIONES_CON_REFERENCIAS for s in root.iter(tag)]:
        for padre in list(seccion.iter()):
            for hijo in list(padre):
                if any(v in nombres for v in hijo.attrib.values()):
                    padre.remove(hijo)
    for tendones in root.iter("tendon"):          # tendones que se han quedado vacios
        for t in list(tendones):
            if len(t) == 0:
                tendones.remove(t)
    for kf in root.findall("keyframe"):
        root.remove(kf)
    print(f"Pinzas originales quitadas ({len(nombres)} elementos)")


def _rutas_absolutas(mano, ruta_archivo):
    """Las mallas de la mano se buscan relativas a SU archivo; al pegarla al TRON 2
    se buscarian en la carpeta del TRON 2. Se pasan a rutas absolutas.
    Ojo: en la OrcaHand v2 el meshdir ("models/assets/...") es relativo a la carpeta v2
    (desde donde la cargan scene_left.xml / scene_right.xml), no a la carpeta mjcf.
    Por eso se prueba la carpeta del archivo y, si no esta ahi, las de encima."""
    carpeta = os.path.dirname(os.path.abspath(ruta_archivo))
    bases = [carpeta]
    for _ in range(4):
        bases.append(os.path.dirname(bases[-1]))
    for lista, campo in (("meshes", "meshdir"), ("textures", "texturedir")):
        sub = getattr(mano, campo, "") or ""
        for a in getattr(mano, lista, []):
            if not a.file or os.path.isabs(a.file):
                continue
            candidatos = [os.path.normpath(os.path.join(b, sub, a.file)) for b in bases]
            encontrado = next((c for c in candidatos if os.path.exists(c)), None)
            if encontrado is None:
                raise FileNotFoundError(f"No encuentro '{a.file}'. Probado en:\n   " + "\n   ".join(candidatos))
            a.file = encontrado
        if hasattr(mano, campo):
            setattr(mano, campo, "")


def _como_xml(ruta):
    """Las versiones nuevas de MuJoCo eligen como leer un archivo por su EXTENSION y no
    reconocen '.mjcf' ("Could not find decoder"). Se usa una copia con extension .xml
    en la MISMA carpeta, para que las rutas relativas (mallas, includes) sigan valiendo."""
    if ruta.lower().endswith(".xml"):
        return ruta
    copia = os.path.splitext(ruta)[0] + ".xml"
    if not os.path.exists(copia) or os.path.getmtime(copia) < os.path.getmtime(ruta):
        shutil.copyfile(ruta, copia)
    return copia


# Colores de los materiales que usan los _body.xml y que no vienen definidos en el .mjcf
# (en el proyecto OrcaHand estaban en arctos_mjcf.xml). Si falta otro, sale gris.
COLORES_MATERIAL = {
    "naranja_personalizado": "0.94 0.49 0.15 1",   # naranja Elecnor
    "azul_personalizado":    "0.00 0.23 0.56 1",   # azul Elecnor
    "white":                 "0.90 0.90 0.90 1",
    "black":                 "0.10 0.10 0.10 1",
}
COLOR_POR_DEFECTO = "0.60 0.60 0.62 1"


def _completar_assets(raiz, carpeta, lado):
    """Anade al <asset> las mallas y materiales que usan los cuerpos pero no estan
    definidos. Las mallas 'right_T-DP_Skin' se buscan como 'T-DP_Skin.stl' en
    v2/models/assets/right/ (o en cualquier subcarpeta de assets)."""
    asset = raiz.find("asset")
    if asset is None:
        asset = ET.SubElement(raiz, "asset")
    definidas = {(e.tag, e.get("name")) for e in asset}
    usadas = {("mesh", e.get("mesh")) for e in raiz.iter() if e.get("mesh")} | \
             {("material", e.get("material")) for e in raiz.iter() if e.get("material")}
    faltan = sorted(u for u in usadas if u not in definidas)
    if not faltan:
        return

    # escala de las mallas: la misma que usan las del .mjcf (suelen estar en mm -> 0.001)
    escalas = [e.get("scale") for e in asset if e.tag == "mesh" and e.get("scale")]
    escala = max(set(escalas), key=escalas.count) if escalas else None

    # todos los .stl que hay bajo la carpeta assets (y la de mjcf), por nombre
    stls = {}
    for base in (os.path.join(carpeta, "..", "assets"), carpeta):
        for dirpath, _, ficheros in os.walk(os.path.normpath(base)):
            for f in ficheros:
                if f.lower().endswith(".stl"):
                    stls.setdefault(f.lower(), []).append(os.path.join(dirpath, f))
    sub = NOMBRE[lado]

    creadas, sin_malla = [], []
    for tipo, nombre in faltan:
        if tipo == "material":
            ET.SubElement(asset, "material", name=nombre,
                          rgba=COLORES_MATERIAL.get(nombre, COLOR_POR_DEFECTO))
            creadas.append(f"material {nombre}")
            continue
        corto = nombre[len(sub) + 1:] if nombre.startswith(sub + "_") else nombre
        candidatos = stls.get(corto.lower() + ".stl", []) + stls.get(nombre.lower() + ".stl", [])
        # preferir la de la carpeta de este lado (right/ o left/)
        candidatos.sort(key=lambda r: os.sep + sub + os.sep not in r)
        if not candidatos:
            sin_malla.append(nombre)
            continue
        atr = {"name": nombre, "file": os.path.abspath(candidatos[0])}
        if escala:
            atr["scale"] = escala
        ET.SubElement(asset, "mesh", atr)
        creadas.append(f"malla {nombre}")
    print(f"   anadidos {len(creadas)} assets que faltaban (escala de mallas: {escala or '1'})")
    if sin_malla:
        raise FileNotFoundError(f"Mano {lado}: no encuentro el .stl de estas mallas: {sin_malla}")


def _archivo_mano(lado, carpeta):
    """Devuelve un .xml que MuJoCo pueda cargar con la mano de ese lado.
    Los cuerpos salen SIEMPRE de orcahand_<lado>_body.xml. Si ese archivo es solo un
    fragmento para <include> (sin mallas ni actuadores), se completa con todo lo que no
    son cuerpos (compiler, default, asset, actuator, equality...) de orcahand_<lado>.mjcf
    y se guarda como orcahand_<lado>_tron2.xml en la MISMA carpeta (asi las rutas valen)."""
    n = NOMBRE[lado]
    ruta_body = os.path.join(carpeta, f"orcahand_{n}_body.xml")
    raiz_b = ET.parse(ruta_body).getroot()
    if raiz_b.tag == "mujoco" and raiz_b.find("worldbody") is not None \
            and raiz_b.find("actuator") is not None and raiz_b.find("asset") is not None:
        print(f"Mano {lado}: {os.path.basename(ruta_body)} es completo, se usa tal cual")
        return ruta_body

    # cuerpos del fragmento: <mujoco><body/></mujoco>, <mujoco><worldbody>...</worldbody></mujoco> o <body> suelto
    if raiz_b.tag == "body":
        cuerpos = [raiz_b]
    elif raiz_b.find("worldbody") is not None:
        cuerpos = list(raiz_b.find("worldbody"))
    else:
        cuerpos = raiz_b.findall("body")
    if not cuerpos:
        raise RuntimeError(f"No encuentro ningun <body> en {ruta_body}")

    ruta_mjcf = os.path.join(carpeta, f"orcahand_{n}.mjcf")
    arbol = ET.parse(ruta_mjcf)                 # ElementTree lee cualquier extension
    raiz = arbol.getroot()
    mundo = raiz.find("worldbody")
    if mundo is None:
        mundo = ET.SubElement(raiz, "worldbody")
    for hijo in list(mundo):                    # fuera los cuerpos del .mjcf (y suelo/luces)
        mundo.remove(hijo)
    for c in cuerpos:
        mundo.append(c)
    # lo que el fragmento traiga ademas de cuerpos (p.ej. mallas del logo) tambien se anade
    if raiz_b.tag == "mujoco":
        for hijo in raiz_b:
            if hijo.tag not in ("asset", "default", "actuator", "equality", "tendon", "contact"):
                continue
            destino = raiz.find(hijo.tag)
            if destino is None:
                raiz.append(hijo)
                continue
            nombres = {e.get("name") for e in destino if e.get("name")}
            for e in hijo:
                if not e.get("name") or e.get("name") not in nombres:
                    destino.append(e)
    salida = os.path.join(carpeta, f"orcahand_{n}_tron2.xml")
    _completar_assets(raiz, carpeta, lado)
    arbol.write(salida)
    print(f"Mano {lado}: cuerpos de {os.path.basename(ruta_body)} + resto de "
          f"{os.path.basename(ruta_mjcf)} -> {os.path.basename(salida)}")
    return salida


def _juntas_mano(m, lado):
    pref = PREFIJO[lado]
    acts = [a for a in range(m.nu)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").startswith(pref)]
    if len(acts) != N_ACT:
        raise RuntimeError(f"Mano {lado}: esperaba {N_ACT} actuadores con prefijo {pref}, hay {len(acts)}")
    return [int(m.actuator_trnid[a, 0]) for a in acts]


def _puntos_geoms(m, d, geoms, max_puntos=4000):
    """Vertices (mundo) de las mallas de esos geoms; los geoms que no son malla aportan su centro."""
    trozos = []
    for g in geoms:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_dataid[g] >= 0:
            k = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[k]: m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            trozos.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
        else:
            trozos.append(d.geom_xpos[g][None, :])
    P = np.vstack(trozos)
    if len(P) > max_puntos:
        P = P[:: len(P) // max_puntos + 1]
    return P


def _geoms_de(m, cuerpos):
    return [g for b in cuerpos for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])]


def _marco_montaje(m, d, lado, montaje):
    """Orientacion (mundo) y origen del marco donde se engancha la mano."""
    b_m = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CUERPO_MUNECA.format(lado=lado))
    R_w = d.xmat[b_m].reshape(3, 3)
    R_q = np.zeros(9)
    mujoco.mju_quat2Mat(R_q, np.asarray(montaje[lado]["quat"], float))
    return R_w @ R_q.reshape(3, 3), d.xpos[b_m] + R_w @ np.asarray(montaje[lado]["pos"], float)


def _torre(m, d, lado, montaje):
    """Eje y base de la TORRE (antebrazo) de la Orca.
    En sus archivos la OrcaHand esta de pie: la torre va a lo largo de uno de los ejes
    de su propio sistema de coordenadas. Se toma el eje (+-x, +-y, +-z del marco de
    montaje) que apunta hacia la palma; asi el eje es EXACTO aunque la malla de la torre
    sea ancha o tenga salientes (la pantalla del logo), que es lo que fallaba antes.
    Base = centro de la caja de la torre en la cara que va pegada al brazo."""
    juntas = _juntas_mano(m, lado)
    palma = int(m.jnt_bodyid[juntas[0]])                # cuerpo que mueve la muneca de la Orca
    pref = PREFIJO[lado]
    nombre_b = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    cadena, b = [], int(m.body_parentid[palma])          # cuerpos entre el montaje y la palma
    while b > 0 and nombre_b(b).startswith(pref):
        cadena.append(b)
        b = int(m.body_parentid[b])
    geoms = _geoms_de(m, cadena)
    if not geoms:
        raise RuntimeError(f"Mano {lado}: no encuentro la malla de la torre")

    R_f, o_f = _marco_montaje(m, d, lado, montaje)
    hacia_palma = d.xpos[palma] - o_f
    k, signo = max(((k, s) for k in range(3) for s in (1.0, -1.0)),
                   key=lambda ks: ks[1] * float(np.dot(R_f[:, ks[0]], hacia_palma)))
    eje = signo * R_f[:, k]

    L = (_puntos_geoms(m, d, geoms) - o_f) @ R_f         # vertices en ejes del marco
    base_local = 0.5 * (L.min(axis=0) + L.max(axis=0))   # centro de la caja de la torre
    base_local[k] = L[:, k].min() if signo > 0 else L[:, k].max()   # cara del lado del brazo
    inclinacion = np.degrees(np.arccos(np.clip(np.dot(eje, hacia_palma / np.linalg.norm(hacia_palma)), -1, 1)))
    return eje, o_f + R_f @ base_local, inclinacion


def _antebrazo(m, d, lado):
    """Eje del antebrazo del TRON 2 (del codo hacia la mano) y punto donde acaba su muneca.
    Eje = el de la junta de muneca que va a lo largo del antebrazo (sigue exactamente la pieza).
    Final = el punto mas alejado de las mallas de la muneca, sobre ese eje."""
    nombre = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
    marca = f"_{lado}_"
    b_muneca = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CUERPO_MUNECA.format(lado=lado))
    codos = [j for j in range(m.njnt) if marca in nombre(j) and "elbow" in nombre(j).lower()]
    if not codos:
        raise RuntimeError(f"No encuentro la junta del codo del brazo {lado}")
    v = d.xpos[b_muneca] - d.xanchor[codos[0]]
    v /= np.linalg.norm(v)
    eje, punto = v, d.xanchor[codos[0]].copy()
    munecas = [j for j in range(m.njnt) if marca in nombre(j) and "wrist" in nombre(j).lower()
               and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    if munecas:
        j = max(munecas, key=lambda j: abs(np.dot(d.xaxis[j], v)))
        e = d.xaxis[j] / np.linalg.norm(d.xaxis[j])
        if abs(np.dot(e, v)) > 0.95:        # solo si de verdad va a lo largo del antebrazo (< 18 grados)
            eje, punto = (e if np.dot(e, v) > 0 else -e), d.xanchor[j].copy()
    # final de la muneca: solo las mallas del propio TRON 2 (no las de la mano montada)
    b, geoms = b_muneca, []
    while b > 0 and not geoms:
        geoms = _geoms_de(m, [b])
        b = int(m.body_parentid[b])
    final = punto + eje * float(((_puntos_geoms(m, d, geoms) - punto) @ eje).max()) if geoms else d.xpos[b_muneca].copy()
    return eje, final


def _palma(m, d, lado):
    """Hacia donde mira la palma: hacia donde se mueven las yemas al cerrar los 4 dedos."""
    juntas = _juntas_mano(m, lado)
    qadr = [int(m.jnt_qposadr[j]) for j in juntas]
    geoms = _geoms_de(m, [int(m.jnt_bodyid[juntas[i]]) for i in (3, 6, 9, 12)])
    guardado = d.qpos[qadr].copy()

    def yemas():
        mujoco.mj_kinematics(m, d)
        return d.geom_xpos[geoms].mean(axis=0)

    d.qpos[qadr] = 0.0
    abierta = yemas()
    for i in (2, 3, 5, 6, 8, 9, 11, 12):
        d.qpos[qadr[i]] = 1.2
    cierre = yemas() - abierta
    d.qpos[qadr] = guardado
    mujoco.mj_kinematics(m, d)
    return cierre / np.linalg.norm(cierre)


def _base(a, b):
    """Base ortonormal [a, b', a x b'] con b' = b sin su componente sobre a."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) - np.dot(b, a) * a
    b /= np.linalg.norm(b)
    return np.column_stack([a, b, np.cross(a, b)])


def _construir(ruta_xml, carpeta_orca, montaje):
    spec = mujoco.MjSpec.from_file(ruta_xml)
    for lado in ("L", "R"):
        ruta_mano = _como_xml(_archivo_mano(lado, carpeta_orca))
        mano = mujoco.MjSpec.from_file(ruta_mano)
        _rutas_absolutas(mano, ruta_mano)
        muneca = spec.body(CUERPO_MUNECA.format(lado=lado))
        if muneca is None:
            raise RuntimeError(f"No encuentro el cuerpo {CUERPO_MUNECA.format(lado=lado)}")
        marco = muneca.add_frame(pos=montaje[lado]["pos"], quat=montaje[lado]["quat"])
        if hasattr(spec, "attach"):                       # MuJoCo >= 3.3
            spec.attach(mano, prefix=PREFIJO[lado], frame=marco)
        else:                                             # MuJoCo 3.2
            marco.attach_body(mano.worldbody.first_body(), PREFIJO[lado], "")
    return spec.compile()


def _comprobar(m, lado, montaje):
    """Imprime lo bien que encaja la torre de la Orca con el antebrazo del TRON 2."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    e_b, final = _antebrazo(m, d, lado)
    e_t, base, _ = _torre(m, d, lado, montaje)
    # control independiente: recta codo -> muneca del TRON 2 frente al eje usado
    nombre = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
    codo = next(j for j in range(m.njnt) if f"_{lado}_" in nombre(j) and "elbow" in nombre(j).lower())
    b_m = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CUERPO_MUNECA.format(lado=lado))
    v = d.xpos[b_m] - d.xanchor[codo]
    print(f"Mano {lado}: eje del antebrazo usado vs recta codo->muneca: "
          f"{np.degrees(np.arccos(np.clip(np.dot(e_b, v / np.linalg.norm(v)), -1, 1))):4.1f} grados")
    ang = np.degrees(np.arccos(np.clip(np.dot(e_b, e_t), -1, 1)))
    lateral = (base - final) - np.dot(base - final, e_b) * e_b
    print(f"Mano {lado}: torre vs antebrazo -> desalineacion {ang:4.1f} grados | "
          f"descentrado {np.linalg.norm(lateral) * 1000:4.0f} mm | "
          f"hueco/solape {np.dot(base - final, e_b) * 1000:+5.0f} mm")


def montar_manos(ruta_xml, carpeta_orca=ORCA_MJCF):
    """Carga el XML del TRON 2 y le engancha una OrcaHand en cada muneca.
    quat=None en MONTAJE: la torre de la Orca se alinea con el antebrazo del TRON 2 y la
    palma mira hacia ORIENTACION_MANO["palma"] (con los brazos en reposo).
    pos=None en MONTAJE: la base de la torre se coloca centrada al final de la muneca."""
    montaje = {l: dict(pos=list(MONTAJE[l]["pos"] or [0.0, 0.0, 0.0]),
                       quat=list(MONTAJE[l]["quat"] or [1.0, 0.0, 0.0, 0.0])) for l in ("L", "R")}
    m = _construir(ruta_xml, carpeta_orca, montaje)
    lados_giro = [l for l in ("L", "R") if MONTAJE[l]["quat"] is None]
    lados_pos = [l for l in ("L", "R") if MONTAJE[l]["pos"] is None]

    # 1) giro: llevar el eje de la torre al eje del antebrazo y la palma hacia atras
    if lados_giro:
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        for lado in lados_giro:
            b_m = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CUERPO_MUNECA.format(lado=lado))
            R_w = d.xmat[b_m].reshape(3, 3).copy()
            e_t, _, incl = _torre(m, d, lado, montaje)
            print(f"Mano {lado}: la mano sale {incl:.0f} grados inclinada respecto a su torre (diseno de la Orca)")
            palma = _palma(m, d, lado)
            dedos_obj = ORIENTACION_MANO["dedos"]
            e_obj = _antebrazo(m, d, lado)[0] if isinstance(dedos_obj, str) else np.asarray(dedos_obj, float)
            R_corr = _base(e_obj, ORIENTACION_MANO["palma"]) @ _base(e_t, palma).T   # giro en el mundo
            R_q = np.zeros(9)
            mujoco.mju_quat2Mat(R_q, np.asarray(montaje[lado]["quat"], float))
            R_local = R_w.T @ R_corr @ R_w @ R_q.reshape(3, 3)                      # visto desde la muneca
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, R_local.flatten())
            montaje[lado]["quat"] = [float(x) for x in q]
        m = _construir(ruta_xml, carpeta_orca, montaje)

    # 2) posicion: centro de la base de la torre = final de la muneca del TRON 2
    if lados_pos:
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        for lado in lados_pos:
            b_m = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CUERPO_MUNECA.format(lado=lado))
            R_w = d.xmat[b_m].reshape(3, 3)
            _, final = _antebrazo(m, d, lado)
            _, base, _ = _torre(m, d, lado, montaje)
            montaje[lado]["pos"] = [float(x) for x in np.asarray(montaje[lado]["pos"]) + R_w.T @ (final - base)]
        m = _construir(ruta_xml, carpeta_orca, montaje)

    for lado in ("L", "R"):
        print(f"Mano {lado}: MONTAJE pos=[{', '.join(f'{x:.4f}' for x in montaje[lado]['pos'])}], "
              f"quat=[{', '.join(f'{x:.4f}' for x in montaje[lado]['quat'])}]")
        _comprobar(m, lado, montaje)
    return m


def _normalizar(mundo, espejar):
    """Acepta los landmarks de MediaPipe 'clasico' (con .landmark) o de MediaPipe Tasks
    (lista de puntos) y devuelve siempre un objeto con .landmark. Con espejar=True
    refleja x -> -x (mano izquierda del operador)."""
    puntos = getattr(mundo, "landmark", mundo)
    s = -1.0 if espejar else 1.0
    return types.SimpleNamespace(landmark=[types.SimpleNamespace(x=s * p.x, y=p.y, z=p.z)
                                           for p in puntos])


# ============================== MANO =========================================
class ManoOrca:
    def __init__(self, m, d, lado):
        self.m, self.d, self.lado = m, d, lado
        pref = PREFIJO[lado]
        nombre = lambda tipo, i: mujoco.mj_id2name(m, tipo, i) or ""

        # Los 17 actuadores de esta mano, en su orden nativo (el mismo que usa config.py)
        acts = [a for a in range(m.nu) if nombre(mujoco.mjtObj.mjOBJ_ACTUATOR, a).startswith(pref)]
        if len(acts) != N_ACT:
            raise RuntimeError(f"Mano {lado}: esperaba {N_ACT} actuadores con prefijo {pref}, hay {len(acts)}")
        self.juntas = [int(m.actuator_trnid[a, 0]) for a in acts]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.juntas])
        self.lo = np.array([l for l, _ in LIMITES_CTRL], dtype=float)
        self.hi = np.array([h for _, h in LIMITES_CTRL], dtype=float)
        print(f"\nOrcaHand {lado}: orden de juntas (comprobar que coincide con LIMITES_CTRL de config.py)")
        for i, j in enumerate(self.juntas):
            print(f"   {i:2d}  {nombre(mujoco.mjtObj.mjOBJ_JOINT, j)}")

        self.signo = SIGNO_IZQ if lado == "L" else np.ones(N_ACT)
        self.q = MANO_ABIERTA.copy()
        self.objetivo = MANO_ABIERTA.copy()

        # El retargeting escribe en datos.ctrl[OFFSET_MANO:...]: le damos unos "datos"
        # de mentira para no tocar retargeting.py. ncon=0 porque aqui no hay contactos.
        self._falso = types.SimpleNamespace(ctrl=np.zeros(OFFSET_MANO + N_ACT), time=0.0,
                                            ncon=0, contact=[])
        self._falso.ctrl[OFFSET_MANO:] = self.q
        self.estado = dict(camara_activa=False,   # False: apaga el print de depuracion
                           muneca_bloqueada=False, pinza_previa=False, ultima_pos_palma=0.0,
                           vec_neutro=None, eje_lateral=None, calc_muneca=0.0,
                           dedo_activo="indice", modo_pinza=False, dedo_bloqueado=None,
                           pinza_tocando=False, pinza_habilitada=True)
        self.suav = SuavizadorND(n_dim=N_ACT - 1, ventana=4)

        # Palma = cuerpo que mueve la junta 0 (muneca de la Orca). Yemas = cuerpos finales
        # que cuelgan de la palma (asi no cuentan la torre ni el antebrazo de la Orca).
        self.palma = int(m.jnt_bodyid[self.juntas[0]])

        def cuelga_de_palma(b):
            while b > 0:
                if b == self.palma:
                    return True
                b = int(m.body_parentid[b])
            return False

        cuerpos = [b for b in range(m.nbody) if b != self.palma and cuelga_de_palma(b)]
        hojas = [b for b in cuerpos if not any(int(m.body_parentid[c]) == b for c in cuerpos)]
        self.geoms_yemas = [g for b in hojas
                            for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])]
        self.acoples = self._buscar_acoples(pref)
        self._escribir()

    # ---------------------------------------------------------------- juntas acopladas
    def _buscar_acoples(self, pref):
        """Juntas que en el XML van 'atadas' a otra con <equality joint>. Sin fisica
        MuJoCo no las resuelve solo, asi que se calculan aqui con su polinomio."""
        m, res = self.m, []
        for e in range(m.neq):
            if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            if not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j1) or "").startswith(pref):
                continue
            a1 = int(m.jnt_qposadr[j1])
            a2 = int(m.jnt_qposadr[j2]) if j2 >= 0 else -1
            res.append((a1, a2, float(m.qpos0[a1]), float(m.qpos0[a2]) if a2 >= 0 else 0.0,
                        np.array(m.eq_data[e][:5], dtype=float)))
        if res:
            print(f"   {len(res)} juntas acopladas por <equality>")
        return res

    def _escribir(self):
        self.d.qpos[self.qadr] = self.q * self.signo
        for a1, a2, y0, x0, c in self.acoples:
            x = self.d.qpos[a2] - x0 if a2 >= 0 else 0.0
            self.d.qpos[a1] = y0 + c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4

    # ---------------------------------------------------------------- objetivos
    def nuevo_objetivo(self, mundo, espejar=False):
        """Llamar UNA vez por fotograma nuevo de la camara.
        mundo: hand_world_landmarks de MediaPipe (21 puntos en metros).
        espejar=True si es una mano IZQUIERDA del operador."""
        mundo = _normalizar(mundo, espejar)
        lm3d = [np.array([p.x, p.y, p.z], dtype=float) for p in mundo.landmark]
        self.estado = retarget_a_ctrl(self.m, self._falso, lm3d, mundo, self.estado, self.suav)
        self.objetivo = self._falso.ctrl[OFFSET_MANO:].copy()
        self.objetivo[0] = 0.0     # muneca de la Orca recta (como la del TRON 2: solo gira)

    def objetivo_por_apertura(self, apertura):
        """Modo sencillo si no hay landmarks de los dedos: 1 = abierta, 0 = puno."""
        c = 1.0 - float(np.clip(apertura, 0.0, 1.0))
        self.objetivo = MANO_ABIERTA + c * (MANO_CERRADA - MANO_ABIERTA)

    # ---------------------------------------------------------------- movimiento
    def mover(self, dt, tope=None):
        """Llamar en CADA ciclo del bucle: lleva las 17 juntas hacia el objetivo con
        velocidad limitada. tope: si no es None, la flexion no pasa de ese valor."""
        objetivo = np.clip(self.objetivo, self.lo, self.hi)
        if tope is not None:
            objetivo[DEDOS_FLEXION] = np.minimum(objetivo[DEDOS_FLEXION], tope)
        paso = VEL_DEDOS * dt
        self.q += np.clip(objetivo - self.q, -paso, paso)
        self._escribir()

    def reposo(self):
        self.q = MANO_ABIERTA.copy()
        self.objetivo = MANO_ABIERTA.copy()
        self._falso.ctrl[OFFSET_MANO:] = self.q
        self.suav.reset()
        self._escribir()

    # ---------------------------------------------------------------- para el escenario
    def punto_agarre(self):
        """Punto entre la palma y las yemas: donde queda un tubo agarrado con la mano."""
        if not self.geoms_yemas:
            return self.d.xpos[self.palma].copy()
        yemas = self.d.geom_xpos[self.geoms_yemas].mean(axis=0)
        return 0.5 * (yemas + self.d.xpos[self.palma])

    def orientacion(self):
        return self.d.xmat[self.palma].reshape(3, 3)


# ============================== PRUEBA RAPIDA =================================
if __name__ == "__main__":
    import mujoco.viewer
    from robot_tron2 import quitar_base_flotante

    src = sys.argv[1] if len(sys.argv) > 1 else "tron2a/DACH_TRON2A/xml/robot_elecnor.xml"
    m = montar_manos(quitar_base_flotante(src, quitar_pinzas))
    d = mujoco.MjData(m)
    manos = {l: ManoOrca(m, d, l) for l in ("L", "R")}
    print("\nLas dos manos abren y cierran solas. Ajusta MONTAJE hasta que queden bien en la muneca.")
    t0 = time.time()
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            s = 0.5 + 0.5 * np.cos(time.time() - t0)          # 1 -> 0 -> 1
            for mano in manos.values():
                mano.objetivo_por_apertura(s)
                mano.mover(0.02)
            mujoco.mj_kinematics(m, d)
            v.sync()
            time.sleep(0.02)