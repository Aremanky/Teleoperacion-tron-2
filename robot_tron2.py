"""
TRON 2 (DACH_TRON2A) en MuJoCo: carga con la base fija + IK de los dos brazos.

La IK no depende del orden ni de los nombres exactos de las articulaciones:
recorre la cadena desde el cuerpo final de cada brazo (grasper_L_Link /
wrist_roll_L_Link), se queda con las bisagras de ese lado y separa
hombro / codo / muneca buscando "elbow" en el nombre. Al arrancar imprime
lo que ha detectado para poder revisarlo.

Tareas de la IK (minimos cuadrados amortiguados, en modo cinematico):
  - llevar el CODO del robot a su objetivo
  - llevar el CENTRO DE LA MUNECA del robot a su objetivo
  - orientar la PALMA como la del operador (solo con las 3 juntas de la muneca,
    que es esferica: no mueve la posicion). La orientacion se compara entre
    marcos SEMANTICOS iguales en el humano y en el robot: eje 0 hacia los dedos,
    eje 1 del nudillo del indice al del menique, eje 2 = normal de la palma.
  - en el espacio nulo: volver a reposo y alejarse de los limites articulares
Ademas: respeta los limites (bloquea la articulacion y compensan las demas),
solo acepta pasos que reducen el error, limita la velocidad articular y,
si el brazo se atasca en un minimo local, busca otra solucion.

CAMBIOS (oct-2026, imitacion):
  - Antes la IK recibia una orientacion CONSTANTE (marco_mano con la identidad) y
    mover_muneca() devolvia las juntas "rigidas" de la muneca al reposo en cada ciclo:
    las dos tareas se peleaban, la busqueda lineal rechazaba pasos buenos y la muneca
    no copiaba al operador. Ahora hay una sola tarea de orientacion, con el marco real
    de la mano, y mover_muneca() ha desaparecido.
  - Sin mano a la vista se mantiene la ULTIMA orientacion objetivo (no se vuelve al reposo).
  - El rescate de posturas incluye la muneca.

CAMBIOS (oct-2026, movimiento natural):
  1. Limites ANATOMICOS del hombro (LIMITES_ANATOMICOS). Los rangos humanos se ponian como
     limites de cada junta, y eso solo es cierto con el brazo colgando: el hombro son 3
     bisagras en cadena (angulos de Euler) y con el brazo abducido 90 grados la junta de
     "flexion" hace abduccion HORIZONTAL. Ahora se mide la postura del humero respecto al
     torso (swing-twist) y se limita eso: un cono irregular de elevacion segun el plano en
     que se levanta el brazo y una rotacion humeral que se acopla con la elevacion. Es una
     tarea BLANDA de la IK que solo actua al pasarse; tambien la respetan los rescates.
  2. Limites BLANDOS en las juntas (minima norma ponderada, Chan & Dubey 1995): una junta
     cuesta mas de mover cuanto mas cerca esta de su limite y se esta acercando a el, asi
     que la IK reparte el movimiento ANTES de chocar en vez de bloquearla de golpe (el
     bloqueo duro se queda como red de seguridad).
  3. Giro del codo natural: cuando el objetivo del codo pierde peso (brazo casi recto), el
     espacio nulo prefiere el codo hacia ABAJO y algo hacia FUERA, como cuelga un codo humano,
     en vez de lo que dijera q_reposo.
  4. Atascos: el detector comparaba el error con el del ciclo anterior y cualquier temblor del
     objetivo lo reiniciaba (a 60 Hz, con 4 mm de ruido, no saltaba nunca). Ahora mira los
     ultimos T_ATASCO s (DetectorAtasco). Y el rescate se compara con la postura actual usando la
     MISMA medida (posicion + limite anatomico); antes se comparaba una medida con penalizacion
     anatomica contra otra sin ella y se podian rechazar rescates buenos.
"""
import os
import time
import xml.etree.ElementTree as ET
from collections import deque

import mujoco
import numpy as np

from suavizado import log_rot, ortonormalizar

HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)


def unwrap_angular_delta(delta):
    """Devuelve el delta angular sin saltos de 2π al cruzar ±π."""
    delta = np.asarray(delta, dtype=float)
    if not np.all(np.isfinite(delta)):
        return np.zeros_like(delta, dtype=float)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def angulo_giro(R, eje):
    """Parte de la rotacion R que es giro alrededor de 'eje' (unitario), en rad.
    Descomposicion swing-twist (se conserva para diagnostico)."""
    w = np.sqrt(max(0.0, 1.0 + np.trace(R))) / 2.0
    if w > 1e-6:
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (4.0 * w)
    else:
        v = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
    ang = 2.0 * np.arctan2(float(np.dot(v, eje)), w)
    return float((ang + np.pi) % (2 * np.pi) - np.pi)


# Cuerpo final de cada brazo, por orden de preferencia ({lado} = L o R)
CANDIDATOS_EF = ["grasper_{lado}_Link", "grasper_base_{lado}_Link", "wrist_roll_{lado}_Link"]

# Distancia maxima (m) del centro estimado a cada eje para considerar
# que el hombro / la muneca son esfericos (los tres ejes se cortan)
TOL_ESFERICO = 0.03

VEL_MAX = 3.5          # rad/s: velocidad maxima de cada articulacion al seguir al operador
W_ORIENT = 0.12        # m/rad: peso de la orientacion de la palma frente a la posicion
                       # (1 rad = 57 grados de error "cuesta" lo mismo que 12 cm de posicion)
VEL_PINZA = 2.5        # 1/s: velocidad de apertura y cierre de la pinza (0 a 1 en 0.4 s)
PINZA_INVERTIDA = False  # True si toda la pinza abre cuando deberia cerrar
# Juntas que llevan "grasp" en el nombre pero NO forman parte del mecanismo de la
# pinza (p.ej. la que orienta la pinza entera). Se dejan quietas.
PINZA_EXCLUIR = ("grasper_base", "orca")   # "orca": juntas de las OrcaHand (las mueve mano_orca.py)
PINZA_JUNTAS_INVERTIDAS = ()
PINZA_RECORRIDO = {}
VEL_TRANSICION = 1.0   # rad/s: al cambiar a otra solucion tras un rescate
MARGEN_LIMITE = 0.1    # rad: junto a un limite se empuja la articulacion hacia dentro
ERROR_ATASCO = 0.10    # m: error a partir del cual se considera que el brazo esta atascado
T_ATASCO = 0.5         # s: tiempo atascado antes de buscar otra solucion
MEJORA_MIN = 0.05      # m/s: si el error baja mas rapido que esto, el brazo NO esta atascado
                       # (oct-2026: medido sobre los ultimos T_ATASCO s, no foto a foto; ver DetectorAtasco)
MEJORA_RESCATE = 0.05  # m: un rescate tambien se acepta si mejora al menos esto (ademas de si divide el error a la mitad)
T_TRANSICION_MAX = 2.0 # s: una transicion de rescate nunca dura mas que esto
CANCELA_TRANSICION = 0.10  # m: si el objetivo se mueve esto durante la transicion, se cancela
# Pinza original (simetrica): su "normal de la palma" es el eje de cierre de las mordazas, y
# el signo se elige para que con los brazos colgando quede como la mano de una persona
# relajada: palma hacia el muslo (hacia +y en la derecha). Para la izquierda el tercer eje
# del marco semantico es el DORSO, que tambien apunta a +y. Asi el reposo de la pinza
# coincide con el del operador y la muneca no arranca pegada a un limite.
NORMAL_PINZA_REPOSO = np.array([0.0, 1.0, 0.0])
# Postura de reposo VISUAL (arranque y tecla r): codo flexionado como en un curl de biceps.
# Solo estetica: la calibracion [c] sigue siendo con el operador con los brazos colgando, y la
# IK sigue usando q_reposo (brazo recto) para el espacio nulo y los rescates. 0 = como antes.
CODO_REPOSO_DEG = 90.0
# Giro humeral (ultima junta del hombro) que abre los antebrazos hacia fuera en ese reposo:
# separa las munecas entre si sin despegar los codos del cuerpo. 0 = antebrazos paralelos.
SEPARACION_MUNECAS_DEG = -15.0
# Abduccion del hombro en ese reposo: separa los CODOS del cuerpo. 0 = codos pegados.
SEPARACION_CODOS_DEG = 20.0
# Rango del codo en grados de flexion "humana" (0 = brazo recto). None = limites del XML.
FLEXION_CODO_DEG = (0.0, 160.0)
# Rangos humanos del hombro en grados (None = limites del XML):
#   flexion:   + = brazo hacia delante/arriba, - = extension (hacia atras)
#   abduccion: + = brazo hacia el lado,        - = aduccion (cruzar por delante del cuerpo)
#   rotacion:  con el codo a 90, + = externa (antebrazo hacia fuera), - = interna
HOMBRO_HUMANO_DEG = dict(flexion=(-60.0, 180.0), abduccion=(-30.0, 180.0), rotacion=(-70.0, 90.0))

# --- (1) Limites ANATOMICOS del hombro --------------------------------------------------------
# True: los rangos de HOMBRO_HUMANO_DEG se aplican a la postura REAL del humero (ver cabecera) y
# las juntas del hombro solo conservan una caja amplia de seguridad. False: como antes (por junta).
LIMITES_ANATOMICOS = True
# Elevacion maxima del brazo (grados desde colgando) segun el PLANO en que se levanta:
#   0 = hacia fuera (abduccion), 90 = hacia delante (flexion), -90 = hacia atras (extension),
#   +-180 = hacia el otro lado (aduccion). Los puntos -90, 0, 90 y +-180 salen de
#   HOMBRO_HUMANO_DEG; estos son los intermedios: abrir el brazo hacia atras a la altura del
#   hombro (abduccion horizontal) y cruzarlo por delante del pecho (aduccion horizontal).
CONO_HOMBRO_EXTRA_DEG = {-45.0: 100.0, 135.0: 125.0}
# Acople rotacion-elevacion: con el brazo muy levantado se pierde rotacion INTERNA
# (pares elevacion -> grados de rotacion interna que se pierden; se interpola entre ellos).
ACOPLE_ROTACION_DEG = ((120.0, 0.0), (180.0, 50.0))
# Caja de seguridad de las juntas del hombro: el rango humano por junta ampliado esto por cada lado
# (evita que la IK salte a la otra solucion de Euler; nunca pasa de los limites del XML).
HOLGURA_CAJA_HOMBRO_DEG = 45.0
W_ANAT = 0.25          # m/rad: peso del limite anatomico (1 rad pasado "cuesta" como 25 cm)
SUAVE_ANAT_DEG = 3.0   # transicion del limite anatomico: sin escalon al tocarlo

# --- (2) Limites BLANDOS de las juntas --------------------------------------------------------
# Cuanto mas "cara" de mover puede llegar a ser una junta pegada a su limite (1 = desactivado).
PESO_LIMITE_MAX = 50.0

# --- (3) Giro del codo sin objetivo de codo ---------------------------------------------------
K_CODO_ABAJO = 0.03    # fuerza de la preferencia "codo hacia abajo" en el espacio nulo (0 = no)
CODO_FUERA = 0.4       # cuanto hacia fuera, ademas de hacia abajo
W_CODO_PLENO = 0.4     # con w_codo >= esto manda el objetivo del codo y la preferencia no actua
# Bloquear la flexion/extension de la muneca (se queda en 0, la de reposo)
BLOQUEAR_FLEXION_MUNECA = True
# Bloquear la desviacion radial/ulnar de la muneca (se queda en 0, la de reposo)
BLOQUEAR_DESVIACION_MUNECA = True
SOPORTE_KUKA = os.environ.get("TRON2_SOPORTE", "kuka") == "kuka"   # arranque.sh: kuka | columna


def nombre(m, tipo, i):
    return mujoco.mj_id2name(m, tipo, i) or f"<{i}>"


def quitar_base_flotante(ruta_xml, extra=None):
    """Crea robot_fixed.xml junto al original, sin free joint (torso fijo al mundo).
    'extra(root)', si se pasa, puede anadir un escenario al modelo antes de guardarlo."""
    tree = ET.parse(ruta_xml)
    root = tree.getroot()
    quitados = 0
    for padre in root.iter():
        for hijo in list(padre):
            if hijo.tag == "freejoint" or (hijo.tag == "joint" and hijo.get("type") == "free"):
                padre.remove(hijo)
                quitados += 1
    if quitados:  # los keyframes pierden los 7 qpos / 6 qvel de la base libre
        for key in root.iter("key"):
            if key.get("qpos"):
                key.set("qpos", " ".join(key.get("qpos").split()[7:]))
            if key.get("qvel"):
                key.set("qvel", " ".join(key.get("qvel").split()[6:]))
    if extra is not None:
        extra(root)
    ruta_fija = os.path.join(os.path.dirname(os.path.abspath(ruta_xml)), "robot_fixed.xml")
    tree.write(ruta_fija)
    return ruta_fija


def centro_de_ejes(d, juntas):
    """Punto mas cercano (minimos cuadrados) a los ejes de las articulaciones dadas.
    En un hombro o una muneca esferica es su centro de giro.
    Devuelve (punto, distancia maxima del punto a los ejes)."""
    anclas = np.array([d.xanchor[j] for j in juntas])
    ejes = np.array([d.xaxis[j] / np.linalg.norm(d.xaxis[j]) for j in juntas])
    A = 1e-4 * np.eye(3)
    b = 1e-4 * anclas.mean(axis=0)
    for a, c in zip(ejes, anclas):
        P = np.eye(3) - np.outer(a, a)
        A += P
        b += P @ c
    p = np.linalg.solve(A, b)
    residuo = max(np.linalg.norm((np.eye(3) - np.outer(a, a)) @ (p - c)) for a, c in zip(ejes, anclas))
    return p, residuo


def marco_semantico(f, a):
    """Marco de una mano a partir de la direccion de los dedos (f) y del vector del nudillo
    del indice al del menique (a): columnas [dedos, ancho de la palma, normal]. Es la MISMA
    construccion que SeguidorManos.marco() con los landmarks de MediaPipe, para que las
    orientaciones del humano y del robot sean comparables. Para la mano izquierda el
    tercer eje sale por el dorso (la construccion es la misma: se compara igual con igual)."""
    f = np.asarray(f, float) / np.linalg.norm(f)
    a = np.asarray(a, float)
    a = a - f * np.dot(a, f)
    a /= np.linalg.norm(a)
    return np.column_stack([f, a, np.cross(f, a)])


# ---------------- anatomia del hombro (funciones puras, sin MuJoCo) ----------------
def angulos_hombro(R_torso, fuera, humero, Q, a0):
    """Postura anatomica del humero respecto al torso, en grados: (elevacion, plano, giro).
      R_torso: orientacion del torso en el mundo (x delante, y izquierda, z arriba)
      fuera:   vector (coordenadas del torso) que apunta hacia fuera de este lado: +-y
      humero:  direccion hombro -> codo en el mundo
      Q:       rotacion del humero desde su postura de reposo, en coordenadas del torso
      a0:      direccion del humero en reposo, en coordenadas del torso
    elevacion: 0 = colgando, 90 = horizontal, 180 = vertical.
    plano: 0 = hacia fuera, 90 = hacia delante, -90 = hacia atras, +-180 = hacia el otro lado.
    giro: rotacion humeral desde el reposo (twist de la descomposicion swing-twist alrededor
    del eje de reposo), en el convenio de las juntas: el signo anatomico lo pone quien llama."""
    b = R_torso.T @ np.asarray(humero, float)
    b = b / max(np.linalg.norm(b), 1e-12)
    elev = float(np.degrees(np.arccos(np.clip(-b[2], -1.0, 1.0))))
    plano = float(np.degrees(np.arctan2(b[0], float(np.dot(b, fuera)))))
    giro = float(np.degrees(angulo_giro(Q, a0)))
    return elev, plano, giro


def tabla_cono(rangos, extra=CONO_HOMBRO_EXTRA_DEG):
    """(planos, elevaciones maximas) del cono del hombro a partir de HOMBRO_HUMANO_DEG."""
    ext, add = -rangos["flexion"][0], -rangos["abduccion"][0]
    puntos = {-180.0: add, -90.0: ext, 0.0: rangos["abduccion"][1],
              90.0: rangos["flexion"][1], 180.0: add}
    puntos.update(extra or {})
    planos = np.array(sorted(puntos))
    return planos, np.array([puntos[p] for p in planos])


def bisagra_suave(v, ancho):
    """max(0, v) sin esquina: cuadratica en [0, ancho] y lineal (pendiente 1) despues.
    Devuelve (valor, derivada)."""
    v = np.asarray(v, float)
    val = np.where(v <= 0, 0.0, np.where(v < ancho, v * v / (2 * ancho), v - ancho / 2))
    der = np.where(v <= 0, 0.0, np.where(v < ancho, v / ancho, 1.0))
    return val, der


def pesos_limites(q, lo, hi, limitado, dH_ant, tope=PESO_LIMITE_MAX):
    """Pesos de la minima norma ponderada (Chan & Dubey 1995): 1 + |dH/dq| con
    H = sum (hi-lo)^2 / (4 (hi-q)(q-lo)), solo para las juntas que se ACERCAN a su limite
    (|dH/dq| crece); las que se alejan pesan 1. Devuelve (pesos, |dH/dq| para el siguiente paso)."""
    rango = hi - lo
    ok = limitado & (rango > 1e-6)                    # las juntas bloqueadas (lo == hi) no cuentan
    dH = np.zeros_like(q, dtype=float)
    a, b = lo[ok], hi[ok]
    x = np.clip(q[ok], a + 1e-6 * rango[ok], b - 1e-6 * rango[ok])
    dH[ok] = np.abs(rango[ok] ** 2 * (2 * x - a - b) / (4 * (b - x) ** 2 * (x - a) ** 2))
    pesos = np.where(dH >= dH_ant, 1.0 + dH, 1.0)
    return np.minimum(pesos, tope), dH


class DetectorAtasco:
    """El brazo esta atascado si su error lleva T_ATASCO s por encima de ERROR_ATASCO y en ese
    tiempo ha bajado menos de MEJORA_MIN * T_ATASCO.
    (oct-2026) Antes se comparaba el error con el del ciclo ANTERIOR: un solo ciclo en que el
    error bajara un poco (ruido del objetivo, que tiembla unos mm) ponia el contador a cero, y con
    un objetivo vivo el rescate podia no llegar nunca (sesion de prueba: brazo izquierdo con 42 cm
    de error durante segundos y los brazos del operador quietos)."""

    def __init__(self, umbral=ERROR_ATASCO, ventana=T_ATASCO, mejora=MEJORA_MIN):
        self.umbral, self.ventana, self.mejora = umbral, ventana, mejora
        self.hist = deque()

    def reiniciar(self):
        self.hist.clear()

    def actualizar(self, t, err):
        """Anota el error del instante t. Devuelve True si el brazo esta atascado."""
        if err <= self.umbral:
            self.hist.clear()
            return False
        self.hist.append((t, err))
        while len(self.hist) > 1 and t - self.hist[1][0] >= self.ventana:
            self.hist.popleft()
        t0, e0 = self.hist[0]
        if t - t0 < self.ventana:
            return False
        return e0 - err < self.mejora * (t - t0)


class Brazo:
    def __init__(self, m, d, lado, reloj=time.monotonic):
        self.m, self.d, self.lado = m, d, lado
        self.reloj = reloj   # inyectable: la reproduccion de sesiones usa un reloj simulado
        self.mano = None   # OrcaHand montada en esta muneca (la asigna RobotTron2 con orca=True)
        self.ef = self._buscar_ef()
        cadena = self._cadena(self.ef)
        self.raiz = cadena[0]
        juntas = self._juntas(cadena)
        nombres = [nombre(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in juntas]

        i_codo = next((i for i, n in enumerate(nombres) if "elbow" in n.lower()), None)
        if i_codo is None and len(juntas) == 7:
            i_codo = 3  # brazo 7 GDL tipo S-R-S: 3 hombro + codo + 3 muneca
        if i_codo is None or i_codo == 0 or i_codo == len(juntas) - 1:
            raise RuntimeError(f"Brazo {lado}: no se separar hombro/codo/muneca en {nombres}")
        self.juntas, self.nombres = juntas, nombres
        self.j_hombro = juntas[:i_codo]
        self.j_codo = juntas[i_codo]
        self.j_muneca = juntas[i_codo + 1:]

        self.qadr = np.array([m.jnt_qposadr[j] for j in juntas])
        self.dofs = np.array([m.jnt_dofadr[j] for j in juntas])
        self.limitado = np.array([bool(m.jnt_limited[j]) for j in juntas])
        self.lo = np.array([m.jnt_range[j][0] for j in juntas], dtype=float)
        self.hi = np.array([m.jnt_range[j][1] for j in juntas], dtype=float)
        self.q_reposo = np.where(self.limitado, np.clip(0.0, self.lo, self.hi), 0.0)

        self._preparar_pinza()
        self._preparar_agarre()

        # --- Geometria del brazo, medida en la postura de reposo (brazo recto) ---
        self.reposo(self.q_reposo)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

        self.hombro, r_h = centro_de_ejes(d, self.j_hombro)
        if r_h > TOL_ESFERICO:
            print(f"[aviso] hombro {lado} no parece esferico (residuo {r_h * 1000:.0f} mm): "
                  f"uso el ancla de la primera articulacion")
            self.hombro = d.xanchor[self.j_hombro[0]].copy()

        muneca, r_m = centro_de_ejes(d, self.j_muneca)
        self.b_muneca = int(m.jnt_bodyid[self.j_muneca[0]])
        if r_m > TOL_ESFERICO:
            print(f"[aviso] muneca {lado} no parece esferica (residuo {r_m * 1000:.0f} mm): "
                  f"uso el ancla de la ultima articulacion")
            muneca = d.xanchor[self.j_muneca[-1]].copy()
            self.b_muneca = int(m.jnt_bodyid[self.j_muneca[-1]])

        codo = d.xanchor[self.j_codo].copy()
        self.b_codo = int(m.jnt_bodyid[self.j_codo])
        self.loc_codo = self._a_local(self.b_codo, codo)
        self.loc_muneca = self._a_local(self.b_muneca, muneca)
        self.L_brazo = float(np.linalg.norm(codo - self.hombro))
        self.L_antebrazo = float(np.linalg.norm(muneca - codo))
        self._jac = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        # columnas de la muneca: las unicas que atienden a la orientacion de la palma
        self.mascara_muneca = np.array([j in self.j_muneca for j in self.juntas], dtype=float)
        # junta de GIRO de la muneca: la de eje mas alineado con el antebrazo (diagnostico)
        antebrazo = (muneca - codo) / max(np.linalg.norm(muneca - codo), 1e-9)
        self.j_giro = max(self.j_muneca, key=lambda j: abs(np.dot(d.xaxis[j], antebrazo)))
        self.i_giro = self.juntas.index(self.j_giro)
        self.i_muneca = [self.juntas.index(j) for j in self.j_muneca]
        # orientacion: cuerpo que la lleva y marco semantico (en ese cuerpo). Con la pinza
        # original se calcula aqui; con la OrcaHand lo fija RobotTron2 (preparar_orientacion)
        self.b_orient = int(self.ef)
        self.marco_sem_local = self._marco_pinza()
        self.R_obj_ult = None
        self.err_ant, self.obj_rescate = 0.0, None
        self.q_transicion = None
        self.t_atascado, self.t_rescate = 0.0, -1e9
        self.atasco = DetectorAtasco()
        self.rng = np.random.default_rng(0)
        self.anat = None                       # limites anatomicos del hombro (_limitar_hombro)
        self._dH_ant = np.zeros(len(juntas))   # limites blandos: |dH/dq| del paso anterior
        # postura visual de reposo (curl) y el brazo se queda en ella al arrancar
        if FLEXION_CODO_DEG is not None:
            self._limitar_codo(*FLEXION_CODO_DEG)
        if HOMBRO_HUMANO_DEG is not None:
            self._limitar_hombro(HOMBRO_HUMANO_DEG)
        self.q_reposo_visual = self._postura_visual()
        self.reposo()

    def _postura_visual(self):
        """q_reposo con el codo doblado CODO_REPOSO_DEG y el antebrazo abierto hacia fuera
        SEPARACION_MUNECAS_DEG. Los signos se eligen solos (no dependen del convenio del URDF):
        el codo, el que lleva la muneca mas hacia delante del torso; el giro humeral, el que
        la lleva mas hacia fuera (y del robot: +y = su izquierda)."""
        q = self.q_reposo.copy()
        R_torso = self.d.xmat[self.raiz].reshape(3, 3)
        delante = R_torso[:, 0]
        fuera = R_torso[:, 1] if self.lado == "L" else -R_torso[:, 1]

        def mejor_signo(q, i, grados, direccion, punto=1):
            if not grados:
                return q
            mejor, mejor_v = q, -np.inf
            for signo in (1.0, -1.0):
                p = q.copy()
                p[i] += signo * np.radians(grados)
                p = self._limitar(p)
                self._fijar(p)
                v = float(np.dot(self.puntos()[punto] - self.hombro, direccion))
                if v > mejor_v:
                    mejor, mejor_v = p, v
            return mejor
        if SEPARACION_CODOS_DEG and len(self.j_hombro) >= 2:
            def codo_fuera(p):
                self._fijar(p)
                return float(np.dot(self.puntos()[0] - self.hombro, fuera))
            # de las dos primeras juntas del hombro, la que mas abre el codo (la abduccion)
            q = max((mejor_signo(q, i, SEPARACION_CODOS_DEG, fuera, punto=0) for i in (0, 1)),
                    key=codo_fuera)
        q = mejor_signo(q, self.juntas.index(self.j_codo), CODO_REPOSO_DEG, delante)
        if len(self.j_hombro) >= 3:
            # positivo = antebrazos hacia fuera, negativo = hacia dentro (munecas mas juntas)
            q = mejor_signo(q, len(self.j_hombro) - 1, abs(SEPARACION_MUNECAS_DEG),
                            fuera if SEPARACION_MUNECAS_DEG > 0 else -fuera)
        self._fijar(self.q_reposo)
        return q

    def _flexion_codo(self):
        """Flexion actual del codo en grados (0 = brazo recto), medida con la geometria."""
        p_c, p_m = self.puntos()
        b, a = p_c - self.hombro, p_m - p_c
        return float(np.degrees(np.arccos(np.clip(
            np.dot(b, a) / (np.linalg.norm(b) * np.linalg.norm(a)), -1.0, 1.0))))

    def _limitar_codo(self, flex_min, flex_max):
        """Cambia los limites de la junta del codo a un rango de flexion humano."""
        i = self.juntas.index(self.j_codo)
        q = self.q_reposo.copy()
        delante = self.d.xmat[self.raiz].reshape(3, 3)[:, 0]

        def avance(dq):
            p = q.copy()
            p[i] += dq
            self._fijar(p)
            return float(np.dot(self.puntos()[1] - self.hombro, delante))

        signo = 1.0 if avance(0.3) > avance(-0.3) else -1.0   # sentido en que flexiona
        self._fijar(q)
        flex0 = self._flexion_codo()                          # flexion con q = q_reposo
        lo = q[i] + signo * np.radians(flex_min - flex0)
        hi = q[i] + signo * np.radians(flex_max - flex0)
        lo, hi = min(lo, hi), max(lo, hi)
        if self.limitado[i]:                                  # nunca mas alla de lo que da el robot
            lo, hi = max(lo, self.lo[i]), min(hi, self.hi[i])
        self.lo[i], self.hi[i], self.limitado[i] = lo, hi, True
        self.q_reposo[i] = np.clip(self.q_reposo[i], lo, hi)
        print(f"[IK {self.lado}] codo {self.nombres[i]}: [{np.degrees(lo):+.0f}, {np.degrees(hi):+.0f}] deg "
              f"(flexion {flex_min:.0f}-{flex_max:.0f}; con q=0 el brazo tiene {flex0:.0f} deg)")

    def _signo(self, q, i, punto, direccion, paso=0.3):
        """+1 o -1: sentido en que hay que girar la junta i (desde q) para llevar el punto
        (0 = codo, 1 = muneca) hacia 'direccion'."""
        def v(dq):
            p = q.copy()
            p[i] += dq
            self._fijar(p)
            return float(np.dot(self.puntos()[punto] - self.hombro, direccion))
        s = 1.0 if v(paso) > v(-paso) else -1.0
        self._fijar(self.q_reposo)
        return s

    def _poner_rango(self, i, q_i, signo, ang0, a_min, a_max, papel):
        """Limites de la junta i para que el angulo anatomico quede en [a_min, a_max] (grados).
        q_i: valor de la junta en la postura medida, donde el angulo anatomico vale ang0."""
        lo = q_i + signo * np.radians(a_min - ang0)
        hi = q_i + signo * np.radians(a_max - ang0)
        lo, hi = min(lo, hi), max(lo, hi)
        if self.limitado[i]:                    # nunca mas alla de lo que da el robot
            lo, hi = max(lo, self.lo[i]), min(hi, self.hi[i])
        self.lo[i], self.hi[i], self.limitado[i] = lo, hi, True
        self.q_reposo[i] = np.clip(self.q_reposo[i], lo, hi)
        print(f"[IK {self.lado}] hombro {papel:9s} {self.nombres[i]}: [{np.degrees(lo):+.0f}, {np.degrees(hi):+.0f}] deg "
              f"({a_min:.0f} a {a_max:.0f}; en reposo {ang0:+.0f})")

    def _limitar_hombro(self, rangos):
        """Rangos humanos en el hombro. Identifica cada junta por geometria con el brazo
        colgando: flexion = la que mas lleva el codo hacia delante, abduccion = la otra de
        las dos primeras, rotacion = la ultima (giro humeral)."""
        if len(self.j_hombro) < 3:
            print(f"[aviso] hombro {self.lado} con {len(self.j_hombro)} juntas: no se limita")
            return
        q = self.q_reposo.copy()
        R = self.d.xmat[self.raiz].reshape(3, 3)
        delante, arriba = R[:, 0], R[:, 2]
        fuera = R[:, 1] if self.lado == "L" else -R[:, 1]
        self._fijar(q)
        c0 = self.puntos()[0].copy()

        def avance(i):
            p = q.copy()
            p[i] += 0.3
            self._fijar(p)
            return abs(float(np.dot(self.puntos()[0] - c0, delante)))

        i_flex = max((0, 1), key=avance)
        i_abd = 1 - i_flex
        i_rot = len(self.j_hombro) - 1
        self._fijar(q)
        b = self.puntos()[0] - self.hombro                     # brazo (hombro -> codo)
        flex0 = np.degrees(np.arctan2(np.dot(b, delante), -np.dot(b, arriba)))
        abd0 = np.degrees(np.arctan2(np.dot(b, fuera), -np.dot(b, arriba)))
        s_flex = self._signo(q, i_flex, 0, delante)
        s_abd = self._signo(q, i_abd, 0, fuera)
        # rotacion: se mide con el codo a 90 grados (0 = antebrazo hacia delante, + = externa)
        i_c = self.juntas.index(self.j_codo)
        q90 = q.copy()
        q90[i_c] += self._signo(q, i_c, 1, delante) * np.pi / 2
        q90 = self._limitar(q90)
        self._fijar(q90)
        p_c, p_m = self.puntos()
        rot0 = np.degrees(np.arctan2(np.dot(p_m - p_c, fuera), np.dot(p_m - p_c, delante)))
        s_rot = self._signo(q90, i_rot, 1, fuera)
        self._fijar(self.q_reposo)
        if LIMITES_ANATOMICOS:
            if all(rangos.get(k) is not None for k in ("flexion", "abduccion", "rotacion")):
                self._calibrar_anatomia(q, R, i_rot, s_rot, rot0, rangos)
            else:
                print(f"[aviso] hombro {self.lado}: LIMITES_ANATOMICOS necesita los tres rangos "
                      f"de HOMBRO_HUMANO_DEG: sigo con limites por junta")
        if self.anat is not None:   # las juntas solo conservan una caja amplia de seguridad
            h = HOLGURA_CAJA_HOMBRO_DEG
            rangos = {k: (v[0] - h, v[1] + h) for k, v in rangos.items() if v is not None}
        for i, s, a0, clave, papel in ((i_flex, s_flex, flex0, "flexion", "flexion"),
                                       (i_abd, s_abd, abd0, "abduccion", "abduccion"),
                                       (i_rot, s_rot, rot0, "rotacion", "rotacion")):
            if rangos.get(clave) is not None:
                self._poner_rango(i, q[i], s, a0, *rangos[clave], papel)

    # ---------------- limites anatomicos del hombro ----------------
    def _calibrar_anatomia(self, q, R, i_rot, s_rot, rot0, rangos):
        """Prepara la medida anatomica del humero. q: postura de reposo (brazo colgando), donde
        la rotacion humeral vale rot0 (grados, medida por _limitar_hombro con el codo a 90).
        s_rot: sentido de la junta i_rot que es rotacion EXTERNA."""
        b_humero = int(self.m.jnt_bodyid[self.j_hombro[-1]])   # cuerpo que mueven las 3 juntas
        self._fijar(q)
        a0 = R.T @ (self.puntos()[0] - self.hombro)
        self.anat = dict(b=b_humero, H0=self.d.xmat[b_humero].reshape(3, 3).copy(),
                         a0=a0 / np.linalg.norm(a0),
                         fuera=np.array([0.0, 1.0 if self.lado == "L" else -1.0, 0.0]),
                         rot0=0.0, signo=1.0)
        p = q.copy()                      # signo: girar hacia la rotacion externa debe sumar
        p[i_rot] += s_rot * 0.2
        self._fijar(p)
        giro = self._anatomia()[2]
        self._fijar(self.q_reposo)
        if abs(giro) < 3.0:               # 0.2 rad son 11 grados: el eje no es el del humero
            print(f"[aviso] hombro {self.lado}: la junta {self.nombres[i_rot]} no gira el humero "
                  f"sobre su eje ({giro:.1f} deg): sigo con limites por junta")
            self.anat = None
            return
        planos, elev_max = tabla_cono(rangos)
        self.anat.update(rot0=float(rot0), signo=float(np.sign(giro)), planos=planos, elev_max=elev_max,
                         rot=tuple(rangos["rotacion"]),
                         acople=(np.array([e for e, _ in ACOPLE_ROTACION_DEG], dtype=float),
                                 np.array([r for _, r in ACOPLE_ROTACION_DEG], dtype=float)))
        elev, plano, rot = self._anatomia()
        cono = ", ".join(f"{p:+.0f}:{e:.0f}" for p, e in zip(planos, elev_max))
        print(f"[IK {self.lado}] hombro: limites ANATOMICOS (cono plano:elevacion max = {cono}; "
              f"rotacion {rangos['rotacion'][0]:.0f} a {rangos['rotacion'][1]:.0f}). "
              f"En reposo: elevacion {elev:.0f}, rotacion {rot:+.0f}. "
              f"Juntas con caja de seguridad +-{HOLGURA_CAJA_HOMBRO_DEG:.0f} deg:")

    def _anatomia(self):
        """(elevacion, plano de elevacion, rotacion humeral) actuales, en grados.
        Rotacion: + = externa, 0 = antebrazo hacia delante con el codo a 90 y el brazo colgando."""
        A = self.anat
        R = self.d.xmat[self.raiz].reshape(3, 3)
        H = self.d.xmat[A["b"]].reshape(3, 3)
        Q = R.T @ H @ A["H0"].T @ R
        elev, plano, giro = angulos_hombro(R, A["fuera"], self.puntos()[0] - self.hombro, Q, A["a0"])
        return elev, plano, A["rot0"] + A["signo"] * giro

    def anatomia(self):
        """Postura anatomica del hombro (diagnostico): dict en grados, o None si no hay limites."""
        if self.anat is None:
            return None
        elev, plano, rot = self._anatomia()
        return dict(elevacion=elev, plano=plano, rotacion=rot,
                    exceso_deg=np.degrees(self._castigo_anat()))

    def _castigo_anat(self):
        """Cuanto se pasa el hombro de sus limites anatomicos (rad, >= 0, sin esquinas):
        [elevacion fuera del cono, rotacion interna de mas, rotacion externa de mas]."""
        A = self.anat
        elev, plano, rot = self._anatomia()
        e_max = float(np.interp(plano, A["planos"], A["elev_max"]))
        r_min = A["rot"][0] + float(np.interp(elev, *A["acople"]))
        r_max = A["rot"][1]
        f = float(np.clip((175.0 - elev) / 15.0, 0.0, 1.0))   # con el brazo vertical el giro no esta definido
        v = np.radians([elev - e_max, f * (r_min - rot), f * (rot - r_max)])
        return bisagra_suave(v, np.radians(SUAVE_ANAT_DEG))[0]

    def _jac_anat(self, q, c0, eps=1e-4):
        """Jacobiano numerico del castigo anatomico (solo dependen de el las juntas del hombro).
        Deja el modelo en q."""
        Jh = np.zeros((len(c0), len(q)))
        for i in range(len(self.j_hombro)):
            p = q.copy()
            p[i] += eps
            self._fijar(p)
            Jh[:, i] = (self._castigo_anat() - c0) / eps
        self._fijar(q)
        return Jh

    def bloquear_muneca(self, flexion=True, desviacion=True, valor=0.0):
        """Fija juntas de la muneca (la de giro nunca se toca):
          flexion/extension -> la de eje mas paralelo al ANCHO de la palma
          desviacion radial/ulnar -> la otra (eje casi paralelo a la NORMAL de la palma)."""
        mujoco.mj_kinematics(self.m, self.d)
        ancho = self.orientacion()[:, 1]
        candidatas = [j for j in self.j_muneca if j != self.j_giro]
        j_flex = max(candidatas, key=lambda j: abs(np.dot(self.d.xaxis[j], ancho)))
        j_desv = next((j for j in candidatas if j != j_flex), None)
        for j, activo, papel in ((j_flex, flexion, "flexion/extension"),
                                 (j_desv, desviacion, "desviacion radial/ulnar")):
            if not activo or j is None:
                continue
            i = self.juntas.index(j)
            self.lo[i] = self.hi[i] = valor
            self.limitado[i] = True
            self.q_reposo[i] = self.q_reposo_visual[i] = valor
            self.d.qpos[self.qadr[i]] = valor
            print(f"[IK {self.lado}] {papel} de la muneca BLOQUEADA: {self.nombres[i]} = {np.degrees(valor):.0f} deg")

    # ---------------- deteccion de la cadena ----------------
    def _buscar_ef(self):
        for patron in CANDIDATOS_EF:
            i = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, patron.format(lado=self.lado))
            if i >= 0:
                return i
        cuerpos = [nombre(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(self.m.nbody)]
        raise RuntimeError(f"No encuentro el cuerpo final del brazo {self.lado}. "
                           f"Ajusta CANDIDATOS_EF. Cuerpos del modelo: {cuerpos}")

    def _cadena(self, b):
        cadena = []
        while b > 0:  # 0 = world
            cadena.append(int(b))
            b = int(self.m.body_parentid[b])
        return cadena[::-1]

    def _juntas(self, cadena):
        m = self.m
        bisagras = [j for b in cadena
                    for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])
                    if m.jnt_type[j] == HINGE]
        marca = f"_{self.lado}_"
        propias = [j for j in bisagras
                   if marca in nombre(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                   and "grasp" not in nombre(m, mujoco.mjtObj.mjOBJ_JOINT, j).lower()]
        if propias:
            return propias
        print(f"[aviso] ninguna articulacion contiene '{marca}': uso todas las bisagras de la cadena")
        return bisagras

    # ---------------- pinza ----------------
    def _preparar_pinza(self):
        """Busca las articulaciones de la pinza: las que cuelgan del efector final,
        mas las que llevan 'grasp' en el nombre dentro de este brazo, quitando las
        de PINZA_EXCLUIR. Para cada una, el extremo del recorrido mas cercano a
        cero se toma como pinza cerrada y el otro como abierta."""
        m = self.m
        descendientes = set()
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p == self.ef:
                    descendientes.add(b)
                    break
                p = int(m.body_parentid[p])
        self.cuerpos_pinza = descendientes
        juntas, sin_limite = [], []
        for j in range(m.njnt):
            if m.jnt_type[j] not in (HINGE, int(mujoco.mjtJoint.mjJNT_SLIDE)):
                continue
            n = nombre(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if any(x in n for x in PINZA_EXCLUIR):
                continue
            if j in self.juntas:      # ya la mueve la IK del brazo
                continue
            propia = int(m.jnt_bodyid[j]) in descendientes or (
                f"_{self.lado}_" in n and "grasp" in n.lower())
            if propia and (m.jnt_limited[j] or n in PINZA_RECORRIDO):
                juntas.append(j)
            elif propia:
                sin_limite.append(n)
        self.j_pinza = juntas
        self.n_pinza = [nombre(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in juntas]
        self.qadr_pinza = np.array([m.jnt_qposadr[j] for j in juntas], dtype=int)
        cerrado, abierto = [], []
        for j, n in zip(juntas, self.n_pinza):
            lo, hi = PINZA_RECORRIDO.get(n, tuple(m.jnt_range[j]))
            c, a = (lo, hi) if abs(lo) <= abs(hi) else (hi, lo)
            if PINZA_INVERTIDA != (n in PINZA_JUNTAS_INVERTIDAS):
                c, a = a, c
            cerrado.append(c)
            abierto.append(a)
        self.q_cerrado = np.array(cerrado, dtype=float)
        self.q_abierto = np.array(abierto, dtype=float)
        self.apertura = 1.0 if juntas else None

    def mover_pinza(self, apertura, dt):
        """Lleva la pinza hacia la apertura pedida (0 = cerrada, 1 = abierta)
        con velocidad limitada, para que no de tirones."""
        if not self.j_pinza:
            return
        objetivo = float(np.clip(apertura, 0.0, 1.0))
        paso = VEL_PINZA * dt
        self.apertura += float(np.clip(objetivo - self.apertura, -paso, paso))
        self.d.qpos[self.qadr_pinza] = (self.q_cerrado
                                        + self.apertura * (self.q_abierto - self.q_cerrado))

    # ---------------- utilidades ----------------
    def _a_local(self, b, p):
        return self.d.xmat[b].reshape(3, 3).T @ (p - self.d.xpos[b])

    def _a_mundo(self, b, loc):
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ loc

    def puntos(self):
        """Posicion actual (mundo) del codo y del centro de la muneca."""
        return self._a_mundo(self.b_codo, self.loc_codo), self._a_mundo(self.b_muneca, self.loc_muneca)

    def reposo(self, q=None):
        """Lleva el brazo a la postura visual de reposo (o a 'q' si se pasa)."""
        self.d.qpos[self.qadr] = self.q_reposo_visual if q is None else q
        if getattr(self, "mano", None) is not None:
            self.mano.reposo()
        self.R_obj_ult = None
        if hasattr(self, "_dH_ant"):
            self._dH_ant = np.zeros(len(self.juntas))
        if hasattr(self, "atasco"):
            self.atasco.reiniciar()
        if self.j_pinza:
            self.apertura = 1.0
            self.d.qpos[self.qadr_pinza] = self.q_abierto

    def _limitar(self, q):
        q[self.limitado] = np.clip(q[self.limitado], self.lo[self.limitado], self.hi[self.limitado])
        return q

    # ---------------- punto de agarre ----------------
    def _preparar_agarre(self):
        m = self.m

        def geoms_de(cuerpos):
            return [g for b in cuerpos for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])
                    if m.body_geomnum[b] > 0]
        cuerpos = getattr(self, "cuerpos_pinza", set())
        mordazas = [b for b in cuerpos if "jaw" in nombre(m, mujoco.mjtObj.mjOBJ_BODY, b)]
        self.geoms_agarre = geoms_de(mordazas) or geoms_de(cuerpos)
        self.cuerpos_mordaza = sorted(mordazas)

    def punto_agarre(self):
        """Punto (mundo) entre las mordazas de la pinza (o en la palma de la OrcaHand)."""
        if self.mano is not None:
            return self.mano.punto_agarre()
        if self.geoms_agarre:
            return self.d.geom_xpos[self.geoms_agarre].mean(axis=0)
        return self.d.xpos[self.ef].copy()

    # ---------------- orientacion de la palma ----------------
    def _marco_pinza(self):
        """Marco semantico de la pinza original, en coordenadas del efector final:
        dedos = del centro de la muneca hacia las mordazas; normal de la palma = eje de
        cierre de las mordazas, con el signo que la deje mirando a PALMA_PINZA_REPOSO
        (hacia atras) en el reposo, que es como cuelga la mano de una persona."""
        d = self.d
        _, muneca = self.puntos()
        if self.geoms_agarre:
            f = d.geom_xpos[self.geoms_agarre].mean(axis=0) - muneca
        else:
            f = d.xpos[self.ef] - muneca
        if np.linalg.norm(f) < 1e-6:
            f = -d.xmat[self.ef].reshape(3, 3)[:, 2]
        f = f / np.linalg.norm(f)
        if len(getattr(self, "cuerpos_mordaza", [])) >= 2:
            n = d.xpos[self.cuerpos_mordaza[0]] - d.xpos[self.cuerpos_mordaza[1]]
        else:   # sin mordazas: cualquier eje perpendicular a los dedos
            n = np.cross(f, [0.0, 0.0, 1.0])
            if np.linalg.norm(n) < 1e-3:
                n = np.cross(f, [1.0, 0.0, 0.0])
        n = n - f * np.dot(n, f)
        n /= np.linalg.norm(n)
        if np.dot(n, NORMAL_PINZA_REPOSO) < 0:
            n = -n
        a = np.cross(n, f)                      # ancho de la palma tal que f x a = n
        R_mundo = marco_semantico(f, a)
        return d.xmat[self.ef].reshape(3, 3).T @ R_mundo

    def preparar_orientacion(self, b_orient, marco_local):
        """Fija el cuerpo que lleva la orientacion y su marco semantico (lo llama
        RobotTron2 cuando monta las OrcaHand)."""
        self.b_orient = int(b_orient)
        self.marco_sem_local = np.asarray(marco_local, dtype=float)

    def orientacion(self):
        """Marco semantico de la palma del robot en el mundo (columnas: dedos, ancho, normal)."""
        return self.d.xmat[self.b_orient].reshape(3, 3) @ self.marco_sem_local

    def eje_giro(self):
        """Eje (mundo) de la junta de giro de la muneca, a lo largo del antebrazo."""
        a = self.d.xaxis[self.j_giro]
        return a / np.linalg.norm(a)

    def giro(self):
        return float(self.d.qpos[self.qadr[self.i_giro]])

    # ---------------- IK ----------------
    def _fijar(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)  # necesario para mj_jac

    def error(self, obj_c, obj_m, w_codo, R_obj=None):
        """(error ponderado, error codo [m], error muneca [m], error de orientacion [rad])."""
        p_c, p_m = self.puntos()
        ec, em = np.linalg.norm(obj_c - p_c), np.linalg.norm(obj_m - p_m)
        eo = 0.0 if R_obj is None else float(np.linalg.norm(log_rot(R_obj @ self.orientacion().T)))
        total2 = (w_codo * ec) ** 2 + em ** 2 + (W_ORIENT * eo) ** 2
        if self.anat is not None:   # pasarse del limite anatomico tambien es error
            total2 += float(np.sum((W_ANAT * self._castigo_anat()) ** 2))
        return float(np.sqrt(total2)), float(ec), float(em), eo

    def _paso(self, obj_c, obj_m, R_obj, w_codo, lam, k_reposo, k_limite):
        """Un paso de IK que respeta los limites: si una articulacion se saldria de su
        rango, se bloquea y se recalcula el paso para que las demas compensen."""
        m, d = self.m, self.d
        p_c, p_m = self.puntos()
        mujoco.mj_jac(m, d, self._jac, None, p_c, self.b_codo)
        J_c = self._jac[:, self.dofs].copy()
        mujoco.mj_jac(m, d, self._jac, None, p_m, self.b_muneca)
        J_m = self._jac[:, self.dofs].copy()
        filas_J = [w_codo * J_c, J_m]
        filas_e = [w_codo * (obj_c - p_c), obj_m - p_m]
        if R_obj is not None:   # tarea de orientacion de la palma (solo las juntas de la muneca)
            mujoco.mj_jac(m, d, None, self._jacr, p_m, self.b_orient)
            filas_J.append(W_ORIENT * self._jacr[:, self.dofs] * self.mascara_muneca)
            filas_e.append(W_ORIENT * log_rot(R_obj @ self.orientacion().T))
        q = d.qpos[self.qadr].copy()
        n = len(q)
        if self.anat is not None:   # (1) limite anatomico: tarea que solo actua al pasarse
            c = self._castigo_anat()
            activas = c > 0
            if activas.any():
                filas_J.append(W_ANAT * self._jac_anat(q, c)[activas])
                filas_e.append(-W_ANAT * c[activas])
        J = np.vstack(filas_J)
        e = np.concatenate(filas_e)

        # Tarea secundaria (espacio nulo): volver a reposo y alejarse de los limites
        d_lo, d_hi = q - self.lo, self.hi - q
        empuje = (np.where(self.limitado & (d_lo < MARGEN_LIMITE), 1 - d_lo / MARGEN_LIMITE, 0.0)
                  - np.where(self.limitado & (d_hi < MARGEN_LIMITE), 1 - d_hi / MARGEN_LIMITE, 0.0))
        g = k_reposo * (self.q_reposo - q) + k_limite * empuje
        # (3) sin objetivo de codo (brazo casi recto) el giro del codo lo decide el espacio nulo:
        # se prefiere el codo hacia abajo y algo hacia fuera, como cuelga el de una persona
        x = float(np.clip(w_codo / W_CODO_PLENO, 0.0, 1.0))
        libertad = 1.0 - x * x * (3.0 - 2.0 * x)
        if K_CODO_ABAJO > 0 and libertad > 0:
            R = d.xmat[self.raiz].reshape(3, 3)
            fuera = R[:, 1] if self.lado == "L" else -R[:, 1]
            preferida = -R[:, 2] + CODO_FUERA * fuera
            g = g + K_CODO_ABAJO * libertad * (J_c.T @ (preferida / np.linalg.norm(preferida)))

        # (2) minima norma ponderada: las juntas que se acercan a su limite cuestan mas de mover
        pesos, self._dH_ant = pesos_limites(q, self.lo, self.hi, self.limitado, self._dH_ant)
        w_inv = 1.0 / pesos
        libres = np.ones(n, dtype=bool)
        for _ in range(3):
            Jl = J * libres  # columnas de las articulaciones bloqueadas a cero
            # J# = W^-1 J^T (J W^-1 J^T + lam^2 I)^-1
            A = (Jl * w_inv) @ Jl.T + lam ** 2 * np.eye(J.shape[0])
            J_pinv = w_inv[:, None] * np.linalg.solve(A, Jl).T
            dq = J_pinv @ e + (np.eye(n) - J_pinv @ Jl) @ g
            dq[~libres] = 0.0
            fuera = self.limitado & libres & (((q + dq) < self.lo) | ((q + dq) > self.hi))
            if not fuera.any():
                break
            libres &= ~fuera
        return q, dq

    def resolver(self, obj_c, obj_m, R_obj=None, w_codo=0.6, iteraciones=4,
                 max_cambio=np.inf, lam=0.05, k_reposo=0.01, k_limite=0.02, paso_max=0.2):
        """IK por minimos cuadrados amortiguados con busqueda lineal: solo se aceptan
        pasos que no empeoran el error, asi el brazo nunca 'da bandazos'.
        max_cambio: giro maximo de cada articulacion en esta llamada (rad).
        Devuelve (error codo [m], error muneca [m], error de orientacion [rad])."""
        q0 = self.d.qpos[self.qadr].copy()
        err = self.error(obj_c, obj_m, w_codo, R_obj)[0]
        for _ in range(iteraciones):
            q, dq = self._paso(obj_c, obj_m, R_obj, w_codo, lam, k_reposo, k_limite)
            mayor = np.abs(dq).max()
            if mayor > paso_max:
                dq *= paso_max / mayor
            alfa, aceptado = 1.0, False
            for _ in range(4):
                q_n = self._limitar(q + alfa * dq)
                delta = unwrap_angular_delta(q_n - q0)
                q_n = q0 + np.clip(delta, -max_cambio, max_cambio)
                self._fijar(q_n)
                err_n = self.error(obj_c, obj_m, w_codo, R_obj)[0]
                if err_n <= err + 1e-4:
                    err, aceptado = err_n, True
                    break
                alfa *= 0.5
            if not aceptado:
                self._fijar(q)
                break
        _, ec, em, eo = self.error(obj_c, obj_m, w_codo, R_obj)
        return ec, em, eo

    def seguir(self, obj_c, obj_m, dt, w_codo=0.6, R_obj=None):
        """Llamar en cada ciclo del bucle de control. IK con limite de velocidad y,
        si el brazo se queda atascado lejos del objetivo (minimo local, limites),
        busca otra solucion y va hacia ella de forma suave.
        R_obj: orientacion objetivo de la palma (marco semantico, mundo) o None. Sin ella
        se mantiene la ultima recibida (la mano no se ve un momento -> no se mueve)."""
        if R_obj is not None:
            self.R_obj_ult = np.asarray(R_obj, dtype=float)
        R_obj = self.R_obj_ult
        ahora = self.reloj()
        if self.q_transicion is not None:
            q = self.d.qpos[self.qadr].copy()
            falta = unwrap_angular_delta(self.q_transicion - q)
            caducada = ahora - self.t_rescate > T_TRANSICION_MAX
            movido = (self.obj_rescate is not None
                      and np.linalg.norm(obj_m - self.obj_rescate) > CANCELA_TRANSICION)
            if np.abs(falta).max() < 0.02 or caducada or movido:
                self.q_transicion = None
            else:
                self._fijar(q + np.clip(falta, -VEL_TRANSICION * dt, VEL_TRANSICION * dt))
                _, ec, em, eo = self.error(obj_c, obj_m, w_codo, R_obj)
                return ec, em, eo

        ec, em, eo = self.resolver(obj_c, obj_m, R_obj, w_codo, max_cambio=VEL_MAX * dt)
        err = float(np.hypot(w_codo * ec, em))   # solo posicion
        self.err_ant = err
        atascado = self.atasco.actualizar(ahora, err)
        self.t_atascado = self.t_atascado + dt if atascado else 0.0     # (diagnostico)
        if atascado and ahora - self.t_rescate > 1.0:
            self.t_rescate = ahora
            self.atasco.reiniciar()
            q_mejor, err_mejor = self._rescate(obj_c, obj_m, R_obj, w_codo)
            # se compara con la MISMA medida (posicion + limite anatomico) en la postura actual
            err = self.error(obj_c, obj_m, w_codo, None)[0]
            if err_mejor < 0.5 * err or err - err_mejor > MEJORA_RESCATE:
                self.q_transicion = q_mejor
                self.obj_rescate = np.array(obj_m, dtype=float)
                print(f"[IK {self.lado}] RESCATE: el brazo cambia de postura "
                      f"(error {err * 100:.0f} cm -> {err_mejor * 100:.0f} cm)")
        return ec, em, eo

    def _semilla_rejilla(self, obj_c, obj_m, n_hombro=(12, 8), n_codo=(16, 12)):
        """Postura de partida por busqueda en rejilla (no se queda en minimos locales):
        1) las dos primeras juntas del hombro apuntan el brazo hacia el codo objetivo;
        2) la ultima del hombro (giro humeral) y el codo colocan la muneca.
        Son ~300 evaluaciones de la cinematica (unos ms); se usa en los rescates."""
        q0 = self.d.qpos[self.qadr].copy()
        n = len(q0)
        i_c = self.juntas.index(self.j_codo)
        rangos = [(self.lo[i], self.hi[i]) if self.limitado[i] else (-np.pi, np.pi) for i in range(n)]
        q = self.q_reposo.copy()
        q[self.i_muneca] = q0[self.i_muneca]
        if len(self.j_hombro) >= 2:
            mejor, mejor_e = q.copy(), np.inf
            for a in np.linspace(*rangos[0], n_hombro[0]):
                for b in np.linspace(*rangos[1], n_hombro[1]):
                    q[0], q[1] = a, b
                    self._fijar(q)
                    e = np.linalg.norm(self.puntos()[0] - obj_c) + self._exceso_anat()
                    if e < mejor_e:
                        mejor, mejor_e = q.copy(), e
            q = mejor
        i_g = len(self.j_hombro) - 1          # ultima junta del hombro: giro humeral
        mejor, mejor_e = q.copy(), np.inf
        for g in np.linspace(*rangos[i_g], n_codo[0]) if i_g >= 2 else [q[i_g]]:
            for c in np.linspace(*rangos[i_c], n_codo[1]):
                q[i_g], q[i_c] = g, c
                self._fijar(q)
                p_c, p_m = self.puntos()
                e = np.hypot(0.6 * np.linalg.norm(p_c - obj_c), np.linalg.norm(p_m - obj_m)) + self._exceso_anat()
                if e < mejor_e:
                    mejor, mejor_e = q.copy(), e
        self._fijar(q0)
        return mejor

    def _exceso_anat(self):
        """Castigo anatomico en metros equivalentes (0 sin limites anatomicos): las semillas de
        los rescates tampoco deben ser posturas que una persona no puede hacer."""
        return 0.0 if self.anat is None else float(W_ANAT * np.linalg.norm(self._castigo_anat()))

    def _rescate(self, obj_c, obj_m, R_obj, w_codo):
        """Prueba varias posturas de partida y devuelve la que mejor alcanza el objetivo."""
        q_act = self.d.qpos[self.qadr].copy()
        dH_act = self._dH_ant.copy()          # los pasos de prueba no deben tocar los pesos
        semillas = [self.q_reposo.copy(), self._semilla_rejilla(obj_c, obj_m)]
        for i in range(len(self.j_hombro)):
            for delta in (-1.2, 1.2):
                s = q_act.copy()
                s[i] += delta
                semillas.append(s)
        mejor_q, mejor_err = q_act, np.inf
        for s in semillas:
            self._fijar(self._limitar(s))
            self.resolver(obj_c, obj_m, R_obj, w_codo, iteraciones=25)
            err = self.error(obj_c, obj_m, w_codo, None)[0]     # posicion (+ limite anatomico)
            if err < mejor_err:
                mejor_q, mejor_err = self.d.qpos[self.qadr].copy(), err
        self._fijar(q_act)
        self._dH_ant = dH_act
        return mejor_q, mejor_err


class RobotTron2:
    def __init__(self, ruta_xml, escenario=None, orca=False, reloj=time.monotonic):
        """orca=True: quita las pinzas y monta una OrcaHand en cada muneca (mano_orca.py).
        reloj: funcion que da el tiempo (time.monotonic, o un reloj simulado al reproducir)."""
        self.origen = ruta_xml
        self.reloj = reloj
        if SOPORTE_KUKA:
            import montar_kuka
            escenario = montar_kuka.con_kuka(escenario)
        if orca:
            import mano_orca

            def extra(root):
                if escenario is not None:
                    escenario(root)
                mano_orca.quitar_pinzas(root)

            self.ruta = quitar_base_flotante(ruta_xml, extra)
            self.m = mano_orca.montar_manos(self.ruta)
        else:
            self.ruta = quitar_base_flotante(ruta_xml, escenario)
            self.m = mujoco.MjModel.from_xml_path(self.ruta)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        self.brazos = {lado: Brazo(self.m, self.d, lado, reloj=reloj) for lado in ("L", "R")}
        self.manos = {}
        if orca:
            for lado, brazo in self.brazos.items():
                brazo.mano = self.manos[lado] = mano_orca.ManoOrca(self.m, self.d, lado, reloj=reloj)
                brazo.preparar_orientacion(brazo.mano.palma, brazo.mano.marco_sem_local)
        if BLOQUEAR_FLEXION_MUNECA or BLOQUEAR_DESVIACION_MUNECA:
            for brazo in self.brazos.values():
                brazo.bloquear_muneca(BLOQUEAR_FLEXION_MUNECA, BLOQUEAR_DESVIACION_MUNECA)
        self.actualizar()
        # Orientacion del torso en el mundo (x delante, y izquierda, z arriba)
        self.R_base = self.d.xmat[self.brazos["L"].raiz].reshape(3, 3).copy()

    def actualizar(self):
        """Recalcula posiciones para dibujar y para la IK (sin mj_forward: colisiones,
        restricciones y sensores no hacen falta y cuestan cientos de ms)."""
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def reposo(self):
        for b in self.brazos.values():
            b.reposo()
            b.q_transicion, b.t_atascado, b.err_ant = None, 0.0, 0.0
        self.actualizar()

    def resumen(self):
        print(f"\nModelo: {self.origen}  (nq={self.m.nq}, nu={self.m.nu})")
        for lado, b in self.brazos.items():
            print(f"\nBrazo {lado}  ->  efector final: {nombre(self.m, mujoco.mjtObj.mjOBJ_BODY, b.ef)}")
            for j, n in zip(b.juntas, b.nombres):
                papel = "hombro" if j in b.j_hombro else ("codo" if j == b.j_codo else "muneca")
                lo, hi = self.m.jnt_range[j]
                print(f"   {n:30s} {papel:7s} [{lo:+.2f}, {hi:+.2f}] rad")
            print(f"   brazo {b.L_brazo * 100:.1f} cm | antebrazo {b.L_antebrazo * 100:.1f} cm | "
                  f"centro del hombro {np.round(b.hombro, 3)}")
            R = b.orientacion()
            print(f"   palma en reposo: dedos {np.round(R[:, 0], 2)} | normal {np.round(R[:, 2], 2)} "
                  f"(cuerpo {nombre(self.m, mujoco.mjtObj.mjOBJ_BODY, b.b_orient)})")
            if b.j_pinza:
                for n, c, a in zip(b.n_pinza, b.q_cerrado, b.q_abierto):
                    print(f"   pinza: {n:28s} cerrada {c:+.2f} -> abierta {a:+.2f}")
            else:
                print("   pinza: este modelo no tiene articulaciones de pinza")
        print()