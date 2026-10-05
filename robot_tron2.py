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
"""
import os
import time
import xml.etree.ElementTree as ET

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
T_ATASCO = 1.0         # s: tiempo atascado antes de buscar otra solucion
MEJORA_MIN = 0.05      # m/s: si el error baja mas rapido que esto, el brazo NO esta atascado
T_TRANSICION_MAX = 2.0 # s: una transicion de rescate nunca dura mas que esto
CANCELA_TRANSICION = 0.10  # m: si el objetivo se mueve esto durante la transicion, se cancela
# Pinza original (simetrica): su "normal de la palma" es el eje de cierre de las mordazas, y
# el signo se elige para que con los brazos colgando quede como la mano de una persona
# relajada: palma hacia el muslo (hacia +y en la derecha). Para la izquierda el tercer eje
# del marco semantico es el DORSO, que tambien apunta a +y. Asi el reposo de la pinza
# coincide con el del operador y la muneca no arranca pegada a un limite.
NORMAL_PINZA_REPOSO = np.array([0.0, 1.0, 0.0])


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

        # --- Geometria del brazo, medida en la postura de reposo ---
        self.reposo()
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
        self.rng = np.random.default_rng(0)

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

    def reposo(self):
        self.d.qpos[self.qadr] = self.q_reposo
        if getattr(self, "mano", None) is not None:
            self.mano.reposo()
        self.R_obj_ult = None
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
        total = np.sqrt((w_codo * ec) ** 2 + em ** 2 + (W_ORIENT * eo) ** 2)
        return float(total), float(ec), float(em), eo

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
        J = np.vstack(filas_J)
        e = np.concatenate(filas_e)

        q = d.qpos[self.qadr].copy()
        n = len(q)
        # Tarea secundaria (espacio nulo): volver a reposo y alejarse de los limites
        d_lo, d_hi = q - self.lo, self.hi - q
        empuje = (np.where(self.limitado & (d_lo < MARGEN_LIMITE), 1 - d_lo / MARGEN_LIMITE, 0.0)
                  - np.where(self.limitado & (d_hi < MARGEN_LIMITE), 1 - d_hi / MARGEN_LIMITE, 0.0))
        g = k_reposo * (self.q_reposo - q) + k_limite * empuje

        libres = np.ones(n, dtype=bool)
        for _ in range(3):
            Jl = J * libres  # columnas de las articulaciones bloqueadas a cero
            J_pinv = np.linalg.solve(Jl @ Jl.T + lam ** 2 * np.eye(J.shape[0]), Jl).T
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
        mejora = self.err_ant - err
        self.err_ant = err
        atascado = err > ERROR_ATASCO and mejora < MEJORA_MIN * dt
        self.t_atascado = self.t_atascado + dt if atascado else 0.0
        if self.t_atascado > T_ATASCO and ahora - self.t_rescate > 1.0:
            self.t_rescate, self.t_atascado = ahora, 0.0
            q_mejor, err_mejor = self._rescate(obj_c, obj_m, R_obj, w_codo)
            if err_mejor < 0.5 * err:
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
                    e = np.linalg.norm(self.puntos()[0] - obj_c)
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
                e = np.hypot(0.6 * np.linalg.norm(p_c - obj_c), np.linalg.norm(p_m - obj_m))
                if e < mejor_e:
                    mejor, mejor_e = q.copy(), e
        self._fijar(q0)
        return mejor

    def _rescate(self, obj_c, obj_m, R_obj, w_codo):
        """Prueba varias posturas de partida y devuelve la que mejor alcanza el objetivo."""
        q_act = self.d.qpos[self.qadr].copy()
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
            err = self.error(obj_c, obj_m, w_codo, None)[0]     # se compara solo la posicion
            if err < mejor_err:
                mejor_q, mejor_err = self.d.qpos[self.qadr].copy(), err
        self._fijar(q_act)
        return mejor_q, mejor_err


class RobotTron2:
    def __init__(self, ruta_xml, escenario=None, orca=False, reloj=time.monotonic):
        """orca=True: quita las pinzas y monta una OrcaHand en cada muneca (mano_orca.py).
        reloj: funcion que da el tiempo (time.monotonic, o un reloj simulado al reproducir)."""
        self.origen = ruta_xml
        self.reloj = reloj
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
