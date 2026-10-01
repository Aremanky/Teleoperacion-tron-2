"""
TRON 2 (DACH_TRON2A) en MuJoCo: carga con la base fija + IK de los dos brazos.

La IK no depende del orden ni de los nombres exactos de las articulaciones:
recorre la cadena desde el cuerpo final de cada brazo (grasper_L_Link /
grasper_R_Link), se queda con las bisagras de ese lado y separa
hombro / codo / muneca buscando "elbow" en el nombre. Al arrancar imprime
lo que ha detectado para poder revisarlo.

Tareas de la IK (minimos cuadrados amortiguados, en modo cinematico):
  - llevar el CODO del robot a su objetivo
  - llevar el CENTRO DE LA MUNECA del robot a su objetivo
  - en el espacio nulo: volver a reposo y alejarse de los limites articulares
Ademas: respeta los limites (bloquea la articulacion y compensan las demas),
solo acepta pasos que reducen el error, limita la velocidad articular y,
si el brazo se atasca en un minimo local, busca otra solucion.
"""
import os
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)


def log_rot(R):
    """Vector rotacion (eje * angulo) de una matriz de rotacion."""
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.arccos(c))
    sen = np.sin(ang)
    if sen < 1e-6:
        return 0.5 * v if ang < 1e-3 else ang * v / max(np.linalg.norm(v), 1e-9)
    return (ang / (2.0 * sen)) * v


def angulo_giro(R, eje):
    """Parte de la rotacion R que es giro alrededor de 'eje' (unitario), en rad.
    Descomposicion swing-twist: se queda solo con el 'retorcer' alrededor del eje y
    descarta cualquier inclinacion, asi doblar la mano no cuenta como girarla."""
    w = np.sqrt(max(0.0, 1.0 + np.trace(R))) / 2.0
    if w > 1e-6:
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (4.0 * w)
    else:   # giro de ~180 grados: eje desde la diagonal
        v = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
    ang = 2.0 * np.arctan2(float(np.dot(v, eje)), w)
    return float((ang + np.pi) % (2 * np.pi) - np.pi)


def ortonormalizar(R):
    """Gram-Schmidt: devuelve la rotacion valida mas cercana a R."""
    x = R[:, 0] / np.linalg.norm(R[:, 0])
    y = R[:, 1] - x * np.dot(R[:, 1], x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])

# Cuerpo final de cada brazo, por orden de preferencia ({lado} = L o R)
CANDIDATOS_EF = ["grasper_{lado}_Link", "wrist_roll_{lado}_Link"]

# Distancia maxima (m) del centro estimado a cada eje para considerar
# que el hombro / la muneca son esfericos (los tres ejes se cortan)
TOL_ESFERICO = 0.03

VEL_MAX = 3.0          # rad/s: velocidad maxima de cada articulacion al seguir al operador
W_ORIENT = 0.08        # m/rad: peso de la orientacion de la pinza frente a la posicion
VEL_PINZA = 2.5        # 1/s: velocidad de apertura y cierre de la pinza (0 a 1 en 0.4 s)
PINZA_INVERTIDA = False  # True si toda la pinza abre cuando deberia cerrar
# Juntas que llevan "grasp" en el nombre pero NO forman parte del mecanismo de la
# pinza (p.ej. la que orienta la pinza entera). Se dejan quietas.
PINZA_EXCLUIR = ("grasper_base", "orca")   # "orca": juntas de las OrcaHand (las mueve mano_orca.py)
# Juntas sueltas del mecanismo cuyo sentido hay que invertir (por su nombre exacto)
PINZA_JUNTAS_INVERTIDAS = ()
# Recorrido a mano para juntas de la pinza SIN limites en el XML (no se puede
# adivinar): {"grasper_L_jaw_left_Joint": (cerrado, abierto), ...}
PINZA_RECORRIDO = {}
VEL_TRANSICION = 1.5   # rad/s: al cambiar a otra solucion tras un rescate
MARGEN_LIMITE = 0.1    # rad: junto a un limite se empuja la articulacion hacia dentro
ERROR_ATASCO = 0.08    # m: error a partir del cual se considera que el brazo esta atascado
T_ATASCO = 0.5         # s: tiempo atascado antes de buscar otra solucion
MEJORA_MIN = 0.05      # m/s: si el error baja mas rapido que esto, el brazo NO esta atascado
                       #      (solo va alcanzando un objetivo que ha saltado): no hay rescate
T_TRANSICION_MAX = 2.0 # s: una transicion de rescate nunca dura mas que esto
CANCELA_TRANSICION = 0.10  # m: si el objetivo se mueve esto durante la transicion, se cancela


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
    # Se guarda junto al original para que las rutas de mallas/includes sigan valiendo
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


class Brazo:
    def __init__(self, m, d, lado):
        self.m, self.d, self.lado = m, d, lado
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
        # columnas de la muneca: las unicas que atienden a la orientacion de la pinza
        self.mascara_muneca = np.array([j in self.j_muneca for j in self.juntas], dtype=float)
        # junta de GIRO de la muneca: la de eje mas alineado con el antebrazo
        # (en el TRON 2, wrist_yaw). Las demas de la muneca se mantienen rectas.
        antebrazo = (muneca - codo) / max(np.linalg.norm(muneca - codo), 1e-9)
        self.j_giro = max(self.j_muneca, key=lambda j: abs(np.dot(d.xaxis[j], antebrazo)))
        self.i_giro = self.juntas.index(self.j_giro)
        self.i_rigidas = [self.juntas.index(j) for j in self.j_muneca if j != self.j_giro]
        self.i_muneca = [self.juntas.index(j) for j in self.j_muneca]
        self.err_ant, self.obj_rescate = 0.0, None
        self.giro_obj = None
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
        cero se toma como pinza cerrada y el otro como abierta; asi vale igual
        para las dos mordazas aunque sus rangos tengan signo opuesto.

        En las pinzas de varillas (cadena cerrada con <equality>) se mueven todas
        las juntas del mecanismo a la vez, en la misma fraccion de su recorrido.
        Aqui no se simula fisica, asi que las restricciones de igualdad no se
        resuelven solas; este reparto proporcional es exacto en los extremos
        (abierta del todo y cerrada del todo) y muy aproximado por el camino."""
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
                sin_limite.append(n)   # no se puede mapear: no sabemos su recorrido
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
        if not juntas:
            self._diagnostico_pinza(sin_limite)

    def _diagnostico_pinza(self, sin_limite):
        """Si no se ha encontrado ninguna junta de pinza, explica por que: lista
        todas las que llevan 'grasp' en el nombre con su tipo, limites y rango."""
        m = self.m
        tipos = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
        print(f"\n[pinza {self.lado}] no he podido mapear ninguna articulacion. "
              f"Juntas con 'grasp' en el nombre que hay en el modelo:")
        hay = False
        for j in range(m.njnt):
            n = nombre(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if "grasp" not in n.lower():
                continue
            hay = True
            cuerpo = nombre(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j]))
            motivo = ""
            if any(x in n for x in PINZA_EXCLUIR):
                motivo = "  <- excluida por PINZA_EXCLUIR"
            elif n in sin_limite:
                motivo = "  <- SIN limites en el XML: anade su recorrido a PINZA_RECORRIDO"
            print(f"   {n:32s} cuerpo {cuerpo:28s} {tipos.get(int(m.jnt_type[j]), '?'):5s} "
                  f"limited={bool(m.jnt_limited[j])} range=[{m.jnt_range[j][0]:+.5f}, "
                  f"{m.jnt_range[j][1]:+.5f}]{motivo}")
        if not hay:
            print("   ninguna: este modelo no tiene pinza articulada")
        print()

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
        if hasattr(self, "giro_obj"):
            self.giro_obj = None
        if self.j_pinza:
            self.apertura = 1.0
            self.d.qpos[self.qadr_pinza] = self.q_abierto

    def _limitar(self, q):
        q[self.limitado] = np.clip(q[self.limitado], self.lo[self.limitado], self.hi[self.limitado])
        return q

    # ---------------- punto de agarre ----------------
    def _preparar_agarre(self):
        """Geoms de las mordazas: el punto de agarre es el centro entre ellas.
        (MuJoCo centra cada malla en su centro de masas, asi que la posicion de
        cada geom de mordaza cae en mitad del dedo.) Si el modelo no tiene
        mordazas articuladas se usan los geoms de la pinza entera."""
        m = self.m
        def geoms_de(cuerpos):
            return [g for b in cuerpos for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])
                    if m.body_geomnum[b] > 0]
        cuerpos = getattr(self, "cuerpos_pinza", set())
        mordazas = [b for b in cuerpos if "jaw" in nombre(m, mujoco.mjtObj.mjOBJ_BODY, b)]
        self.geoms_agarre = geoms_de(mordazas) or geoms_de(cuerpos)

    def punto_agarre(self):
        """Punto (mundo) entre las mordazas de la pinza (o en la palma de la OrcaHand)."""
        if self.mano is not None:
            return self.mano.punto_agarre()
        if self.geoms_agarre:
            return self.d.geom_xpos[self.geoms_agarre].mean(axis=0)
        return self.d.xpos[self.ef].copy()

    # ---------------- giro de la muneca ----------------
    def eje_giro(self):
        """Eje (mundo) de la junta de giro de la muneca, a lo largo del antebrazo."""
        a = self.d.xaxis[self.j_giro]
        return a / np.linalg.norm(a)

    def giro(self):
        return float(self.d.qpos[self.qadr[self.i_giro]])

    def mover_muneca(self, giro_obj, dt):
        """Muneca solo con giro: la junta de giro va hacia 'giro_obj' (rad) y las
        demas juntas de la muneca se quedan rectas, en reposo. Como la muneca es
        esferica, esto no mueve la posicion del brazo. Con giro_obj=None mantiene
        el ultimo objetivo recibido."""
        if giro_obj is not None:
            self.giro_obj = float(np.clip(giro_obj, self.lo[self.i_giro], self.hi[self.i_giro])
                                  if self.limitado[self.i_giro] else giro_obj)
        q = self.d.qpos[self.qadr].copy()
        paso = VEL_MAX * dt
        if self.giro_obj is not None:
            q[self.i_giro] += np.clip(self.giro_obj - q[self.i_giro], -paso, paso)
        for k in self.i_rigidas:
            q[k] += np.clip(self.q_reposo[k] - q[k], -paso, paso)
        self._fijar(self._limitar(q))

    # ---------------- IK ----------------
    def _fijar(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)  # necesario para mj_jac

    def orientacion(self):
        """Orientacion actual de la pinza (marco del efector final) en el mundo."""
        if self.mano is not None:
            return self.mano.orientacion()
        return self.d.xmat[self.ef].reshape(3, 3)

    def error(self, obj_c, obj_m, w_codo, R_obj=None):
        """(error ponderado, error codo [m], error muneca [m], error giro [rad])."""
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
        J_c = self._jac[:, self.dofs]
        mujoco.mj_jac(m, d, self._jac, None, p_m, self.b_muneca)
        J_m = self._jac[:, self.dofs]
        filas_J = [w_codo * J_c, J_m]
        filas_e = [w_codo * (obj_c - p_c), obj_m - p_m]
        if R_obj is not None:   # tarea de orientacion de la pinza
            mujoco.mj_jac(m, d, None, self._jacr, p_m, self.ef)
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
        Devuelve el error final (m) del codo y de la muneca."""
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
                q_n = q0 + np.clip(q_n - q0, -max_cambio, max_cambio)
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
        busca otra solucion y va hacia ella de forma suave."""
        ahora = time.monotonic()
        if self.q_transicion is not None:
            q = self.d.qpos[self.qadr].copy()
            falta = self.q_transicion - q
            # La muneca NO entra en la transicion: la controla mover_muneca() en cada ciclo.
            # (Antes si entraba: los dos tiraban de ella hacia valores distintos, 'falta'
            # nunca bajaba de 0.02 y el brazo se quedaba congelado para siempre.)
            falta[self.i_muneca] = 0.0
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
        # Atascado = error grande Y que no baja. Si baja, el brazo solo va alcanzando un
        # objetivo que ha saltado y lo mejor es dejarle seguir (antes se rescataba igual).
        mejora = self.err_ant - err
        self.err_ant = err
        atascado = err > ERROR_ATASCO and mejora < MEJORA_MIN * dt
        self.t_atascado = self.t_atascado + dt if atascado else 0.0
        if self.t_atascado > T_ATASCO and ahora - self.t_rescate > 1.0:
            self.t_rescate, self.t_atascado = ahora, 0.0
            q_mejor, err_mejor = self._rescate(obj_c, obj_m, None, w_codo)
            if err_mejor < 0.5 * err:
                self.q_transicion = q_mejor
                self.obj_rescate = np.array(obj_m, dtype=float)
        return ec, em, eo

    def _rescate(self, obj_c, obj_m, R_obj, w_codo):
        """Prueba varias posturas de partida y devuelve la que mejor alcanza el objetivo."""
        q_act = self.d.qpos[self.qadr].copy()
        semillas = [self.q_reposo.copy()]
        for i in range(len(self.j_hombro)):
            for delta in (-1.2, 1.2):
                s = q_act.copy()
                s[i] += delta
                semillas.append(s)
        lo = np.where(self.limitado, self.lo, -np.pi)
        hi = np.where(self.limitado, self.hi, np.pi)
        semillas += [self.rng.uniform(lo, hi) for _ in range(3)]

        mejor_q, mejor_err = q_act, np.inf
        for s in semillas:
            self._fijar(self._limitar(s))
            self.resolver(obj_c, obj_m, R_obj, w_codo, iteraciones=25)
            err = self.error(obj_c, obj_m, w_codo, R_obj)[0]
            if err < mejor_err:
                mejor_q, mejor_err = self.d.qpos[self.qadr].copy(), err
        self._fijar(q_act)
        mejor_q[self.i_muneca] = q_act[self.i_muneca]   # la muneca se queda como esta
        return mejor_q, mejor_err


class RobotTron2:
    def __init__(self, ruta_xml, escenario=None, orca=False):
        """orca=True: quita las pinzas y monta una OrcaHand en cada muneca (mano_orca.py)."""
        self.origen = ruta_xml
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
        self.brazos = {lado: Brazo(self.m, self.d, lado) for lado in ("L", "R")}
        self.manos = {}
        if orca:
            for lado, brazo in self.brazos.items():
                brazo.mano = self.manos[lado] = mano_orca.ManoOrca(self.m, self.d, lado)
        self.actualizar()
        # Orientacion del torso en el mundo (x delante, y izquierda, z arriba)
        self.R_base = self.d.xmat[self.brazos["L"].raiz].reshape(3, 3).copy()

    def actualizar(self):
        """Recalcula posiciones para dibujar y para la IK. Se evita mj_forward a
        proposito: ese hace ademas colisiones, restricciones y sensores, que aqui
        no hacen falta y con las mallas de la pinza cuestan cientos de ms."""
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
            if b.j_pinza:
                for n, c, a in zip(b.n_pinza, b.q_cerrado, b.q_abierto):
                    print(f"   pinza: {n:28s} cerrada {c:+.2f} -> abierta {a:+.2f}")
            else:
                print("   pinza: este modelo no tiene articulaciones de pinza")
        print()