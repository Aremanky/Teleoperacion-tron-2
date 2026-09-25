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

# Cuerpo final de cada brazo, por orden de preferencia ({lado} = L o R)
CANDIDATOS_EF = ["grasper_{lado}_Link", "wrist_roll_{lado}_Link"]

# Distancia maxima (m) del centro estimado a cada eje para considerar
# que el hombro / la muneca son esfericos (los tres ejes se cortan)
TOL_ESFERICO = 0.03

VEL_MAX = 3.0          # rad/s: velocidad maxima de cada articulacion al seguir al operador
VEL_PINZA = 2.5        # 1/s: velocidad de apertura y cierre de la pinza (0 a 1 en 0.4 s)
PINZA_INVERTIDA = False  # True si toda la pinza abre cuando deberia cerrar
# Juntas que llevan "grasp" en el nombre pero NO forman parte del mecanismo de la
# pinza (p.ej. la que orienta la pinza entera). Se dejan quietas.
PINZA_EXCLUIR = ("grasper_base",)
# Juntas sueltas del mecanismo cuyo sentido hay que invertir (por su nombre exacto)
PINZA_JUNTAS_INVERTIDAS = ()
# Recorrido a mano para juntas de la pinza SIN limites en el XML (no se puede
# adivinar): {"grasper_L_jaw_left_Joint": (cerrado, abierto), ...}
PINZA_RECORRIDO = {}
VEL_TRANSICION = 1.5   # rad/s: al cambiar a otra solucion tras un rescate
MARGEN_LIMITE = 0.1    # rad: junto a un limite se empuja la articulacion hacia dentro
ERROR_ATASCO = 0.08    # m: error a partir del cual se considera que el brazo esta atascado
T_ATASCO = 0.5         # s: tiempo atascado antes de buscar otra solucion


def nombre(m, tipo, i):
    return mujoco.mj_id2name(m, tipo, i) or f"<{i}>"


def quitar_base_flotante(ruta_xml):
    """Crea robot_fixed.xml junto al original, sin free joint (torso fijo al mundo)."""
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
        if self.j_pinza:
            self.apertura = 1.0
            self.d.qpos[self.qadr_pinza] = self.q_abierto

    def _limitar(self, q):
        q[self.limitado] = np.clip(q[self.limitado], self.lo[self.limitado], self.hi[self.limitado])
        return q

    # ---------------- IK ----------------
    def _fijar(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)  # necesario para mj_jac

    def error(self, obj_c, obj_m, w_codo):
        """(error ponderado, error codo, error muneca) en metros."""
        p_c, p_m = self.puntos()
        ec, em = np.linalg.norm(obj_c - p_c), np.linalg.norm(obj_m - p_m)
        return float(np.hypot(w_codo * ec, em)), float(ec), float(em)

    def _paso(self, obj_c, obj_m, w_codo, lam, k_reposo, k_limite):
        """Un paso de IK que respeta los limites: si una articulacion se saldria de su
        rango, se bloquea y se recalcula el paso para que las demas compensen."""
        m, d = self.m, self.d
        p_c, p_m = self.puntos()
        mujoco.mj_jac(m, d, self._jac, None, p_c, self.b_codo)
        J_c = self._jac[:, self.dofs]
        mujoco.mj_jac(m, d, self._jac, None, p_m, self.b_muneca)
        J_m = self._jac[:, self.dofs]
        J = np.vstack([w_codo * J_c, J_m])
        e = np.concatenate([w_codo * (obj_c - p_c), obj_m - p_m])

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

    def resolver(self, obj_c, obj_m, w_codo=0.6, iteraciones=4, max_cambio=np.inf,
                 lam=0.05, k_reposo=0.01, k_limite=0.02, paso_max=0.2):
        """IK por minimos cuadrados amortiguados con busqueda lineal: solo se aceptan
        pasos que no empeoran el error, asi el brazo nunca 'da bandazos'.
        max_cambio: giro maximo de cada articulacion en esta llamada (rad).
        Devuelve el error final (m) del codo y de la muneca."""
        q0 = self.d.qpos[self.qadr].copy()
        err = self.error(obj_c, obj_m, w_codo)[0]
        for _ in range(iteraciones):
            q, dq = self._paso(obj_c, obj_m, w_codo, lam, k_reposo, k_limite)
            mayor = np.abs(dq).max()
            if mayor > paso_max:
                dq *= paso_max / mayor
            alfa, aceptado = 1.0, False
            for _ in range(4):
                q_n = self._limitar(q + alfa * dq)
                q_n = q0 + np.clip(q_n - q0, -max_cambio, max_cambio)
                self._fijar(q_n)
                err_n = self.error(obj_c, obj_m, w_codo)[0]
                if err_n <= err + 1e-4:
                    err, aceptado = err_n, True
                    break
                alfa *= 0.5
            if not aceptado:
                self._fijar(q)
                break
        _, ec, em = self.error(obj_c, obj_m, w_codo)
        return ec, em

    def seguir(self, obj_c, obj_m, dt, w_codo=0.6):
        """Llamar en cada ciclo del bucle de control. IK con limite de velocidad y,
        si el brazo se queda atascado lejos del objetivo (minimo local, limites),
        busca otra solucion y va hacia ella de forma suave."""
        ahora = time.monotonic()
        if self.q_transicion is not None:
            q = self.d.qpos[self.qadr].copy()
            falta = self.q_transicion - q
            if np.abs(falta).max() < 0.02:
                self.q_transicion = None
            else:
                self._fijar(q + np.clip(falta, -VEL_TRANSICION * dt, VEL_TRANSICION * dt))
                _, ec, em = self.error(obj_c, obj_m, w_codo)
                return ec, em

        ec, em = self.resolver(obj_c, obj_m, w_codo, max_cambio=VEL_MAX * dt)
        err = self.error(obj_c, obj_m, w_codo)[0]
        self.t_atascado = self.t_atascado + dt if err > ERROR_ATASCO else 0.0
        if self.t_atascado > T_ATASCO and ahora - self.t_rescate > 1.0:
            self.t_rescate, self.t_atascado = ahora, 0.0
            q_mejor, err_mejor = self._rescate(obj_c, obj_m, w_codo)
            if err_mejor < 0.5 * err:
                self.q_transicion = q_mejor
        return ec, em

    def _rescate(self, obj_c, obj_m, w_codo):
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
            self.resolver(obj_c, obj_m, w_codo, iteraciones=25)
            err = self.error(obj_c, obj_m, w_codo)[0]
            if err < mejor_err:
                mejor_q, mejor_err = self.d.qpos[self.qadr].copy(), err
        self._fijar(q_act)
        return mejor_q, mejor_err


class RobotTron2:
    def __init__(self, ruta_xml):
        self.origen = ruta_xml
        self.ruta = quitar_base_flotante(ruta_xml)
        self.m = mujoco.MjModel.from_xml_path(self.ruta)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        self.brazos = {lado: Brazo(self.m, self.d, lado) for lado in ("L", "R")}
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
            b.q_transicion, b.t_atascado = None, 0.0
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