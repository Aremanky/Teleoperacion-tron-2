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
  - tirar suavemente de todas las articulaciones hacia la postura de reposo
    (asi la orientacion de la muneca, que aun no se controla, se queda quieta)
"""
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)

# Cuerpo final de cada brazo, por orden de preferencia ({lado} = L o R)
CANDIDATOS_EF = ["grasper_{lado}_Link", "wrist_roll_{lado}_Link"]

# Distancia maxima (m) del centro estimado a cada eje para considerar
# que el hombro / la muneca son esfericos (los tres ejes se cortan)
TOL_ESFERICO = 0.03


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

        # --- Geometria del brazo, medida en la postura de reposo ---
        self.reposo()
        mujoco.mj_forward(m, d)

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

    def _limitar(self, q):
        q[self.limitado] = np.clip(q[self.limitado], self.lo[self.limitado], self.hi[self.limitado])
        return q

    # ---------------- IK ----------------
    def resolver(self, obj_codo, obj_muneca, iteraciones=8, w_codo=0.6,
                 lam=0.05, k_reposo=0.01, paso_max=0.2, max_por_llamada=0.3):
        """Mueve las articulaciones del brazo hacia los objetivos (coordenadas del mundo).
        La vuelta a reposo se aplica solo en el espacio nulo de las tareas, para
        no alejar el brazo de los objetivos. max_por_llamada limita cuanto puede
        girar cada articulacion por frame (rad).
        Devuelve el error final (m) del codo y de la muneca."""
        m, d = self.m, self.d
        q0 = d.qpos[self.qadr].copy()
        I = np.eye(len(self.juntas))
        for _ in range(iteraciones):
            p_c, p_m = self.puntos()
            mujoco.mj_jac(m, d, self._jac, None, p_c, self.b_codo)
            J_c = self._jac[:, self.dofs]
            mujoco.mj_jac(m, d, self._jac, None, p_m, self.b_muneca)
            J_m = self._jac[:, self.dofs]

            J = np.vstack([w_codo * J_c, J_m])
            e = np.concatenate([w_codo * (obj_codo - p_c), obj_muneca - p_m])
            q = d.qpos[self.qadr]
            # Pseudoinversa amortiguada: J+ = J^T (J J^T + lam^2 I)^-1
            J_pinv = np.linalg.solve(J @ J.T + lam ** 2 * np.eye(J.shape[0]), J).T
            dq = J_pinv @ e + (I - J_pinv @ J) @ (k_reposo * (self.q_reposo - q))
            mayor = np.abs(dq).max()
            if mayor > paso_max:
                dq *= paso_max / mayor
            q = self._limitar(q + dq)
            q = q0 + np.clip(q - q0, -max_por_llamada, max_por_llamada)
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)  # necesario para mj_jac
        p_c, p_m = self.puntos()
        return float(np.linalg.norm(obj_codo - p_c)), float(np.linalg.norm(obj_muneca - p_m))


class RobotTron2:
    def __init__(self, ruta_xml):
        self.ruta = quitar_base_flotante(ruta_xml)
        self.m = mujoco.MjModel.from_xml_path(self.ruta)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.brazos = {lado: Brazo(self.m, self.d, lado) for lado in ("L", "R")}
        mujoco.mj_forward(self.m, self.d)
        # Orientacion del torso en el mundo (x delante, y izquierda, z arriba)
        self.R_base = self.d.xmat[self.brazos["L"].raiz].reshape(3, 3).copy()

    def reposo(self):
        for b in self.brazos.values():
            b.reposo()
        mujoco.mj_forward(self.m, self.d)

    def resumen(self):
        print(f"\nModelo: {self.ruta}  (nq={self.m.nq}, nu={self.m.nu})")
        for lado, b in self.brazos.items():
            print(f"\nBrazo {lado}  ->  efector final: {nombre(self.m, mujoco.mjtObj.mjOBJ_BODY, b.ef)}")
            for j, n in zip(b.juntas, b.nombres):
                papel = "hombro" if j in b.j_hombro else ("codo" if j == b.j_codo else "muneca")
                lo, hi = self.m.jnt_range[j]
                print(f"   {n:30s} {papel:7s} [{lo:+.2f}, {hi:+.2f}] rad")
            print(f"   brazo {b.L_brazo * 100:.1f} cm | antebrazo {b.L_antebrazo * 100:.1f} cm | "
                  f"centro del hombro {np.round(b.hombro, 3)}")
        print()