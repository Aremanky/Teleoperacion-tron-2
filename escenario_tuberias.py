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

MESA_CENTRO = (0.46, -0.22)
MESA_MEDIO = (0.20, 0.24)        # semiancho en x, y
MESA_ALTO = 0.95                 # altura del tablero

# Pieza suelta: origen en el codo. Tramo 1 por +y local, tramo 2 por -z local.
EXTREMOS = [  # (punto local, direccion hacia fuera)
    (np.array([0.0, L1, 0.0]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, -L2]), np.array([0.0, 0.0, -1.0])),
]
# Sobre la mesa, tumbada: tramo 1 hacia +y del mundo, tramo 2 hacia +x
POS_INICIAL = np.array([0.34, -0.36, MESA_ALTO + R_COPA])
ROT_INICIAL = np.array([[0.0, 0.0, -1.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0]])

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
        self.activa = False             # oculto y sin efecto hasta que se active
        self.reiniciar()

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
        c = self.p + self.R @ (sum(p for p, _ in EXTREMOS) / (len(EXTREMOS) + 1))
        dentro = (abs(c[0] - MESA_CENTRO[0]) < MESA_MEDIO[0] and abs(c[1] - MESA_CENTRO[1]) < MESA_MEDIO[1])
        return MESA_ALTO if dentro else 0.0

    def _z_minima(self):
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

        if self.estado == "sujeta":
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
            return "Tuberia: INSTALADA  (1 = empezar de nuevo)"
        if self.estado == "sujeta":
            s = f"Tuberia: en pinza {nombres[self.sujeta]}"
            if self.dist is not None:
                s += f" | conexion a {self.dist * 100:4.1f} cm, {self.ang:3.0f} deg"
                s += f" (encaja < {TOL_POS * 100:.0f} cm y < {TOL_ANG:.0f} deg)"
            return s
        if self.estado == "cayendo":
            return "Tuberia: cayendo"
        if self.estado == "suelo":
            return "Tuberia: se ha caido al suelo, vuelve a la mesa..."
        return "Tuberia: en la mesa  (cierra la mano junto a ella para cogerla)"