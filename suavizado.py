"""
Suavizado continuo para los objetivos del robot.

PROBLEMA QUE RESUELVE
  La camara (y MediaPipe) entregan un objetivo nuevo solo ~10-15 veces por segundo,
  pero el robot se mueve a 60 Hz. Un filtro exponencial sobre esa "escalera" produce
  un movimiento a tirones: el robot acelera hacia cada escalon y frena antes del
  siguiente. Aqui se usa un filtro de 2.o orden criticamente amortiguado (muelle +
  amortiguador): la salida tiene posicion Y velocidad continuas, no rebasa el
  objetivo y se comporta igual aunque el intervalo entre ciclos varie.

  suavizar() es la solucion exacta aproximada de "Game Programming Gems 4"
  (SmoothDamp): estable para cualquier dt.

  ZonaMuerta evita el temblor en reposo: el objetivo solo se mueve cuando la medida
  se aleja mas de 'radio' del punto que se mantiene (y entonces lo arrastra).

  FiltroOneEuro (Casiez 2012) para las MEDIDAS (direcciones, ctrl de los dedos,
  orientaciones): poco temblor en reposo y poco retraso al moverse. Sustituye a las
  medias moviles, que retrasan siempre lo mismo.

  RotacionSuave: muelle criticamente amortiguado sobre una ORIENTACION (matriz 3x3),
  trabajando con el vector de rotacion relativo; no tiene los problemas de interpolar
  matrices o cuaterniones elemento a elemento.
"""
import numpy as np


def suavizar(x, v, objetivo, omega, dt):
    """Un paso de filtro criticamente amortiguado.
    x, v: posicion y velocidad actuales (escalares o arrays). omega: rad/s
    (mas alto = mas rapido; la constante de tiempo es ~ 2/omega). Devuelve (x, v)."""
    if dt <= 0:
        return x, v
    h = omega * dt
    exp = 1.0 / (1.0 + h + 0.48 * h * h + 0.235 * h ** 3)
    cambio = x - objetivo
    temp = (v + omega * cambio) * dt
    v_nueva = (v - omega * temp) * exp
    x_nueva = objetivo + (cambio + temp) * exp
    return x_nueva, v_nueva


class ObjetivoSuave:
    """Vector (o escalar) que persigue un objetivo con zona muerta + muelle amortiguado.
    Uso:  obj = ObjetivoSuave(omega=14, zona_muerta=0.004)
          valor = obj.actualizar(medida, dt)      # en CADA ciclo, con la ultima medida
    """

    def __init__(self, omega=14.0, zona_muerta=0.0):
        self.omega, self.zona_muerta = omega, zona_muerta
        self.x = self.v = self.ancla = None

    def reiniciar(self):
        self.x = self.v = self.ancla = None

    def actualizar(self, medida, dt):
        medida = np.asarray(medida, dtype=float)
        if self.x is None:
            self.x, self.v, self.ancla = medida.copy(), np.zeros_like(medida), medida.copy()
            return self.x.copy()
        if self.zona_muerta > 0:
            d = medida - self.ancla
            n = float(np.linalg.norm(d))
            if n > self.zona_muerta:       # solo se mueve lo que sobrepasa la zona muerta
                self.ancla = self.ancla + d * (1.0 - self.zona_muerta / n)
            objetivo = self.ancla
        else:
            objetivo = medida
        self.x, self.v = suavizar(self.x, self.v, objetivo, self.omega, dt)
        return self.x.copy()

    @property
    def valor(self):
        return None if self.x is None else self.x.copy()


def comprimir_alcance(hombro, punto, alcance, r_suave=0.92, r_max=0.97):
    """Acerca 'punto' al hombro si cae cerca del alcance maximo del brazo.
    Con el brazo del todo estirado la IK esta en una SINGULARIDAD (el jacobiano pierde
    una direccion): el robot tiembla y no llega con precision. Se comprime suavemente
    la distancia: por debajo de r_suave*alcance no cambia nada y nunca pasa de
    r_max*alcance (sin escalones: la curva es continua y de pendiente continua)."""
    d = np.asarray(punto, float) - np.asarray(hombro, float)
    dist = float(np.linalg.norm(d))
    if dist < 1e-9 or alcance <= 0:
        return np.asarray(punto, float)
    r = dist / alcance
    if r <= r_suave:
        return np.asarray(punto, float)
    ancho = r_max - r_suave
    r_nueva = r_suave + ancho * np.tanh((r - r_suave) / ancho)
    return np.asarray(hombro, float) + d * (r_nueva * alcance / dist)


# ------------------------------------------------------------------ One Euro
class FiltroOneEuro:
    """Filtro One Euro (Casiez et al., 2012) para escalares o vectores.
    min_cutoff (Hz): mas bajo = menos temblor en reposo (pero mas retraso).
    beta: mas alto = menos retraso en movimientos rapidos (pero mas temblor).
    Se reinicia solo si pasa 'reinicio' s sin recibir datos."""

    def __init__(self, min_cutoff=1.0, beta=0.5, d_cutoff=1.0, reinicio=0.5):
        self.min_cutoff, self.beta, self.d_cutoff, self.reinicio = min_cutoff, beta, d_cutoff, reinicio
        self.x = self.dx = self.t = None

    @staticmethod
    def _alfa(corte, dt):
        tau = 1.0 / (2 * np.pi * corte)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x is None or t - self.t > self.reinicio:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self.t
        if dt < 1e-4:
            return self.x.copy()
        a_d = self._alfa(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alfa(self.min_cutoff + self.beta * float(np.linalg.norm(self.dx)), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


class FiltroDedos:
    """One Euro para los 16 ctrl de los dedos, con la misma interfaz que tenia el antiguo
    SuavizadorND (media movil): actualizar(valores), reset(), set_ventana() (no hace nada:
    se mantiene para no tocar retargeting.py). Usa el reloj de pared, porque el
    retargeting no recibe el instante de captura."""

    def __init__(self, n_dim, min_cutoff=1.5, beta=0.8, reloj=None):
        import time
        self.n_dim = n_dim
        self.reloj = reloj or time.monotonic
        self.filtro = FiltroOneEuro(min_cutoff, beta)

    def set_ventana(self, nueva_ventana):
        pass

    def actualizar(self, valores):
        return self.filtro(np.asarray(valores, dtype=float), self.reloj())

    def reset(self):
        self.filtro.reset()


# ------------------------------------------------------------------ rotaciones
def log_rot(R):
    """Vector eje*angulo (rad) de una matriz de rotacion."""
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    ang = float(np.arccos(c))
    if ang < 1e-9:
        return np.zeros(3)
    if np.pi - ang < 1e-4:          # ~180 grados: el eje sale de R + I
        M = R + np.eye(3)
        v = M[:, int(np.argmax(np.linalg.norm(M, axis=0)))]
        return v / max(np.linalg.norm(v), 1e-12) * ang
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(ang))
    return v * ang


def exp_rot(w):
    """Matriz de rotacion de un vector eje*angulo (Rodrigues)."""
    w = np.asarray(w, dtype=float)
    ang = float(np.linalg.norm(w))
    if ang < 1e-12:
        return np.eye(3)
    k = w / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def ortonormalizar(R):
    """Gram-Schmidt: devuelve la rotacion valida mas cercana a R."""
    x = R[:, 0] / np.linalg.norm(R[:, 0])
    y = R[:, 1] - x * np.dot(R[:, 1], x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


class RotacionSuave:
    """Orientacion (3x3) que persigue una orientacion objetivo con un muelle criticamente
    amortiguado. Trabaja con el vector de rotacion RELATIVO (objetivo respecto al valor
    actual), asi que es continuo y no rebasa. Igual que ObjetivoSuave, pero para rotaciones.
    Un salto del objetivo mayor que 'salto_max' (rad) se ignora salvo que se repita
    'saltos_repetidos' veces seguidas (los marcos de MediaPipe a veces dan una vuelta)."""

    def __init__(self, omega=12.0, salto_max=np.radians(60), saltos_repetidos=3):
        self.omega, self.salto_max, self.saltos_repetidos = omega, salto_max, saltos_repetidos
        self.R = self.v = None
        self.obj = None
        self._saltos = 0

    def reiniciar(self):
        self.R = self.v = self.obj = None
        self._saltos = 0

    def fijar_objetivo(self, R_obj):
        """Nuevo objetivo (una vez por foto). Filtra saltos espurios."""
        if R_obj is None:
            return
        R_obj = ortonormalizar(np.asarray(R_obj, dtype=float))
        if self.obj is not None and self.salto_max > 0:
            salto = float(np.linalg.norm(log_rot(R_obj @ self.obj.T)))
            if salto > self.salto_max and self._saltos < self.saltos_repetidos:
                self._saltos += 1
                return
        self._saltos = 0
        self.obj = R_obj

    def actualizar(self, dt):
        """Un paso del muelle (en CADA ciclo). Devuelve la orientacion suavizada o None."""
        if self.obj is None:
            return None
        if self.R is None:
            self.R, self.v = self.obj.copy(), np.zeros(3)
            return self.R.copy()
        # estado = vector de rotacion de R respecto al objetivo (R = exp(x) @ obj);
        # el muelle lo lleva a 0 con posicion y velocidad continuas
        x = log_rot(self.R @ self.obj.T)
        x, self.v = suavizar(x, self.v, np.zeros(3), self.omega, dt)
        self.R = ortonormalizar(exp_rot(x) @ self.obj)
        return self.R.copy()

    @property
    def valor(self):
        return None if self.R is None else self.R.copy()
