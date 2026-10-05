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
