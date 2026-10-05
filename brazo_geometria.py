"""
Geometria del brazo humano que comparten seguimiento_brazos.py (una camara) y fusion.py
(varias camaras).

codo_en_circulo: cuando NADIE ve el codo (brazo apuntando a la camara, oclusion), antes se
daba el brazo por RECTO de hombro a muneca, lo que hacia saltar la muneca del robot 20-30 cm
cuando el codo estaba doblado. Con las longitudes del operador (LongitudesHumano) el codo
esta en una CIRCUNFERENCIA alrededor de la recta hombro-muneca; se elige el punto mas
cercano al ultimo codo conocido (mismo "giro" del codo), y si no hay ninguno, el codo hacia
abajo y hacia fuera, que es como cuelga un codo humano relajado.
"""
import numpy as np


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def codo_en_circulo(hombro, muneca, L_brazo, L_antebrazo, codo_previo=None, abajo=None, fuera=None):
    """Posicion plausible del codo con el hombro y la muneca conocidos.
    Devuelve (codo, reconstruido): reconstruido=False si la geometria obliga a brazo recto."""
    S, W = np.asarray(hombro, float), np.asarray(muneca, float)
    d = W - S
    D = float(np.linalg.norm(d))
    if D < 1e-6:
        return S + np.array([0.0, 0.0, -L_brazo]), False
    u = d / D
    if D >= L_brazo + L_antebrazo - 0.005:          # estirado del todo: recto
        return S + L_brazo * u, False
    if D <= abs(L_brazo - L_antebrazo) + 0.005:     # imposible (muneca pegada al hombro): recto hacia la muneca
        return S + L_brazo * u, False
    a = (L_brazo ** 2 - L_antebrazo ** 2 + D ** 2) / (2.0 * D)
    r = float(np.sqrt(max(L_brazo ** 2 - a ** 2, 1e-9)))
    c = S + a * u
    # referencia del giro del codo: el codo anterior; si no, hacia abajo (+ un poco hacia fuera)
    v = None
    if codo_previo is not None:
        v = np.asarray(codo_previo, float) - c
        v = v - u * np.dot(v, u)
        if np.linalg.norm(v) < 0.02:
            v = None
    if v is None:
        ref = np.array([0.0, 0.0, -1.0]) if abajo is None else -np.asarray(abajo, float)
        if fuera is not None:
            ref = ref + 0.5 * np.asarray(fuera, float)
        v = ref - u * np.dot(ref, u)
        if np.linalg.norm(v) < 1e-3:                # recta vertical: cualquier perpendicular
            v = np.cross(u, [1.0, 0.0, 0.0])
    return c + r * _unit(v), True


def direcciones(hombro, codo, muneca):
    """(direccion del brazo, direccion del antebrazo) unitarias, o (None, None)."""
    d_b, d_a = _unit(np.asarray(codo, float) - np.asarray(hombro, float)), _unit(np.asarray(muneca, float) - np.asarray(codo, float))
    return d_b, d_a
