"""Cruces izquierda/derecha (fusion.FiltroCruces): casos de la demo de oct-2026."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion import FiltroCruces


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _brazo(flexion_deg, abierto):
    """Brazo colgando (marco camara: y abajo) con 'flexion' en el codo."""
    d_b = _unit([abierto, 1.0, 0.0])
    a = np.radians(flexion_deg)
    d_a = _unit([abierto, np.cos(a), -np.sin(a)])
    return {"dir_brazo": d_b, "dir_antebrazo": d_a, "fuente": "profundidad", "vis": 0.9}


def _snap(n, t, flex_l, flex_r, cara):
    lm2d = [(0.0, 0.0, 0.0)] * 33
    for i in (0, 2, 5):
        lm2d[i] = (320.0, 100.0, cara)
    return {"n": n, "t": t, "lm2d": lm2d,
            "brazos": {"L": _brazo(flex_l, -0.1), "R": _brazo(flex_r, 0.1)}}


def test_camara_sin_calibrar_no_da_la_vuelta_a_la_principal():
    """A (en uso, sin ver la cabeza) y B (sin calibrar, ve la cara) no coinciden en los
    rasgos de los brazos. B no esta en la fusion: A no se puede dar la vuelta por ella."""
    filtro = FiltroCruces(2)
    R = [np.eye(3), np.eye(3)]
    cruzar = [False, False]
    for k in range(200):
        t = 0.035 * k
        sa = _snap(k, t, 8.0 + np.sin(k), 25.0, cara=0.1)        # A: sin cara (cabeza cortada)
        sb = _snap(k, t + 0.01, 25.0, 8.0 + np.sin(k), cara=0.95)  # B: ve la cara, rasgos al reves
        cruzar = filtro.actualizar([sa, sb], [0, 1], [0], R)      # solo A calibrada y en uso
    assert cruzar[0] is False


def test_dos_calibradas_corrigen_el_cruce_de_la_que_cambia_de_etiquetas():
    """Con las dos calibradas, si B cambia de golpe izq/der (continuidad) se corrige B."""
    filtro = FiltroCruces(2)
    R = [np.eye(3), np.eye(3)]
    cruzar = [False, False]
    for k in range(120):
        t = 0.035 * k
        sa = _snap(k, t, 10.0, 60.0, cara=0.95)
        sb = _snap(k, t + 0.01, 10.0, 60.0, cara=0.95)
        if k >= 60:                                   # MediaPipe cambia las etiquetas en B
            sb["brazos"] = {"L": sb["brazos"]["R"], "R": sb["brazos"]["L"]}
        cruzar = filtro.actualizar([sa, sb], [0, 1], [0, 1], R)
    assert cruzar == [False, True]

