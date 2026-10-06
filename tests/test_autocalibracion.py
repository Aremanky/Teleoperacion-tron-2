"""Calibracion automatica de la rotacion de una camara (fusion.SolucionadorRotacion)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion import SolucionadorRotacion, _unitario, angulo_rotacion_deg, rot_eje

ARRIBA = np.array([0.0, 0.0, 1.0])
R_B = rot_eje(ARRIBA, np.radians(38))       # B girada 38 grados respecto a A (como en la demo)


def _solver(ruido_deg, semilla=0):
    rng = np.random.default_rng(semilla)
    s = SolucionadorRotacion()
    for k in range(500):
        a = _unitario(rng.normal(size=3) * [1.0, 1.0, 0.6] + [0.3, 0.0, -0.3])
        b = _unitario(R_B.T @ a + rng.normal(scale=np.radians(ruido_deg), size=3))
        s.anadir(0.03 * k, a, b, 1.0)
    return s


def test_con_imu_buena_se_usa_la_gravedad():
    r = _solver(8.0).resolver(None, ARRIBA, R_B.T @ ARRIBA)
    assert r["ok"] and r["modo"] == "gravedad" and r["imu_incoherente"] is None
    assert angulo_rotacion_deg(r["R"] @ R_B.T) < 2.0


def test_imu_desviada_no_da_una_calibracion_torcida():
    """La gravedad de la IMU de B desviada 17 grados: antes se aceptaba una calibracion con
    ese error (residuo ~12 grados, 'aceptable'); ahora se detecta y se calibra sin la IMU."""
    arriba_mal = rot_eje(np.array([1.0, 0.0, 0.0]), np.radians(17.0)) @ (R_B.T @ ARRIBA)
    r = _solver(8.0).resolver(None, ARRIBA, arriba_mal)
    assert r["imu_incoherente"] is not None
    assert r["ok"] and angulo_rotacion_deg(r["R"] @ R_B.T) < 3.0


def test_parejas_sin_estructura_no_se_aceptan():
    r = _solver(45.0).resolver(None, ARRIBA, R_B.T @ ARRIBA)
    assert not r["ok"]
