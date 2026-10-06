"""Deteccion de movimientos de la IMU (camaras.EstadoIMU): vibraciones frente a movimientos reales."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camaras import EstadoIMU

G = np.array([0.0, 0.0, 9.81])
HZ = 100.0


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _alimentar(imu, segundos, t0, rng, vibracion=0.0, giro_z=0.0, R0=np.eye(3)):
    """Muestras a 100 Hz. vibracion: amplitud (m/s2) del balanceo del tripode cuando alguien
    anda cerca o lo roza (3 Hz, +-0.25 grados, +-1 mm: la camara acaba donde estaba);
    giro_z: rad/s de giro real alrededor de la vertical."""
    estado = {"girando": False, "desplazandose": False}
    R = R0.copy()
    n = int(segundos * HZ)
    for k in range(n):
        t = t0 + k / HZ
        w = np.array([0.0, 0.0, giro_z]) + rng.normal(scale=0.004, size=3)
        a = R.T @ G + rng.normal(scale=0.02, size=3)
        if vibracion:
            w += 0.08 * np.sin(2 * np.pi * 3.0 * t) * np.array([0.6, 0.8, 0.0])
            a += vibracion * np.sin(2 * np.pi * 3.0 * t + 0.3) * np.array([0.8, 0.0, 0.6])
        imu.muestra(a, w, t)
        R = R @ _rot_z(giro_z / HZ)
        estado["girando"] |= imu.girando()
        estado["desplazandose"] |= imu.desplazandose()
    return t0 + n / HZ, R, estado


def test_vibraciones_no_cuentan_como_movimiento():
    rng = np.random.default_rng(1)
    imu = EstadoIMU(np.eye(3))
    t, R, _ = _alimentar(imu, 2.0, 0.0, rng)                    # arranque: quieta
    t, R, est = _alimentar(imu, 6.0, t, rng, vibracion=0.25)    # alguien anda al lado
    t, R, est2 = _alimentar(imu, 2.0, t, rng)
    # ni giro (vaiven que va y vuelve) ni la camara fuera de la fusion mientras vibra
    assert not est["girando"] and not est["desplazandose"]
    assert imu.movimientos == 0 or not imu.ultimo_mov["rotacion"]
    # y al parar la vibracion vuelve a estar quieta enseguida (antes: 10 s "moviendose")
    assert not imu.en_movimiento()
    # un desplazamiento sin giro integrado dos veces NO es fiable: la fusion lo comprueba con
    # las medidas del cuerpo antes de tirar una calibracion (tests/test_fusion_imu.py)


def test_un_giro_real_si_cuenta():
    rng = np.random.default_rng(2)
    imu = EstadoIMU(np.eye(3))
    t, R, _ = _alimentar(imu, 2.0, 0.0, rng)
    t, R, est = _alimentar(imu, 1.0, t, rng, giro_z=np.radians(10.0))   # la giran 10 grados
    t, R, _ = _alimentar(imu, 2.0, t, rng, R0=R)
    assert est["girando"]
    assert imu.movimientos == 1 and imu.ultimo_mov["rotacion"]
    assert 7.0 < imu.ultimo_mov["giro"] < 13.0
