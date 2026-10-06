"""La fusion ante un aviso de desplazamiento de la IMU (camara B con calibracion ChArUco)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grabacion
from fusion import FusionCamaras
from simular_operador import OperadorSintetico


class _Reloj:
    def __init__(self, t=5000.0):
        self.t = t

    def __call__(self):
        return self.t


def _fusion_con_sesion(tmp_path, mover_b=None, t_mover=6.0):
    """mover_b: desplazamiento (m, en W) de la camara B a partir de t_mover s: sus puntos
    grabados se recalculan como los veria desde su nueva posicion."""
    ruta = tmp_path / "s.pkl"
    op = OperadorSintetico(14.0, fps=15.0)
    datos = op.grabar(str(ruta))
    if mover_b is not None:
        RB, _ = op.cams["B"]
        d_cam = RB.T @ np.asarray(mover_b, float)      # en el marco de la camara B
        for r in datos["registros"]["B"]:
            if r["t"] - datos["t0"] < t_mover:
                continue
            K = r["K"]
            for pt in r["puntos"].values():
                if pt["z"] is None:
                    continue
                u, v = pt["uv"]
                p = np.array([(u - K[0, 2]) * pt["z"] / K[0, 0], (v - K[1, 2]) * pt["z"] / K[1, 1], pt["z"]]) - d_cam
                pt["uv"] = (K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2])
                pt["z"] = float(p[2])
    reloj = _Reloj()
    hilos, _ = grabacion.hilos_reproduccion(datos, reloj=reloj)
    R_ab, t_ab = (np.asarray(x) for x in datos["meta"]["extrinsecas"])
    fusion = FusionCamaras(hilos, reloj=reloj, extrinsecas=(R_ab, t_ab))
    fusion.op = op
    return fusion, hilos, reloj


def _correr(fusion, reloj, segundos, al_llegar=None):
    t_fin = reloj.t + segundos
    while reloj.t < t_fin:
        fusion.ultimo()
        if al_llegar is not None and al_llegar(reloj.t):
            al_llegar = None
        reloj.t += 1.0 / 60.0


def _aviso_desplazada(hilo, cm):
    imu = hilo.camara.imu
    imu.movimientos += 1
    imu.ultimo_mov = dict(giro=0.3, incl=0.0, despl=cm / 100.0, rotacion=False)


def test_falsa_alarma_de_la_imu_no_tira_la_chArUco(tmp_path):
    fusion, hilos, reloj = _fusion_con_sesion(tmp_path)
    _correr(fusion, reloj, 3.0)
    t_antes = fusion.t[1].copy()
    _aviso_desplazada(hilos[1], 6)          # vibraciones: la IMU cree que se ha desplazado
    _correr(fusion, reloj, 8.0)
    assert fusion._fija[1] and fusion.t_valida[1]
    assert np.linalg.norm(fusion.t[1] - t_antes) < 1e-9
    assert fusion._verificar_t[1] is None   # comprobado
    assert "3D" in fusion.estado_texto().split("|")[1]


def test_desplazamiento_real_se_resitua(tmp_path):
    mover = np.array([0.0, 0.12, 0.0])                   # la mueven 12 cm de verdad en t = 6 s
    fusion, hilos, reloj = _fusion_con_sesion(tmp_path, mover_b=mover, t_mover=6.0)
    _correr(fusion, reloj, 5.5)
    t_antes = fusion.t[1].copy()
    _correr(fusion, reloj, 0.6)
    _aviso_desplazada(hilos[1], 12)
    _correr(fusion, reloj, 7.0)
    assert fusion.t_valida[1]
    RA, _ = fusion.op.cams["A"]
    t_verdad = t_antes + fusion.R_op[0] @ RA.T @ mover   # el mismo desplazamiento en el marco de la fusion
    assert np.linalg.norm(fusion.t[1] - t_verdad) < 0.03
