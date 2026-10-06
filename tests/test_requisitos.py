"""Avisos de requisitos (requisitos.py): encuadre y extrinsecas."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requisitos
from requisitos import Requisitos, encuadre


def _lm2d(cabeza_y=60.0, muneca_y=300.0, hombros=(260.0, 380.0)):
    lm = [(320.0, 240.0, 0.9)] * 33
    for i in (0, 2, 5):
        lm[i] = (320.0, cabeza_y, 0.9)
    lm[11], lm[12] = (hombros[1], 150.0, 0.9), (hombros[0], 150.0, 0.9)
    for i in (13, 14):
        lm[i] = (320.0, 220.0, 0.9)
    for i in (15, 16):
        lm[i] = (320.0, muneca_y, 0.9)
    return lm


def test_encuadre_como_en_la_demo():
    bien = encuadre(_lm2d(), 640, 480)
    assert not bien["cabeza_cortada"] and bien["brazo_fuera"] is None
    demo = encuadre(_lm2d(cabeza_y=-40.0, muneca_y=-60.0), 640, 480)     # A: sin cabeza ni manos arriba
    assert demo["cabeza_cortada"] and demo["brazo_fuera"] == "arriba"
    cerca = encuadre(_lm2d(hombros=(100.0, 500.0)), 640, 480)
    assert cerca["ancho_hombros"] > requisitos.ANCHO_HOMBROS_MAX


def test_sin_extrinsecas_se_avisa_desde_el_principio():
    t = [0.0]
    hilo = SimpleNamespace(fps=25.0, fps_camara=30.0, ultimo=lambda: None)
    fusion = SimpleNamespace(valida=[True, False], t_valida=[True, False], desalineada=[False, False])
    req = Requisitos([hilo, hilo], fusion, reloj=lambda: t[0], extrinsecas=False,
                     motivo_extrinsecas="no existe camaras_extrinsecas.json")
    lista = req.actualizar(60.0)
    assert any("SIN calibracion ChArUco" in x and "no existe" in x for x in lista)
    fusion.valida[1] = fusion.t_valida[1] = True        # B se calibro sola: aviso suave
    t[0] = 1.0
    assert any("situada sola" in x for x in req.actualizar(60.0))
