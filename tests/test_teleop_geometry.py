import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_tron2 import unwrap_angular_delta
from seguimiento_brazos import SeguidorBrazos
from teleop_tron2 import marco_mano, objetivo_brazo, objetivos


def test_objetivos_mirror_uses_robot_frame_after_camera_rotation():
    robot = SimpleNamespace(
        R_base=np.eye(3),
        brazos={
            "L": SimpleNamespace(
                hombro=np.zeros(3, dtype=float),
                L_brazo=1.0,
                L_antebrazo=1.0,
            )
        },
    )
    datos = {
        "dir_brazo": np.array([1.0, 0.0, 0.0]),
        "dir_antebrazo": np.array([1.0, 0.0, 0.0]),
    }
    R_cam = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    codo, muneca = objetivos(robot, "L", datos, R_cam, espejo=True)

    assert np.allclose(codo, np.array([0.0, -1.0, 0.0]))
    assert muneca[0] == 0.0
    assert muneca[1] < 0.0
    assert np.linalg.norm(muneca - codo) <= 2.0


def test_objetivo_brazo_includes_palm_orientation_for_ik():
    robot = SimpleNamespace(
        R_base=np.eye(3),
        brazos={
            "L": SimpleNamespace(
                hombro=np.zeros(3, dtype=float),
                L_brazo=1.0,
                L_antebrazo=1.0,
            )
        },
    )
    datos = {
        "dir_brazo": np.array([1.0, 0.0, 0.0]),
        "dir_antebrazo": np.array([1.0, 0.0, 0.0]),
        "marco": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    }
    R_cam = np.eye(3)

    codo, muneca, R_obj = objetivo_brazo(robot, "L", datos, R_cam, espejo=False)

    assert np.allclose(codo, np.array([1.0, 0.0, 0.0]))
    assert muneca[0] > 0.0 and muneca[1] == 0.0
    assert np.linalg.norm(muneca - codo) <= 2.0
    assert np.allclose(R_obj[:, 0], np.array([1.0, 0.0, 0.0]))


def test_objetivo_brazo_handles_missing_palm_frame():
    robot = SimpleNamespace(
        R_base=np.eye(3),
        brazos={
            "L": SimpleNamespace(
                hombro=np.zeros(3, dtype=float),
                L_brazo=1.0,
                L_antebrazo=1.0,
            )
        },
    )
    datos = {
        "dir_brazo": np.array([1.0, 0.0, 0.0]),
        "dir_antebrazo": np.array([1.0, 0.0, 0.0]),
    }
    R_cam = np.eye(3)

    codo, muneca, R_obj = objetivo_brazo(robot, "L", datos, R_cam, espejo=False)

    assert np.allclose(codo, np.array([1.0, 0.0, 0.0]))
    assert muneca[0] > 0.0 and np.isclose(muneca[1], 0.0) and np.isclose(muneca[2], 0.0)
    assert np.linalg.norm(muneca - codo) <= 2.0
    assert np.allclose(R_obj, np.eye(3))


def test_unwrap_angular_delta_avoids_shoulder_spin_across_pi():
    delta = np.array([3.20, -3.20, 0.05])
    unwrapped = unwrap_angular_delta(delta)

    assert np.all(np.abs(unwrapped) < np.pi)
    assert np.allclose(unwrapped, np.array([-3.08318531, 3.08318531, 0.05]), atol=1e-6)


def test_mano_marco_none_falls_back_to_identity():
    robot = SimpleNamespace(R_base=np.eye(3))
    datos = {"marco": None}
    R = marco_mano(robot, datos, np.eye(3), espejo=False)
    assert np.allclose(R, np.eye(3))


def test_seguidor_rechaza_brazos_imposibles():
    pts_ok = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.30, 0.0, 0.0]),
        np.array([0.57, 0.0, 0.0]),
    ]
    pts_bad = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.10, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0]),
    ]

    assert SeguidorBrazos._es_brazo_valido(pts_ok, codo_tapado=False)
    assert not SeguidorBrazos._es_brazo_valido(pts_bad, codo_tapado=False)
