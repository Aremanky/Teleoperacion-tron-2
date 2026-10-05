"""
Utilidades de retargeting biomecánico (flexión, abertura, distancia, flexión de muñeca).

(Extraído de Skill_Shield_Nivel_6.py — código sin cambios, solo reubicado.)
"""
import numpy as np

from config_mano import LM_ANULAR_MCP, LM_CORAZON_MCP, LM_INDICE_MCP, LM_MENIQUE_MCP, LM_MUNECA


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE RETARGETING BIOMECÁNICO
# ══════════════════════════════════════════════════════════════════════════════

def calcular_flexion(p1, p2, p3) -> float:
    """
    Flexión en la articulación p2 formada por los segmentos p1→p2 y p2→p3.
    Devuelve un valor ≥ 0 en radianes (0 = dedo extendido).
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return max(0.0, (np.pi - np.arccos(dot)) - 0.10)


def calcular_abertura(p_base1, p_punta1, p_base2, p_punta2) -> float:
    """Ángulo de separación angular entre dos dedos (en radianes, puede ser < 0)."""
    v1 = np.array(p_punta1) - np.array(p_base1)
    v2 = np.array(p_punta2) - np.array(p_base2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)) - 0.18


def calcular_distancia(p1, p2) -> float:
    """Distancia euclidiana 3D entre dos puntos."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def calcular_flexion_muneca(hand_world_lm, ref_vec=None, eje_lat_fijo=None):
    """
    Calcula la flexión/extensión de la muñeca en el plano de bisagra correcto.

    Modo calibración  (ref_vec=None):
        Devuelve (vec_adelante, eje_lateral) — congela la posición neutra.

    Modo medición (ref_vec, eje_lat_fijo provistos):
        Devuelve el ángulo de flexión en radianes (+ flexión, - extensión).
    """
    def lm(i):
        l = hand_world_lm.landmark[i]
        return np.array([l.x, l.y, l.z], float)

    mcp_centro = (lm(LM_INDICE_MCP) + lm(LM_CORAZON_MCP) +
                  lm(LM_ANULAR_MCP) + lm(LM_MENIQUE_MCP)) / 4.0
    direccion  = mcp_centro - lm(LM_MUNECA)
    n          = np.linalg.norm(direccion)
    if n < 1e-6:
        return (np.array([0, 1, 0], float), np.array([1, 0, 0], float)) \
               if ref_vec is None else 0.0
    direccion /= n

    if ref_vec is None:
        # — Modo calibración —
        eje = lm(LM_MENIQUE_MCP) - lm(LM_INDICE_MCP)
        eje = eje - np.dot(eje, direccion) * direccion
        n2  = np.linalg.norm(eje)
        eje = eje / n2 if n2 > 1e-6 else np.array([1.0, 0.0, 0.0])
        return direccion, eje

    # — Modo medición —
    eje    = eje_lat_fijo if eje_lat_fijo is not None else np.array([1.0, 0.0, 0.0])
    ref_p  = ref_vec  - np.dot(ref_vec,  eje) * eje
    dir_p  = direccion - np.dot(direccion, eje) * eje
    nr, nd = np.linalg.norm(ref_p), np.linalg.norm(dir_p)
    if nr < 1e-6 or nd < 1e-6:
        return 0.0
    ref_p /= nr
    dir_p /= nd
    cross = np.cross(ref_p, dir_p)
    return -np.arctan2(np.dot(cross, eje), np.dot(ref_p, dir_p))
