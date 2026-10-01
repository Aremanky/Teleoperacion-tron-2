"""
Suavizador N-DoF (media móvil) para las señales de control.

(Extraído de Skill_Shield_Nivel_6.py — código sin cambios, solo reubicado.)
"""
import numpy as np
from collections import deque


# ══════════════════════════════════════════════════════════════════════════════
# SUAVIZADOR N-DoF  (filtro de media móvil)
# ══════════════════════════════════════════════════════════════════════════════

class SuavizadorND:
    """Filtro de media móvil (rolling average) para vectores de N dimensiones.
    
    Soporta cambio dinámico de ventana: en zona de puño se reduce a 1
    (sin suavizado) para que el cierre llegue al máximo sin dilución.
    """

    def __init__(self, n_dim: int, ventana: int = 4):
        self._ventana_base = ventana
        self._buf = deque(maxlen=ventana)
        self.n_dim = n_dim

    def set_ventana(self, nueva_ventana: int):
        """Cambia el tamaño de ventana en caliente sin perder el buffer."""
        nueva_ventana = max(1, nueva_ventana)
        if nueva_ventana != self._buf.maxlen:
            datos_actuales = list(self._buf)
            self._buf = deque(datos_actuales[-nueva_ventana:], maxlen=nueva_ventana)

    def actualizar(self, valores) -> np.ndarray:
        self._buf.append(np.asarray(valores, dtype=float))
        return np.mean(self._buf, axis=0)

    def reset(self):
        self._buf.clear()
        self._buf = deque(maxlen=self._ventana_base)
