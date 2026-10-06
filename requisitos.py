"""
Condiciones de arranque y de uso de la teleoperacion, comprobadas EN MARCHA y dichas en
pantalla (en rojo, debajo del estado) en lugar de degradarse en silencio.

Nace de la demo de oct-2026 en la que el robot no imitaba al operador y nada en pantalla
decia por que: la camara B no entro nunca en la fusion (sin calibracion ChArUco), la
camara A cortaba la cabeza y las manos levantadas del operador y el bucle del robot iba a
20 Hz. Cada comprobacion tiene su histeresis (no parpadea) y se anota en la consola al
aparecer y al desaparecer.

  Requisito                                   Si no se cumple
  ------------------------------------------  ---------------------------------------------
  2 camaras: extrinsecas ChArUco cargadas     B tarda en entrar (o no entra) y las manos no se
    (camaras_extrinsecas.json de ESTAS dos)   triangulan -> python calibrar_extrinseca.py
  seguimiento (MediaPipe) >= 15 fps / camara  EDAD_MAX, CADUCIDAD y los filtros van mal
  camara >= 20 fps                            USB saturado / cable / mismo controlador
  la camara te ve la cabeza                   sin cara, izquierda/derecha dudosos y la c falla
  los brazos no se salen del cuadro           esa camara no puede dar el brazo (lo inventaria)
  no estar demasiado cerca                    hombros > 55 % del ancho: todo se sale
  bucle del robot >= 30 Hz                    retraso y movimiento a saltos
"""
import time

import numpy as np

FPS_SEG_MIN = 15.0          # fps de seguimiento (MediaPipe) por camara
FPS_CAM_MIN = 20.0          # fps de la camara
HZ_ROBOT_MIN = 30.0         # Hz del bucle del robot
T_GRACIA = 5.0              # s tras arrancar sin avisar de fps (todo se esta poniendo en marcha)
T_ACTIVAR = 2.0             # s que tiene que durar un problema para avisar
T_DESACTIVAR = 2.0          # s sin el problema para quitar el aviso
VENTANA_ENCUADRE = 3.0      # s de fotogramas para decidir si los brazos se salen del cuadro
FRAC_FUERA = 0.3            # fraccion de esos fotogramas con algun punto del brazo fuera
ANCHO_HOMBROS_MAX = 0.55    # fraccion del ancho de la imagen: mas es estar demasiado cerca
PERIODO = 0.25              # s entre comprobaciones
ETIQUETAS = "ABCD"
CARA = (0, 2, 5)            # nariz y ojos (MediaPipe Pose)
BRAZOS = (13, 14, 15, 16)   # codos y munecas
HOMBROS = (11, 12)


def _fuera(u, v, w, h):
    return not (0.0 <= u < w and 0.0 <= v < h)


def _lado_fuera(u, v, w, h):
    """Por donde se sale un punto (en la imagen SIN voltear; la ventana se ve en espejo)."""
    if v < 0:
        return "arriba"
    if v >= h:
        return "abajo"
    return "por un lado"


def encuadre(lm2d, w, h):
    """Rasgos de encuadre de UN fotograma: dict(cabeza_cortada, brazo_fuera (None o por donde),
    ancho_hombros (fraccion) o None) o None si MediaPipe no ve a nadie."""
    if not lm2d:
        return None
    hombros = [lm2d[i] for i in HOMBROS]
    if any(_fuera(u, v, w, h) for u, v, _ in hombros):
        return dict(cabeza_cortada=True, brazo_fuera="arriba", ancho_hombros=None)
    cabeza = sum(1 for i in CARA if _fuera(lm2d[i][0], lm2d[i][1], w, h)) >= 2
    fuera = [_lado_fuera(lm2d[i][0], lm2d[i][1], w, h) for i in BRAZOS if _fuera(lm2d[i][0], lm2d[i][1], w, h)]
    ancho = abs(hombros[0][0] - hombros[1][0]) / max(w, 1)
    return dict(cabeza_cortada=cabeza, brazo_fuera=(max(set(fuera), key=fuera.count) if fuera else None),
                ancho_hombros=ancho)


class _Aviso:
    """Un problema con histeresis: se activa si dura T_ACTIVAR y se quita tras T_DESACTIVAR."""

    def __init__(self):
        self.activo, self._desde, self._hasta, self.texto = False, None, None, ""

    def actualizar(self, hay, texto, ahora, t_activar=T_ACTIVAR):
        if hay:
            self.texto = texto
            self._hasta = None
            self._desde = ahora if self._desde is None else self._desde
            if not self.activo and ahora - self._desde >= t_activar:
                self.activo = True
                return "+"
        else:
            self._desde = None
            self._hasta = ahora if self._hasta is None else self._hasta
            if self.activo and ahora - self._hasta >= T_DESACTIVAR:
                self.activo = False
                return "-"
        return None


class Requisitos:
    def __init__(self, hilos, fusion=None, reloj=time.monotonic, extrinsecas=None, motivo_extrinsecas=None):
        """extrinsecas: True si se cargo la calibracion ChArUco entre camaras (solo con 2 o mas)."""
        self.hilos, self.fusion, self.reloj = list(hilos), fusion, reloj
        self.extrinsecas, self.motivo_extrinsecas = extrinsecas, motivo_extrinsecas
        self.t0 = reloj()
        self._t_ult = -1e9
        self._avisos = {}
        self._hist = [[] for _ in self.hilos]      # (t, brazo_fuera) por camara
        self._n = [-1] * len(self.hilos)
        self.lista = []

    def _aviso(self, clave, hay, texto, ahora, inmediato=False):
        a = self._avisos.setdefault(clave, _Aviso())
        cambio = a.actualizar(hay, texto, ahora, 0.0 if inmediato else T_ACTIVAR)
        if cambio == "+":
            print(f"[requisitos] PROBLEMA: {texto}")
        elif cambio == "-":
            print(f"[requisitos] resuelto: {a.texto}")

    def actualizar(self, hz_robot=None):
        """Lista de textos de los problemas activos (para pintarlos en rojo)."""
        ahora = self.reloj()
        if ahora - self._t_ult < PERIODO:
            return self.lista
        self._t_ult = ahora
        varias = len(self.hilos) > 1
        arrancado = ahora - self.t0 > T_GRACIA
        fu = self.fusion

        # --- calibracion entre camaras ---
        if varias and not self.extrinsecas:
            b_ok = fu is not None and fu.valida[1] and fu.t_valida[1] and not fu.desalineada[1]
            motivo = f" ({self.motivo_extrinsecas})" if self.motivo_extrinsecas else ""
            if b_ok:
                texto = ("Camaras sin calibracion ChArUco: B situada sola (menos precisa). "
                         "Ejecuta: python calibrar_extrinseca.py")
            else:
                texto = (f"SIN calibracion ChArUco entre camaras{motivo}: B NO entra en la fusion hasta "
                         "calibrarse sola o con c (delante de las dos). Ejecuta: python calibrar_extrinseca.py")
            self._aviso("extrinsecas", True, texto, ahora, inmediato=True)
        else:
            self._aviso("extrinsecas", False, "", ahora)

        # --- por camara: fps y encuadre ---
        for i, h in enumerate(self.hilos):
            letra = ETIQUETAS[i]
            fps, fps_cam = float(getattr(h, "fps", 0.0)), float(getattr(h, "fps_camara", 0.0))
            self._aviso(f"fps{i}", arrancado and 0.0 < fps < FPS_SEG_MIN,
                        f"Camara {letra}: seguimiento a {fps:.0f} fps (< {FPS_SEG_MIN:.0f}): "
                        "las dos camaras en controladores USB distintos; o --modelo lite / --sin-hd", ahora)
            reproduccion = hasattr(h, "terminado")      # sesion grabada: no hay camara de verdad
            self._aviso(f"fpscam{i}", arrancado and not reproduccion and 0.0 < fps_cam < FPS_CAM_MIN,
                        f"Camara {letra}: llega a {fps_cam:.0f} fps (< {FPS_CAM_MIN:.0f}): cable/puerto USB 3", ahora)
            s = h.ultimo()
            if s is None or s.get("n") == self._n[i]:
                continue
            self._n[i] = s.get("n")
            img = s.get("bgr")
            hh, ww = img.shape[:2] if img is not None else (480, 640)
            e = encuadre(s.get("lm2d"), ww, hh)
            hist = [x for x in self._hist[i] if ahora - x[0] < VENTANA_ENCUADRE]
            hist.append((ahora, None if e is None else e["brazo_fuera"]))
            self._hist[i] = hist
            if e is None:
                continue
            self._aviso(f"cabeza{i}", e["cabeza_cortada"],
                        f"Camara {letra} te corta la CABEZA: alejate o sube/inclina la camara "
                        "(sin cara se confunden izquierda y derecha y la c no calibra)", ahora)
            fuera = [x[1] for x in hist if x[1] is not None]
            frac = len(fuera) / max(len(hist), 1)
            por = max(set(fuera), key=fuera.count) if fuera else ""
            self._aviso(f"brazos{i}", len(hist) >= 5 and frac > FRAC_FUERA,
                        f"Camara {letra}: se te salen los brazos del cuadro ({por}): alejate o abre el encuadre",
                        ahora)
            self._aviso(f"cerca{i}", e["ancho_hombros"] is not None and e["ancho_hombros"] > ANCHO_HOMBROS_MAX,
                        f"Camara {letra}: estas DEMASIADO CERCA: alejate", ahora)

        # --- bucle del robot ---
        if hz_robot is not None:
            self._aviso("robot", arrancado and 0.0 < hz_robot < HZ_ROBOT_MIN,
                        f"Bucle del robot a {hz_robot:.0f} Hz (< {HZ_ROBOT_MIN:.0f}): mira [rendimiento] en la consola",
                        ahora)
        self.lista = [a.texto for a in self._avisos.values() if a.activo]
        return self.lista

    def resumen(self):
        """Texto para la consola: estado de todos los requisitos (al arrancar)."""
        lineas = ["[requisitos] comprobacion:"]
        if len(self.hilos) > 1:
            lineas.append("  extrinsecas ChArUco: " + ("CARGADAS" if self.extrinsecas else
                                                         f"NO ({self.motivo_extrinsecas or 'sin archivo'})"))
        for i, h in enumerate(self.hilos):
            lineas.append(f"  camara {ETIQUETAS[i]}: camara {getattr(h, 'fps_camara', 0.0):.0f} fps, "
                          f"seguimiento {getattr(h, 'fps', 0.0):.1f} fps")
        lineas += [f"  PROBLEMA: {t}" for t in self.lista] or ["  sin problemas"]
        return "\n".join(lineas)
