"""
Grabar y reproducir sesiones de teleoperacion SIN camaras.

  python teleop_tron2.py --camara2 --orca --grabar sesion.pkl
      guarda, por camara y fotograma, TODO lo que sale de MediaPipe (esqueleto 2D, puntos 3D
      con profundidad, direcciones, manos con sus 21 pixeles y world landmarks, K, IMU) mas
      los eventos del teleop (calibracion con [c], pausas). Sin imagenes: ~30 KB/s por camara.
      Con --grabar-video se guardan tambien los fotogramas en JPEG (~1 MB/s por camara).

  python teleop_tron2.py --reproducir sesion.pkl [--velocidad 1.0]
      vuelve a pasar la sesion por la fusion y el robot como si las camaras estuvieran
      conectadas: se puede cambiar cualquier parametro de fusion.py / robot_tron2.py y ver
      el efecto sobre EL MISMO movimiento del operador. La calibracion grabada se reaplica
      sola en su instante (no hace falta pulsar c).

  python metricas_imitacion.py sesion.pkl
      lo mismo, sin visor y a toda velocidad, midiendo lo bien que el robot imita.

Los HiloReproduccion imitan la interfaz de seguimiento_brazos.HiloSeguimiento (ultimo(),
ultimo_crudo(), fps, fps_camara, error, camara.imu) para que fusion.FusionCamaras y el
teleop no se enteren de que no hay camaras.
"""
import pickle
import threading
import time

import cv2
import numpy as np

import manos3d

VERSION = 2


def _landmarks_a_array(lms):
    if lms is None:
        return None
    lista = getattr(lms, "landmark", lms)
    return np.array([[l.x, l.y, l.z] for l in lista], dtype=np.float32)


def _ligero(resultado, con_video=False, calidad_jpeg=80):
    """Copia serializable y ligera de un resultado de HiloSeguimiento."""
    r = {k: resultado[k] for k in ("n", "t", "t_proc", "brazos", "puntos", "lm2d", "linea_hombros")
         if k in resultado}
    r["K"] = None if resultado.get("K") is None else np.asarray(resultado["K"], dtype=np.float64)
    r["arriba"] = None if resultado.get("arriba") is None else np.asarray(resultado["arriba"], dtype=np.float64)
    aperturas = {}
    for lado, m in (resultado.get("aperturas") or {}).items():
        mm = {k: m.get(k) for k in ("apertura", "marco", "px", "z_palma", "tam_px", "calidad", "confianza")}
        mm["mundo"] = _landmarks_a_array(m.get("mundo"))
        aperturas[lado] = mm
    r["aperturas"] = aperturas
    r["manos"] = [{"lado": m["lado"], "apertura": m["apertura"], "bruto": m.get("bruto"),
                   "puntos_px": np.asarray(m["puntos_px"], dtype=np.float32)} for m in (resultado.get("manos") or [])]
    if con_video and resultado.get("bgr") is not None:
        ok, buf = cv2.imencode(".jpg", resultado["bgr"], [int(cv2.IMWRITE_JPEG_QUALITY), calidad_jpeg])
        r["jpeg"] = buf.tobytes() if ok else None
        r["forma"] = resultado["bgr"].shape
    return r


class Grabadora:
    """Acumula los resultados de cada camara y los eventos del teleop; guardar() los escribe."""

    def __init__(self, ruta, con_video=False, meta=None):
        self.ruta = ruta
        self.con_video = con_video
        self.meta = dict(meta or {})
        self.registros = {}       # etiqueta -> [resultado ligero]
        self.eventos = []         # (t, nombre, datos)
        self._lock = threading.Lock()
        self.t0 = time.monotonic()

    def anotar(self, etiqueta, resultado):
        r = _ligero(resultado, self.con_video)
        with self._lock:
            self.registros.setdefault(etiqueta, []).append(r)

    def evento(self, nombre, **datos):
        with self._lock:
            self.eventos.append((time.monotonic(), nombre, datos))

    def guardar(self):
        with self._lock:
            datos = dict(version=VERSION, meta=self.meta, t0=self.t0, registros=self.registros,
                         eventos=self.eventos, con_video=self.con_video)
        with open(self.ruta, "wb") as f:
            pickle.dump(datos, f, protocol=pickle.HIGHEST_PROTOCOL)
        n = {k: len(v) for k, v in self.registros.items()}
        print(f"[grabacion] guardado {self.ruta}: fotogramas por camara {n}, {len(self.eventos)} eventos")


def cargar(ruta):
    with open(ruta, "rb") as f:
        return pickle.load(f)


class _ImuReproducida:
    """IMU de mentira: da el 'arriba' grabado y nunca se mueve."""

    def __init__(self):
        self.R = np.eye(3)
        self._arriba = None
        self.movimientos = 0
        self.ultimo_mov = None
        self._listo = True

    def arriba(self):
        return None if self._arriba is None else self._arriba.copy()

    def girando(self):
        return False

    def desplazandose(self):
        return False

    def en_movimiento(self):
        return False


class _CamaraReproducida:
    def __init__(self, K, con_imu):
        self.K = K
        self.imu = _ImuReproducida() if con_imu else None
        self.mxid = "grabada"


class HiloReproduccion:
    """Sirve los resultados grabados de UNA camara al ritmo del reloj (o mas deprisa)."""

    def __init__(self, etiqueta, registros, t0_grabacion, reloj=time.monotonic, velocidad=1.0,
                 t_inicio=None):
        self.etiqueta = etiqueta
        self.reg = registros
        self.reloj = reloj
        self.velocidad = velocidad
        self.t0 = t0_grabacion
        self.t_inicio = reloj() if t_inicio is None else t_inicio
        self.error = None
        self.fps, self.fps_camara = 0.0, 0.0
        self._i = 0                # siguiente registro por servir
        self._ultimo = None
        self._crudo = None
        con_imu = any(r.get("arriba") is not None for r in registros[:50])
        K = next((r["K"] for r in registros if r.get("K") is not None), None)
        self.camara = _CamaraReproducida(K, con_imu)
        self.activo = True
        self._fps_hist = []

    def _ahora_rel(self):
        return (self.reloj() - self.t_inicio) * self.velocidad

    def _desplazar(self, t):
        """Instante grabado -> instante en el reloj actual."""
        return self.t_inicio + (t - self.t0) / self.velocidad

    def start(self):
        pass

    def terminado(self):
        return self._i >= len(self.reg)

    def _avanzar(self):
        rel = self._ahora_rel()
        avanzado = False
        while self._i < len(self.reg) and self.reg[self._i]["t_proc"] - self.t0 <= rel:
            r = self.reg[self._i]
            self._i += 1
            avanzado = True
            self._ultimo = self._reconstruir(r)
        if avanzado and self._i >= 2:
            dt = (self.reg[self._i - 1]["t_proc"] - self.reg[max(0, self._i - 11)]["t_proc"]) / min(10, self._i - 1)
            self.fps = self.fps_camara = (1.0 / dt) if dt > 1e-6 else 0.0

    def _reconstruir(self, r):
        s = dict(r)
        s["t"] = self._desplazar(r["t"])
        s["t_proc"] = self._desplazar(r["t_proc"])
        aperturas = {}
        for lado, m in (r.get("aperturas") or {}).items():
            mm = dict(m)
            mm["mundo"] = None if m.get("mundo") is None else manos3d.a_puntos(m["mundo"])
            aperturas[lado] = mm
        s["aperturas"] = aperturas
        if r.get("jpeg") is not None:
            s["bgr"] = cv2.imdecode(np.frombuffer(r["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            h, w = (480, 640) if r.get("forma") is None else r["forma"][:2]
            s["bgr"] = np.zeros((h, w, 3), dtype=np.uint8)
        s["depth"] = None
        if self.camara.imu is not None:
            self.camara.imu._arriba = r.get("arriba")
        return s

    def ultimo(self):
        self._avanzar()
        return self._ultimo

    def ultimo_crudo(self):
        self._avanzar()
        if self._ultimo is None:
            return None
        return dict(n=self._ultimo["n"], bgr=self._ultimo["bgr"], depth=None, t=self._ultimo["t"])

    def parar(self):
        self.activo = False


def hilos_reproduccion(datos, reloj=time.monotonic, velocidad=1.0):
    """[HiloReproduccion] por camara (en el orden A, B...) y la lista de eventos con sus
    instantes ya pasados al reloj actual: [(t_reloj, nombre, datos)]."""
    etiquetas = sorted(datos["registros"])
    t_inicio = reloj()
    hilos = [HiloReproduccion(e, datos["registros"][e], datos["t0"], reloj, velocidad, t_inicio) for e in etiquetas]
    eventos = [(t_inicio + (t - datos["t0"]) / velocidad, nombre, d) for t, nombre, d in datos["eventos"]]
    return hilos, eventos
