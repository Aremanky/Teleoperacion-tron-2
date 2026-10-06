"""
Fuentes de imagen para la teleoperacion.

CamaraOAK    -> OAK-D con DepthAI v3: imagen RGB + profundidad alineada al RGB (mm) + copia HD
                del mismo fotograma (bgr_hd, 1280x960) para recortar las manos con mas pixeles
                y, si la camara tiene IMU, gravedad y deteccion de movimientos.
                En las OAK-D Pro enciende el proyector de puntos IR (estereo activo):
                da textura a la piel y a la ropa lisa y la profundidad de los brazos
                mejora mucho. Varias Pro a la vez no se estorban (cada una solo usa
                los puntos como textura) y el RGB no los ve, asi que MediaPipe tampoco
CamaraWebcam -> webcam normal (sin profundidad ni IMU), util para probar sin la OAK-D

Las dos devuelven en leer(): (imagen_bgr, profundidad_mm o None) y exponen
  K          matriz de intrinsecas 3x3 del RGB (None si no hay profundidad)
  t_captura  instante (reloj time.monotonic) en que se CAPTURO el ultimo frame
  imu        EstadoIMU o None

Con varias OAK-D conectadas, cada CamaraOAK se abre con el id de SU dispositivo
(CamaraOAK(mxid=...)). Para ver los ids:      python camaras.py --listar
Para comprobar la IMU de una camara:         python camaras.py --imu [ID]
"""
import threading
import time
from datetime import timedelta

import cv2
import numpy as np

# ------------------------------------------------------------------ IMU
G = 9.81
IMU_HZ = 100
UMBRAL_GIRO = 0.04     # rad/s (~2.3 deg/s): umbral MINIMO de giro (se sube si el sensor es ruidoso)
UMBRAL_ACC = 0.10      # m/s2: umbral MINIMO de aceleracion, diferencia VECTORIAL con la gravedad
                       # (el modulo no vale: un empujon horizontal casi no cambia |acc|)
T_ARRANQUE = 1.5       # s al arrancar para medir el sesgo del giroscopo y el ruido del sensor
                       # (la camara tiene que estar quieta; si no, se repite solo, ver T_MAX_MOVIENDO)
SIGMAS_UMBRAL = 5.0    # umbral = max(umbral minimo, este numero de veces el ruido medido)
SUAVIZADO = 0.2        # media movil exponencial de giro y aceleracion antes de comparar
T_MAX_MOVIENDO = 10.0  # s "moviendose" sin parar: seguramente es el sensor -> se recalibra
T_ASENTADA = 0.8       # s quieta para dar un movimiento por terminado
MIN_GIRO_MOV = 2.0     # deg girados para que cuente como "la han movido" (un golpecito no)
MIN_INCL_MOV = 1.5     # deg de cambio de inclinacion, idem
MIN_DESPL_MOV = 0.04   # m de desplazamiento estimado para contar como desplazada
GIRO_EXCLUIR = 1.5     # deg: girada mas que esto, la fusion deja de usarla hasta que pare
DESPL_EXCLUIR = 0.02   # m: desplazada mas que esto (ademas del error de integrar), idem
ALFA_LENTO = 0.02      # media lenta de la aceleracion (~0.5 s a 100 Hz): "quieta" = sin cambios
                       # respecto a ella, no respecto a la gravedad de antes (ver EstadoIMU)


def _unitario(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class EstadoIMU:
    """Gravedad y movimientos de UNA camara, a partir de acelerometro y giroscopo.

    La alimenta el hilo de la camara (muestra()) y la consulta la fusion desde el
    hilo principal, de ahi el lock. Solo DETECTA movimientos (no sigue la
    orientacion ni la posicion, que derivan): cuando la camara se mueve, la fusion
    la recalibra con el propio operador.

    Detecta giros (giroscopo) y desplazamientos sin giro (integrando la aceleracion
    sin gravedad dos veces SOLO durante el movimiento: lo justo para saber si la
    han desplazado unos centimetros). Un golpe a la camara o las vibraciones de
    alguien andando cerca no cuentan: van y vuelven, la camara acaba donde estaba.

    R: rotacion IMU -> camara RGB (de la calibracion de fabrica). Sin ella se sigue
    detectando el movimiento, pero no se puede dar la gravedad en el marco de la camara.

    (demo oct-2026) Con el tripode balanceandose +-0.25 grados porque alguien pasa al lado:
      - el giro se integraba como suma de |w|: el vaiven (y el ruido) se acumulaba sin
        parar (18 grados en 6 s) -> "girando" -> la fusion dejaba la camara fuera y al
        final contaba un movimiento falso que tiraba la calibracion. Ahora se integra el
        VECTOR de giro: un vaiven va y vuelve y suma ~0; un giro real si se acumula.
      - "quieta" se comparaba con la gravedad de referencia, y esa referencia se seguia
        actualizando en los instantes "quietos" DENTRO de la vibracion (sesgados por la
        fase): acababa desviada y la camara ya no volvia a estar quieta hasta pasados
        T_MAX_MOVIENDO (10 s fuera de la fusion). Ahora "quieta" = la aceleracion no cambia
        respecto a su media lenta; la gravedad solo se actualiza fuera de un movimiento y,
        al terminarlo, con la media de la ventana ya asentada.
      - el desplazamiento integrado dos veces deriva con el ruido: para dejar la camara
        fuera mientras se mueve se exige superar ademas ese error de integracion.
    """

    def __init__(self, R_imu_cam=None):
        self.R = None if R_imu_cam is None else np.asarray(R_imu_cam, dtype=float)
        self._lock = threading.Lock()
        self._g = None               # gravedad filtrada (vector, m/s2) en el marco de la IMU
        self._g_lento = None         # idem, mas lenta: referencia para integrar durante un movimiento
        self._sesgo = np.zeros(3)    # sesgo del giroscopo, se estima con la camara quieta
        self._t = None
        self._t_activo = -1e9        # ultimo instante en que NO estaba quieta
        self._moviendo = False
        self._giro = 0.0             # deg girados en el movimiento en curso
        self._vel = np.zeros(3)      # velocidad y desplazamiento estimados en el movimiento en curso
        self._despl = np.zeros(3)
        self._despl_max = 0.0
        self._t_ini = 0.0
        self._v_log = []             # (t, velocidad) del movimiento en curso
        self._g_ini = None
        self._arranque = []          # (acc, gyro) del arranque, para medir sesgo y ruido
        self._t_arranque = None
        self._listo = False
        self._umbral_giro, self._umbral_acc = UMBRAL_GIRO, UMBRAL_ACC
        self._w_f = np.zeros(3)      # giro y aceleracion suavizados
        self._a_f = None
        self._a_lento = None         # media lenta de la aceleracion (referencia de "quieta")
        self._rotv = np.zeros(3)     # vector de giro integrado en el movimiento en curso (rad)
        self._suma_asentada = np.zeros(3)   # aceleracion acumulada desde que dejo de moverse
        self._n_asentada = 0
        self.reinicios = 0           # veces que se ha recalibrado por creerse en movimiento sin parar
        self.movimientos = 0         # cuantas veces la han movido "de verdad"
        self.ultimo_mov = None       # dict(giro, incl, despl, rotacion) del ultimo movimiento contado
        self.muestras = 0

    def muestra(self, acc, gyro, t):
        acc, gyro = np.asarray(acc, dtype=float), np.asarray(gyro, dtype=float)
        if np.linalg.norm(acc) < 1e-3:
            return
        with self._lock:
            dt = 0.0 if self._t is None else float(np.clip(t - self._t, 0.0, 0.05))
            self._t = t
            self.muestras += 1
            if not self._listo:
                self._arrancar(acc, gyro, t)
                return
            w = gyro - self._sesgo
            self._w_f += SUAVIZADO * (w - self._w_f)
            self._a_f += SUAVIZADO * (acc - self._a_f)
            self._a_lento += ALFA_LENTO * (acc - self._a_lento)
            quieta = (float(np.linalg.norm(self._w_f)) < self._umbral_giro
                      and float(np.linalg.norm(self._a_f - self._a_lento)) < self._umbral_acc)
            if self._moviendo and t - self._t_ini > T_MAX_MOVIENDO:
                # Diez segundos "moviendose" sin parar: casi seguro es el sensor (sesgo que
                # ha cambiado, ruido mayor del medido). Se vuelve a medir, sin contar movimiento.
                self._moviendo, self._listo, self._arranque = False, False, []
                self.reinicios += 1
                return
            if quieta:
                self._suma_asentada += acc
                self._n_asentada += 1
                if not self._moviendo:
                    # solo con la camara asentada (fuera de un movimiento): dentro de una
                    # vibracion los instantes "quietos" estan sesgados por la fase
                    self._sesgo += 0.002 * (gyro - self._sesgo)
                    self._g += 0.05 * (acc - self._g)     # en reposo el acelerometro mide +g hacia arriba
                    self._g_lento += 0.005 * (acc - self._g_lento)
            else:
                self._t_activo = t
                self._suma_asentada, self._n_asentada = np.zeros(3), 0
                if not self._moviendo:
                    self._moviendo, self._giro = True, 0.0
                    self._rotv = np.zeros(3)
                    self._vel, self._despl, self._despl_max = np.zeros(3), np.zeros(3), 0.0
                    self._t_ini, self._v_log = t, []
                    self._g_ini = self._g_lento.copy()
            if self._moviendo:
                self._rotv += w * dt          # vector (aprox. angulos pequenos): el vaiven se anula
                self._giro = float(np.degrees(np.linalg.norm(self._rotv)))
                self._vel += (acc - self._g_ini) * dt
                self._despl += self._vel * dt
                self._despl_max = max(self._despl_max, float(np.linalg.norm(self._despl)))
                if len(self._v_log) < 3000:
                    self._v_log.append((t - self._t_ini, self._vel.copy()))
                if t - self._t_activo > T_ASENTADA:
                    self._moviendo = False
                    if self._n_asentada > 0:      # gravedad nueva: media de la ventana ya asentada
                        self._g = self._suma_asentada / self._n_asentada
                    self._g_lento = self._g.copy()
                    c = np.dot(_unitario(self._g_ini), _unitario(self._g))
                    incl = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
                    rotacion = self._giro > MIN_GIRO_MOV or incl > MIN_INCL_MOV
                    despl = self._desplazamiento_corregido(t - self._t_ini)
                    if rotacion or despl > MIN_DESPL_MOV:
                        self.movimientos += 1
                        self.ultimo_mov = dict(giro=self._giro, incl=incl, despl=despl, rotacion=rotacion)

    def _arrancar(self, acc, gyro, t):
        """Primeros T_ARRANQUE s: mide el sesgo del giroscopo, la gravedad y el ruido del
        sensor, y ajusta los umbrales a ESTE sensor. Si la camara se estaba moviendo, el
        ruido medido sale enorme: se descarta y se vuelve a medir."""
        if self._t_arranque is None or not self._arranque:
            self._t_arranque = t
        self._arranque.append((acc, gyro))
        if t - self._t_arranque < T_ARRANQUE or len(self._arranque) < 20:
            return
        A = np.array([a for a, _ in self._arranque])
        Gy = np.array([g for _, g in self._arranque])
        self._arranque = []
        ruido_giro = float(np.linalg.norm(Gy.std(axis=0)))
        ruido_acc = float(np.linalg.norm(A.std(axis=0)))
        if ruido_giro > 0.1 or ruido_acc > 1.0:   # no estaba quieta: otra vez
            return
        k = SIGMAS_UMBRAL * np.sqrt(SUAVIZADO / (2 - SUAVIZADO))   # el suavizado reduce el ruido
        self._sesgo = np.median(Gy, axis=0)
        self._g = np.median(A, axis=0)
        if self._g_lento is None or not self._listo:
            self._g_lento = self._g.copy()
        self._umbral_giro = max(UMBRAL_GIRO, k * ruido_giro)
        self._umbral_acc = max(UMBRAL_ACC, k * ruido_acc)
        self._w_f, self._a_f, self._a_lento = np.zeros(3), self._g.copy(), self._g.copy()
        self._suma_asentada, self._n_asentada = np.zeros(3), 0
        self._listo = True

    def umbrales(self):
        """(umbral de giro en deg/s, umbral de aceleracion en m/s2), para diagnostico."""
        return float(np.degrees(self._umbral_giro)), self._umbral_acc

    def _desplazamiento_corregido(self, T):
        """Donde ha acabado la camara (m), no el maximo. En el tramo final la camara ya
        esta quieta, asi que la velocidad estimada ahi es error de integracion: un
        error constante (haber empezado en mitad de una vibracion) mas uno que crece
        (gravedad de referencia algo desviada). Se ajusta una recta v = c + e*t a ese
        tramo y se descuenta su efecto sobre el desplazamiento: c*T + e*T^2/2."""
        cola = [(tt, v) for tt, v in self._v_log if tt >= T - T_ASENTADA]
        if len(cola) < 10:
            return float(np.linalg.norm(self._despl))
        ts = np.array([c[0] for c in cola])
        V = np.array([c[1] for c in cola])
        A = np.column_stack([np.ones_like(ts), ts])
        (c, e), *_ = np.linalg.lstsq(A, V, rcond=None)
        return float(np.linalg.norm(self._despl - (c * T + 0.5 * e * T * T)))

    def arriba(self):
        """Direccion 'arriba' (unitaria) en el marco de la camara RGB, o None."""
        with self._lock:
            if self._g is None or self.R is None or not self._listo or (self._moviendo and self._giro > GIRO_EXCLUIR):
                return None
            return _unitario(self.R @ self._g)

    def girando(self):
        """True mientras la estan girando de forma apreciable."""
        with self._lock:
            return self._listo and self._moviendo and self._giro > GIRO_EXCLUIR

    def desplazandose(self):
        """True mientras la estan desplazando de forma apreciable (aunque no gire): mas de
        DESPL_EXCLUIR por encima de lo que deriva la doble integracion con el ruido del
        acelerometro (~ umbral * T^2 / 2)."""
        with self._lock:
            if not (self._listo and self._moviendo):
                return False
            T = (self._t or 0.0) - self._t_ini
            return self._despl_max > DESPL_EXCLUIR + 0.5 * self._umbral_acc * T * T

    def en_movimiento(self):
        with self._lock:
            return self._moviendo


# ------------------------------------------------------------------ OAK-D
def _id_dispositivo(info):
    """Id del dispositivo. El nombre del metodo cambia entre versiones de DepthAI."""
    for nombre in ("getDeviceId", "getMxId"):
        f = getattr(info, nombre, None)
        if callable(f):
            return str(f())
    for nombre in ("deviceId", "mxid"):
        if hasattr(info, nombre):
            return str(getattr(info, nombre))
    return repr(info)


def listar_oak():
    """[(id, info)] de las OAK conectadas y libres."""
    import depthai as dai
    return [(_id_dispositivo(i), i) for i in dai.Device.getAllAvailableDevices()]


def _cola(salida, max_size):
    try:
        return salida.createOutputQueue(maxSize=max_size, blocking=False)
    except TypeError:
        return salida.createOutputQueue()


class CamaraOAK:
    def __init__(self, ancho=640, alto=480, fps=30, mxid=None, usar_imu=True, proyector=0.7,
                 estereo_fino=True, hd=True, ancho_hd=1280, alto_hd=960):
        """proyector: intensidad 0..1 del proyector de puntos IR (solo modelos Pro). 0 = apagado.
        estereo_fino: subpixel + comprobacion izquierda/derecha en el propio estereo
        (profundidad mucho menos ruidosa, que es lo que limita la precision hacia delante).
        hd: ademas del fotograma de ancho x alto (MediaPipe Pose, profundidad, ventana), pide
        al ISP el MISMO fotograma a ancho_hd x alto_hd, sincronizado, para recortar las manos
        con mas pixeles (self.bgr_hd). Misma relacion de aspecto y misma correccion de
        distorsion: la imagen HD es la de 640 escalada x2. Si la camara no lo admite
        (USB2, ISP sin recursos) se reintenta sin HD."""
        import depthai as dai  # DepthAI v3  (pip install "depthai>=3")

        self.dai = dai
        self.K = None
        self.mxid = mxid
        self.t_captura = None
        self.bgr_hd = None
        self.hd = bool(hd)
        self.imu, self.cola_imu = None, None
        self._args = dict(ancho=ancho, alto=alto, fps=fps, usar_imu=usar_imu, proyector=proyector,
                          estereo_fino=estereo_fino, ancho_hd=ancho_hd, alto_hd=alto_hd)
        try:
            self._construir(**self._args, hd=self.hd)
        except Exception as e:
            if not self.hd:
                raise
            print(f"[{mxid or 'OAK'}] no puedo abrir la camara con la salida HD ({e}): reintento sin HD")
            self.hd = False
            try:
                if getattr(self, "pipeline", None) is not None:
                    self.pipeline.stop()
            except Exception:
                pass
            self._construir(**self._args, hd=False)

    def _construir(self, ancho, alto, fps, usar_imu, proyector, estereo_fino, ancho_hd, alto_hd, hd):
        dai, mxid = self.dai, self.mxid
        if mxid is None:   # comportamiento de siempre: la primera OAK libre
            self.pipeline = dai.Pipeline()
        else:              # una OAK concreta (necesario con dos camaras)
            info = next((i for ident, i in listar_oak() if ident == str(mxid)), None)
            if info is None:
                libres = [ident for ident, _ in listar_oak()]
                raise RuntimeError(f"No encuentro la OAK '{mxid}'. Libres ahora: {libres}")
            self.pipeline = dai.Pipeline(dai.Device(info))
        dispositivo = self.pipeline.getDefaultDevice()
        plataforma = dispositivo.getPlatform()

        cam_rgb = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_izq = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        cam_der = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        if estereo_fino:
            self._afinar_estereo(stereo, mxid or "OAK")
        sync = self.pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(seconds=1 / (2 * fps)))

        salida_rgb = cam_rgb.requestOutput(size=(ancho, alto), fps=fps, enableUndistortion=True)
        cam_izq.requestOutput(size=(640, 400), fps=fps).link(stereo.left)
        cam_der.requestOutput(size=(640, 400), fps=fps).link(stereo.right)
        salida_rgb.link(sync.inputs["rgb"])
        if hd:
            salida_hd = cam_rgb.requestOutput(size=(ancho_hd, alto_hd), fps=fps, enableUndistortion=True)
            salida_hd.link(sync.inputs["rgb_hd"])

        if plataforma == dai.Platform.RVC4:
            alinear = self.pipeline.create(dai.node.ImageAlign)
            stereo.depth.link(alinear.input)
            salida_rgb.link(alinear.inputAlignTo)
            alinear.outputAligned.link(sync.inputs["depth"])
        else:  # OAK-D / OAK-D Lite / OAK-D Pro (RVC2): alinea el propio StereoDepth
            stereo.depth.link(sync.inputs["depth"])
            salida_rgb.link(stereo.inputAlignTo)

        # Cola de 1 elemento y no bloqueante: siempre el frame mas reciente (menos latencia)
        self.cola = _cola(sync.out, 1)
        if usar_imu:
            self._crear_imu(dispositivo)
        self.pipeline.start()
        if hd:   # comprobar que de verdad llegan los tres fotogramas
            grupo = self.cola.get()
            try:
                hd_ok = grupo is not None and grupo["rgb_hd"] is not None
            except Exception:
                hd_ok = False
            if not hd_ok:
                raise RuntimeError("el Sync no entrega la salida HD")
            print(f"[{mxid or 'OAK'}] salida HD {ancho_hd}x{alto_hd} activa para las manos")
        if proyector > 0:
            try:   # en DepthAI v3 se fija con el pipeline ya arrancado
                dispositivo.setIrLaserDotProjectorIntensity(float(proyector))
                print(f"[{mxid or 'OAK'}] proyector IR al {proyector * 100:.0f} %")
            except Exception as e:
                print(f"[{mxid or 'OAK'}] sin proyector IR ({e}): profundidad pasiva")

    def _afinar_estereo(self, stereo, nombre):
        """Cada ajuste va por separado y protegido: segun la version de DepthAI alguno puede
        no existir, y en ese caso se sigue con el resto (se avisa por consola)."""
        dai = self.dai
        ajustes = [
            ("subpixel", lambda: stereo.setSubpixel(True)),
            ("comprobacion izq/der", lambda: stereo.setLeftRightCheck(True)),
            ("disparidad extendida desactivada", lambda: stereo.setExtendedDisparity(False)),
        ]
        puestos, fallos = [], []
        for etiqueta, f in ajustes:
            try:
                f()
                puestos.append(etiqueta)
            except Exception as e:
                fallos.append(f"{etiqueta} ({type(e).__name__})")
        print(f"[{nombre}] estereo fino: {', '.join(puestos) or 'nada'}"
              + (f" | no disponibles: {', '.join(fallos)}" if fallos else ""))

    # ------------------------------------------------------------ IMU
    def _crear_imu(self, dispositivo):
        dai = self.dai
        nombre = self.mxid or "OAK"
        try:
            tipo = str(dispositivo.getConnectedIMU()) if hasattr(dispositivo, "getConnectedIMU") else "?"
            if tipo.strip().upper() in ("", "NONE"):
                raise RuntimeError("esta camara no lleva IMU")
            nodo = self.pipeline.create(dai.node.IMU)
            nodo.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], IMU_HZ)
            nodo.setBatchReportThreshold(5)
            nodo.setMaxBatchReports(20)
            self.cola_imu = _cola(nodo.out, 50)
        except Exception as e:
            print(f"[{nombre}] IMU no disponible ({e}): sin gravedad ni deteccion de movimientos")
            return
        R = None
        try:
            calib = dispositivo.readCalibration()
            T = np.array(calib.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_A, True), dtype=float)
            R = T[:3, :3]
            if abs(np.linalg.det(R) - 1.0) > 0.05:
                raise RuntimeError(f"rotacion no valida (det={np.linalg.det(R):.2f})")
        except Exception as e:
            print(f"[{nombre}] IMU {tipo} sin extrinsecas IMU->RGB ({e}): "
                  "solo se usara para detectar movimientos, no la gravedad")
            R = None
        print(f"[{nombre}] IMU {tipo} activa")
        self.imu = EstadoIMU(R)

    def _leer_imu(self):
        if self.cola_imu is None:
            return
        try:
            while True:
                datos = self.cola_imu.tryGet()
                if datos is None:
                    break
                for p in datos.packets:
                    a, g = p.acceleroMeter, p.gyroscope
                    t = a.getTimestamp().total_seconds()   # solo se usan diferencias de tiempo
                    self.imu.muestra((a.x, a.y, a.z), (g.x, g.y, g.z), t)
        except Exception as e:
            print(f"[{self.mxid or 'OAK'}] Error leyendo la IMU ({e}): se desactiva")
            self.cola_imu, self.imu = None, None

    # ------------------------------------------------------------ imagen
    def _instante_captura(self, frame):
        """Instante de captura en el reloj de time.monotonic(). DepthAI sella cada
        frame con un reloj sincronizado con el del PC; se mide cuanto hace de eso
        (latencia) y se resta al reloj de Python, asi no hay que suponer que los
        dos relojes son el mismo (en Windows no lo son)."""
        try:
            latencia = (self.dai.Clock.now() - frame.getTimestamp()).total_seconds()
            return time.monotonic() - float(np.clip(latencia, 0.0, 0.5))
        except Exception:
            return time.monotonic()

    def leer(self):
        grupo = self.cola.get()
        f_rgb, f_depth = grupo["rgb"], grupo["depth"]
        self.t_captura = self._instante_captura(f_rgb)
        bgr = f_rgb.getCvFrame()
        depth = f_depth.getFrame()
        if self.hd:
            try:
                self.bgr_hd = grupo["rgb_hd"].getCvFrame()
            except Exception:
                self.bgr_hd = None
        if depth.shape[:2] != bgr.shape[:2]:
            depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.K is None:
            self.K = self._intrinsecas(f_rgb, bgr.shape[1], bgr.shape[0])
        self._leer_imu()
        return bgr, depth

    def _intrinsecas(self, f_rgb, ancho, alto):
        try:  # intrinsecas de la imagen tal y como sale (tras escalado y recorte)
            K = np.array(f_rgb.getTransformation().getIntrinsicMatrix(), dtype=float)
        except Exception:
            calib = self.pipeline.getDefaultDevice().readCalibration()
            K = np.array(calib.getCameraIntrinsics(self.dai.CameraBoardSocket.CAM_A, ancho, alto), dtype=float)
        print("Intrinsecas del RGB:\n", np.round(K, 1))
        return K

    def cerrar(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class CamaraWebcam:
    def __init__(self, indice=0, ancho=640, alto=480):
        self.cap = cv2.VideoCapture(indice)
        if not self.cap.isOpened():
            raise RuntimeError(f"No puedo abrir la webcam {indice}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ancho, self.alto = ancho, alto
        self.K = None
        self.imu = None
        self.t_captura = None
        self.bgr_hd = None   # si la webcam da mas de 640x480, la imagen completa hace de HD

    def leer(self):
        ok, bgr = self.cap.read()
        self.t_captura = time.monotonic()
        if not ok:
            return None, None
        if bgr.shape[1] > self.ancho:
            self.bgr_hd = bgr
            bgr = cv2.resize(bgr, (self.ancho, self.alto), interpolation=cv2.INTER_AREA)
        return bgr, None

    def cerrar(self):
        self.cap.release()


if __name__ == "__main__":
    import sys
    if "--listar" in sys.argv:
        encontradas = listar_oak()
        print(f"{len(encontradas)} OAK libre(s):")
        for ident, info in encontradas:
            print("  ", ident, "|", getattr(info, "state", "?"))
    elif "--imu" in sys.argv:
        # Comprobacion de la IMU (y de sus extrinsecas). Marco de la camara: x derecha,
        # y abajo, z hacia delante. Con la camara HORIZONTAL mirando al frente,
        # 'arriba' tiene que salir cerca de (0, -1, 0). Inclinandola hacia el suelo
        # debe aparecer una z NEGATIVA; girandola de lado (roll), una x.
        # Si sale otra cosa (p. ej. (1, 0, 0) en horizontal), las extrinsecas no cuadran.
        resto = [a for a in sys.argv[1:] if a != "--imu"]
        cam = CamaraOAK(mxid=resto[0] if resto else None)
        try:
            t_ant = 0.0
            while True:
                cam.leer()
                if cam.imu is None:
                    print("Sin IMU")
                    break
                if time.monotonic() - t_ant > 0.5:
                    t_ant = time.monotonic()
                    a = cam.imu.arriba()
                    print("arriba (camara):", None if a is None else np.round(a, 3),
                          "| movimientos:", cam.imu.movimientos,
                          "| girando" if cam.imu.girando() else "",
                          "| desplazandose" if cam.imu.desplazandose() else "",
                          "| ultimo:", cam.imu.ultimo_mov,
                          "| umbrales (deg/s, m/s2):", np.round(cam.imu.umbrales(), 3),
                          "| recalibraciones:", cam.imu.reinicios)
        except KeyboardInterrupt:
            pass
        finally:
            cam.cerrar()
    else:
        print("Uso: python camaras.py --listar | --imu [ID]")