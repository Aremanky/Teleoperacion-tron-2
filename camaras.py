"""
Fuentes de imagen para la teleoperacion.

CamaraOAK    -> OAK-D con DepthAI v3: imagen RGB + profundidad alineada al RGB (mm)
CamaraWebcam -> webcam normal (sin profundidad), util para probar sin la OAK-D

Las dos devuelven en leer(): (imagen_bgr, profundidad_mm o None) y exponen
K (matriz de intrinsecas 3x3 del RGB, o None si no hay profundidad).
"""
from datetime import timedelta

import cv2
import numpy as np


class CamaraOAK:
    def __init__(self, ancho=640, alto=480, fps=30):
        import depthai as dai  # DepthAI v3  (pip install "depthai>=3")

        self.dai = dai
        self.K = None
        self.pipeline = dai.Pipeline()
        plataforma = self.pipeline.getDefaultDevice().getPlatform()

        cam_rgb = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_izq = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        cam_der = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        sync = self.pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(seconds=1 / (2 * fps)))

        salida_rgb = cam_rgb.requestOutput(size=(ancho, alto), fps=fps, enableUndistortion=True)
        cam_izq.requestOutput(size=(640, 400), fps=fps).link(stereo.left)
        cam_der.requestOutput(size=(640, 400), fps=fps).link(stereo.right)
        salida_rgb.link(sync.inputs["rgb"])

        if plataforma == dai.Platform.RVC4:
            alinear = self.pipeline.create(dai.node.ImageAlign)
            stereo.depth.link(alinear.input)
            salida_rgb.link(alinear.inputAlignTo)
            alinear.outputAligned.link(sync.inputs["depth"])
        else:  # OAK-D / OAK-D Lite / OAK-D Pro (RVC2): alinea el propio StereoDepth
            stereo.depth.link(sync.inputs["depth"])
            salida_rgb.link(stereo.inputAlignTo)

        # Cola de 1 elemento y no bloqueante: siempre el frame mas reciente (menos latencia)
        try:
            self.cola = sync.out.createOutputQueue(maxSize=1, blocking=False)
        except TypeError:
            self.cola = sync.out.createOutputQueue()
        self.pipeline.start()

    def leer(self):
        grupo = self.cola.get()
        f_rgb, f_depth = grupo["rgb"], grupo["depth"]
        bgr = f_rgb.getCvFrame()
        depth = f_depth.getFrame()
        if depth.shape[:2] != bgr.shape[:2]:
            depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.K is None:
            self.K = self._intrinsecas(f_rgb, bgr.shape[1], bgr.shape[0])
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
    def __init__(self, indice=0):
        self.cap = cv2.VideoCapture(indice)
        if not self.cap.isOpened():
            raise RuntimeError(f"No puedo abrir la webcam {indice}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.K = None

    def leer(self):
        ok, bgr = self.cap.read()
        return (bgr if ok else None), None

    def cerrar(self):
        self.cap.release()