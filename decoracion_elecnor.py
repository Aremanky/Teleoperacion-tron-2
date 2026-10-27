"""
Decoracion corporativa ELECNOR de los escenarios del TRON 2 (solo estetica: no hay colisiones
ni masa, y la logica de la teleoperacion y del escenario de tuberias no cambia).

DOS PARTES
  decorar_base(root)          -> SIEMPRE a la vista (grupo GRUPO_BASE = 2, visible por defecto):
      - suelo de resina antracita-azul en losetas de 1 m (sustituye al damero)
      - zona del robot: disco azul Elecnor con aro naranja, logo en el suelo delante del robot
        y cinta naranja que delimita la celda de trabajo
      - sala tipo stand: paredes gris claro con zocalo azul, filete naranja y cornisa azul;
        paneles con el logo de Elecnor y el de I+D+i en las paredes de delante y de detras
        (a la altura que se ve con la camara del visor) y rotulo del proyecto
      - cuatro roll-ups corporativos (dos para cada vista del visor)
      - con el KUKA: placa de anclaje de acero y franjas de seguridad amarillo/negro alrededor
  decorar_tuberias(root, esc) -> en el grupo del escenario (GRUPO_ESCENARIO = 5): solo se ve
     al activar el escenario (tecla 1), como el resto de las tuberias:
      - techo de placas registrables de 60 x 60 con paneles LED
      - faldon (tabica) azul Elecnor con filete naranja y logos alrededor del techo
      - atril de senalizacion "ZONA DE MONTAJE" (dos caras)
      - carro de herramientas azul Elecnor con tiradores naranjas y una caja de accesorios
      - pale con tubos de PVC
      - dos conos de seguridad

USO (teleop_tron2.crear_robot):
    import decoracion_elecnor
    robot = RobotTron2(xml, decoracion_elecnor.con_decoracion(escenario_tuberias.anadir_al_modelo), ...)
  con_decoracion() decora la base, monta el escenario y despues lo decora. Con el KUKA,
  RobotTron2 lo envuelve con montar_kuka.con_kuka(), asi que el KUKA ya esta montado cuando se
  decora (por eso se puede saber si hay KUKA y donde).
  Desactivar sin tocar codigo:  TRON2_DECORACION=0 python teleop_tron2.py ...

TEXTURAS
  Se generan la primera vez en decoracion/generadas/ (necesita Pillow y numpy) a partir de
  decoracion/Logo_Elecnor_AZUL.png y decoracion/I+D+I_DENOMIN_GRUPO_RGB.png. Los logos se
  recortan solos (se quita el margen blanco/transparente) y se respeta su proporcion. Si
  cambias un logo, borra decoracion/generadas/ y se regeneran. Sin Pillow todo sale en colores
  lisos y sin logos (se avisa al arrancar).
  Los logos y carteles van en MALLAS con sus coordenadas de textura (como el tablero de la
  mesa), no en cajas: asi su orientacion no depende de como MuJoCo reparte la textura por las
  caras de una caja. Si en tu version de MuJoCo salieran boca abajo, pon INVERTIR_V = True.

PRUEBA RAPIDA (sin camaras): abre el visor con todo a la vista
    python decoracion_elecnor.py              # soporte segun TRON2_SOPORTE (por defecto KUKA)
    TRON2_SOPORTE=columna python decoracion_elecnor.py

Medidas en metros, marco del mundo del robot: x hacia delante, y a su izquierda, z arriba.
"""
import os
import sys
import unicodedata
import xml.etree.ElementTree as ET

import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
CARPETA_DECORACION = os.path.join(RAIZ, "decoracion")
CARPETA_GENERADAS = os.path.join(CARPETA_DECORACION, "generadas")
LOGO_ELECNOR = "Logo_Elecnor_AZUL.png"
LOGO_IDI = "I+D+I_DENOMIN_GRUPO_RGB.png"
VERSION = 1                     # va en el nombre de las texturas: subirlo las regenera

ACTIVA = os.environ.get("TRON2_DECORACION", "1") != "0"
GRUPO_BASE = 2                  # grupo de visualizacion de la decoracion base (0-2 se ven por defecto)
INVERTIR_V = False              # True si los logos salen boca abajo en tu version de MuJoCo
REFLEJO_SUELO = 0.0             # 0.05-0.1 da un suelo brillante de showroom (cuesta otra pasada de render)

# =============================== COLORES =====================================
AZUL = (0.000, 0.227, 0.561)            # azul Elecnor (el del robot)
AZUL_MEDIO = (0.106, 0.353, 0.651)
AZUL_OSCURO = (0.016, 0.098, 0.239)
NARANJA = (0.941, 0.494, 0.149)         # naranja Elecnor
BLANCO = (0.96, 0.96, 0.96)
PARED = (0.86, 0.87, 0.88)
ALUMINIO = (0.74, 0.76, 0.79)
ACERO = (0.30, 0.32, 0.35)
NEGRO = (0.10, 0.10, 0.11)
COLOR_SUELO = (0.165, 0.185, 0.225)     # resina antracita azulada
COLOR_TECHO = (0.93, 0.93, 0.91)
PVC = (0.62, 0.64, 0.66)
MADERA = (0.74, 0.57, 0.38)
MADERA_OSCURA = (0.58, 0.43, 0.27)
CARTON = (0.71, 0.55, 0.37)

# =============================== SALA (base) =================================
SALA_X = (-4.60, 3.40)          # pared de detras / pared de delante (el KUKA esta en x = -2)
SALA_Y = (-3.60, 3.60)          # pared derecha / izquierda del robot
ALTO_SALA = 3.50
ZOCALO = 0.60                   # zocalo azul
FILETE = 0.04                   # filete naranja sobre el zocalo
CORNISA = 0.25                  # banda azul arriba
Z_LOGOS = 1.15                  # centro de los paneles con logo. La camara del visor (a 1.8 m del
                                # robot, 25 grados hacia abajo) ve la pared del fondo solo hasta
                                # ~1.5 m: mas arriba no se verian sin girar la vista.
LATERAL_LOGOS = 1.70            # a cada lado del robot (en el centro los tapa el torso)
ANCHO_LOGO_ELECNOR = 2.10
ANCHO_LOGO_IDI = 1.70
ALTO_MAX_LOGO = 0.70
ROTULO = "TRON 2 · Teleoperación"   # bajo el logo de Elecnor de las paredes de delante y detras (None = sin rotulo)
LEMA_ROLLUP = "Innovación"          # texto de los roll-ups, entre los dos logos (None = sin texto)

# zona del robot (en el suelo, alrededor del torso)
R_ZONA = 0.86                   # disco azul
ANCHO_ARO = 0.045               # aro naranja
CELDA = (-1.00, 1.70, -1.40, 1.40)  # cinta naranja: x0, x1, y0, y1
ANCHO_CINTA = 0.05

# Las dos vistas del visor (teleop_tron2.colocar_camara, tecla v), en (x, y): con azimuth 180
# (la de arranque) la camara queda DELANTE del robot (x+) mirando hacia atras; con 0, detras.
CAMARA_FRENTE = (1.93, 0.0)
CAMARA_DETRAS = (-1.33, 0.0)
# roll-ups: ((x, y), punto al que miran). Dos al fondo para la vista de arranque y dos delante
# para la otra (cada pareja queda a la espalda de la otra camara)
ROLLUPS = [((-2.40, 2.65), CAMARA_FRENTE), ((-2.40, -2.65), CAMARA_FRENTE),
           ((2.45, 2.40), CAMARA_DETRAS), ((2.45, -2.40), CAMARA_DETRAS)]

# KUKA (solo si esta montado)
PLACA_KUKA = 0.60               # semilado de la placa de anclaje
ZONA_KUKA = (0.85, 1.10, 0.85)   # franja de seguridad: cuanto se extiende delante, detras y a los lados
ANCHO_FRANJA = 0.10

# =============================== TUBERIAS ====================================
PLACA_TECHO = 0.60
LEDS_TECHO = [(0.80, -0.55), (0.80, 0.45), (-0.45, -0.55), (-0.45, 0.45),
              (-1.75, -0.55), (-1.75, 0.45)]   # (x, y): fuera de las varillas de la red
ALTO_FALDON = 0.14              # tabica alrededor del techo
Z_FALDON_COLGADO = 2.20         # con el techo por encima de esto (KUKA: 2.60) la tabica cuelga
                                # bajo el techo; por debajo (columna: 1.85, casi a la altura de la
                                # camara del visor, 1.82) taparia el fondo, y va por ENCIMA del techo
ATRIL = (-0.45, 1.25)           # cartel de pie, a la izquierda del robot, mirando a CAMARA_FRENTE
CARRO_DESDE_MESA = (-0.75, -0.43)   # respecto al centro de la mesa (sigue a la mesa si se mueve)
PALE = (-1.45, -1.85)
CONOS = [(-1.00, 1.45), (-0.65, -1.60)]


# =============================== UTILIDADES ==================================
def _f(v, dec=4):
    return " ".join(f"{float(x):.{dec}f}" for x in np.atleast_1d(v))


def _rgba(c, a=1.0):
    return _f(list(c[:3]) + [a], 3)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / max(np.linalg.norm(v), 1e-12)


def _marco(normal, arriba):
    """Rotacion con columnas (derecha, arriba, normal): la normal sale hacia quien mira y
    'derecha' es la derecha de esa persona. Sirve para paredes, carteles y cosas en el suelo."""
    n = _unit(normal)
    u = _unit(np.cross(arriba, n))
    return np.column_stack([u, np.cross(n, u), n])


def _quat(R):
    """Matriz de rotacion -> cuaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


class _Malla:
    """Malla cerrada con normales y coordenadas de textura (u a la derecha, v hacia ABAJO de la
    imagen: v = 0 es la fila de arriba del PNG)."""

    def __init__(self):
        self.V, self.N, self.T, self.F = [], [], [], []

    def quad(self, p, n, t):
        p = [np.asarray(x, dtype=float) for x in p]
        t = list(t)
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), n) < 0:   # que la cara mire hacia n
            p, t = p[::-1], t[::-1]
        b = len(self.V)
        for q, uv in zip(p, t):
            self.V.append(q)
            self.N.append(_unit(n))
            self.T.append(uv)
        self.F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]

    def tri(self, p, n, t):
        p = [np.asarray(x, dtype=float) for x in p]
        t = list(t)
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), n) < 0:
            p, t = p[::-1], t[::-1]
        b = len(self.V)
        for q, uv in zip(p, t):
            self.V.append(q)
            self.N.append(_unit(n))
            self.T.append(uv)
        self.F.append((b, b + 1, b + 2))

    def atributos(self):
        T = [(u, 1.0 - v) if INVERTIR_V else (u, v) for u, v in self.T]
        unir = lambda filas, fmt: " ".join(fmt.format(*f) for f in filas)
        return {"vertex": unir(self.V, "{:.5f} {:.5f} {:.5f}"), "normal": unir(self.N, "{:.4f} {:.4f} {:.4f}"),
                "texcoord": unir(T, "{:.5f} {:.5f}"), "face": unir(self.F, "{} {} {}")}


def _malla_placa(w, h, t, uv=(0.0, 0.0, 1.0, 1.0), uv_solido=(0.005, 0.005), espejo_detras=False):
    """Losa de semilados w (derecha), h (arriba), t (grosor). La imagen (recorte uv = u0, v0,
    u1, v1) va en la cara +z; detras va la misma imagen vista por detras (espejo_detras) o un
    color liso (el pixel uv_solido), como los cantos."""
    u0, v0, u1, v1 = uv
    m = _Malla()
    esquinas = [(-w, -h), (w, -h), (w, h), (-w, h)]
    m.quad([(x, y, t) for x, y in esquinas], (0, 0, 1), [(u0, v1), (u1, v1), (u1, v0), (u0, v0)])
    detras = [(u1, v1), (u0, v1), (u0, v0), (u1, v0)] if espejo_detras else [uv_solido] * 4
    m.quad([(x, y, -t) for x, y in esquinas], (0, 0, -1), detras)
    s = [uv_solido] * 4
    m.quad([(w, -h, -t), (w, h, -t), (w, h, t), (w, -h, t)], (1, 0, 0), s)
    m.quad([(-w, -h, -t), (-w, h, -t), (-w, h, t), (-w, -h, t)], (-1, 0, 0), s)
    m.quad([(-w, h, -t), (w, h, -t), (w, h, t), (-w, h, t)], (0, 1, 0), s)
    m.quad([(-w, -h, -t), (w, -h, -t), (w, -h, t), (-w, -h, t)], (0, -1, 0), s)
    return m


def _malla_revolucion(perfil, n=36):
    """Solido de revolucion alrededor de z. perfil: [(radio, z), ...] de abajo arriba, todos
    los radios > 0; se cierra con una tapa plana abajo y otra arriba."""
    m = _Malla()
    ang = np.linspace(0, 2 * np.pi, n + 1)
    c, s = np.cos(ang), np.sin(ang)
    for (r0, z0), (r1, z1) in zip(perfil[:-1], perfil[1:]):
        nr, nz = z1 - z0, -(r1 - r0)                       # normal hacia fuera en el plano (r, z)
        nr, nz = np.array([nr, nz]) / max(np.hypot(nr, nz), 1e-12)
        for k in range(n):
            p = [(r0 * c[k], r0 * s[k], z0), (r0 * c[k + 1], r0 * s[k + 1], z0),
                 (r1 * c[k + 1], r1 * s[k + 1], z1), (r1 * c[k], r1 * s[k], z1)]
            am = 0.5 * (ang[k] + ang[k + 1])
            normal = (nr * np.cos(am), nr * np.sin(am), nz)
            m.quad(p, normal, [(0, 0)] * 4)
    for (r, z), nz in ((perfil[0], -1.0), (perfil[-1], 1.0)):
        if r > 1e-9:
            for k in range(n):
                m.tri([(0, 0, z), (r * c[k], r * s[k], z), (r * c[k + 1], r * s[k + 1], z)],
                      (0, 0, nz), [(0, 0)] * 3)
    return m


# =============================== TEXTURAS ====================================
def _fuente(tam, negrita=True):
    from PIL import ImageFont
    nombres = ["DejaVuSans-Bold.ttf" if negrita else "DejaVuSans.ttf"]
    rutas = []
    try:
        import matplotlib
        rutas.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", nombres[0]))
    except Exception:
        pass
    rutas += [os.path.join("/usr/share/fonts/truetype/dejavu", nombres[0]), nombres[0],
              r"C:\Windows\Fonts\arialbd.ttf" if negrita else r"C:\Windows\Fonts\arial.ttf",
              "/Library/Fonts/Arial Bold.ttf" if negrita else "/Library/Fonts/Arial.ttf"]
    for r in rutas:
        try:
            return ImageFont.truetype(r, tam)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=tam)
    except TypeError:
        return ImageFont.load_default()


def _texto(dib, caja, texto, color, tam_max, negrita=True):
    """Escribe 'texto' centrado en caja = (x0, y0, x1, y1), lo mas grande que quepa."""
    x0, y0, x1, y1 = caja
    tam = tam_max
    while tam > 8:
        f = _fuente(tam, negrita)
        l, t, r, b = dib.textbbox((0, 0), texto, font=f)
        if r - l <= (x1 - x0) and b - t <= (y1 - y0):
            break
        tam = int(tam * 0.92)
    f = _fuente(tam, negrita)
    l, t, r, b = dib.textbbox((0, 0), texto, font=f)
    dib.text(((x0 + x1 - (r - l)) / 2 - l, (y0 + y1 - (b - t)) / 2 - t), texto, font=f, fill=color)


def _c8(c):
    return tuple(int(round(255 * x)) for x in c[:3])


def _ruido(rng, alto, ancho, celdas):
    """Ruido suave en [-1, 1] (rejilla aleatoria interpolada)."""
    from PIL import Image
    base = rng.uniform(0, 255, (celdas[0], celdas[1])).astype(np.uint8)
    img = Image.fromarray(base, "L").resize((ancho, alto), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 127.5 - 1.0


def _guardar(arr, ruta):
    from PIL import Image
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB").save(ruta)


def _logo(nombre, margen=0.10):
    """Logo sobre fondo blanco, sin el margen que traiga y con 'margen' (fraccion del lado
    menor) alrededor. Devuelve una imagen PIL RGB o None si no esta el archivo."""
    from PIL import Image
    ruta = os.path.join(CARPETA_DECORACION, nombre)
    if not os.path.isfile(ruta):
        print(f"[decoracion] no encuentro {ruta}: sin ese logo")
        return None
    img = Image.open(ruta).convert("RGBA")
    fondo = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(fondo, img).convert("RGB")
    a = np.asarray(img, dtype=np.int16)
    tinta = np.where((255 - a).max(axis=2) > 24)
    if len(tinta[0]):
        img = img.crop((int(tinta[1].min()), int(tinta[0].min()),
                        int(tinta[1].max()) + 1, int(tinta[0].max()) + 1))
    if max(img.size) > 2048:                       # textura de tamano razonable
        k = 2048 / max(img.size)
        img = img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.LANCZOS)
    w, h = img.size
    p = int(round(margen * min(w, h)))
    lienzo = Image.new("RGB", (w + 2 * p, h + 2 * p), (255, 255, 255))
    lienzo.paste(img, (p, p))
    return lienzo


def _pegar(lienzo, logo, caja, fondo_blanco=True, radio=0):
    """Pega 'logo' (PIL) centrado y lo mas grande que quepa en caja (x0, y0, x1, y1) de
    'lienzo'; con fondo_blanco, sobre un rectangulo blanco (redondeado) que ocupa la caja."""
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = (int(round(v)) for v in caja)
    if fondo_blanco:
        ImageDraw.Draw(lienzo).rounded_rectangle((x0, y0, x1, y1), radius=radio, fill=(255, 255, 255))
    if logo is None:
        return
    esc = min((x1 - x0) / logo.width, (y1 - y0) / logo.height)
    w, h = max(1, int(logo.width * esc)), max(1, int(logo.height * esc))
    lienzo.paste(logo.resize((w, h), Image.LANCZOS), (x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2))


def _generar(nombre, fabricar):
    """Ruta de decoracion/generadas/<nombre>; la crea con fabricar(ruta) si no existe."""
    ruta = os.path.join(CARPETA_GENERADAS, nombre)
    if not os.path.isfile(ruta):
        fabricar(ruta)
    return ruta


def _texturas():
    """Genera (si faltan) todas las texturas. Devuelve {clave: ruta} y {clave: ancho/alto} de
    las que hay. Sin Pillow devuelve diccionarios vacios: todo en colores lisos."""
    rutas, aspecto = {}, {}
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[decoracion] sin Pillow (pip install pillow): decoracion en colores lisos y sin logos")
        return rutas, aspecto
    try:
        os.makedirs(CARPETA_GENERADAS, exist_ok=True)
        v = f"v{VERSION}"
        logo_e, logo_i = _logo(LOGO_ELECNOR), _logo(LOGO_IDI)

        # --- logos tal cual (en placas blancas) ---
        for clave, logo in (("logo_elecnor", logo_e), ("logo_idi", logo_i)):
            if logo is not None:
                rutas[clave] = _generar(f"{clave}_{v}.png", lambda r, l=logo: l.save(r))
                aspecto[clave] = logo.width / logo.height

        # --- suelo: loseta de resina de 1 m con su junta (la imagen es 1 m x 1 m) ---
        def suelo(ruta, n=512):
            rng = np.random.default_rng(11)
            img = np.ones((n, n, 3)) * COLOR_SUELO
            img *= (1 + 0.06 * _ruido(rng, n, n, (6, 6)) + 0.025 * _ruido(rng, n, n, (64, 64)))[..., None]
            motas = rng.random((n, n)) > 0.996                  # motas de la resina
            img[motas] = img[motas] * 0.6 + np.array(AZUL_MEDIO) * 0.4
            img[:2, :], img[-2:, :], img[:, :2], img[:, -2:] = (img[:2, :] * 0.55, img[-2:, :] * 0.55,
                                                                img[:, :2] * 0.55, img[:, -2:] * 0.55)
            _guardar(img, ruta)
        rutas["suelo"] = _generar(f"suelo_resina_{v}.png", suelo)

        # --- techo: 2 x 2 placas de 60 cm (la imagen es 1.2 m x 1.2 m) con perfil en T ---
        def techo(ruta, n=512):
            rng = np.random.default_rng(5)
            img = np.ones((n, n, 3)) * COLOR_TECHO
            img *= (1 + 0.015 * _ruido(rng, n, n, (128, 128)))[..., None]
            yy, xx = np.mgrid[0:n, 0:n]
            poros = ((xx % 9 == 4) & (yy % 9 == 4)) & (rng.random((n, n)) > 0.35)
            img[poros] *= 0.86
            for k in (0, n // 2 - 2, n - 3):                    # perfiles (juntas) cada 60 cm
                img[k:k + 3, :] = np.array(COLOR_TECHO) * 0.80
                img[:, k:k + 3] = np.array(COLOR_TECHO) * 0.80
            _guardar(img, ruta)
        rutas["techo"] = _generar(f"techo_placas_{v}.png", techo)

        # --- franjas de seguridad amarillo/negro a 45 grados (0.4 m x 0.1 m, se repite en u) ---
        def peligro(ruta):
            yy, xx = np.mgrid[0:64, 0:256]
            amarillo = ((xx + yy) % 64) < 32
            img = np.where(amarillo[..., None], np.array([0.98, 0.78, 0.06]), np.array(NEGRO))
            _guardar(img, ruta)
        rutas["peligro"] = _generar(f"franjas_peligro_{v}.png", peligro)

        # --- rotulo del proyecto: texto blanco sobre azul Elecnor (2.0 m x 0.25 m) ---
        if ROTULO:
            def rotulo(ruta, W=1600, H=200):
                im = Image.new("RGB", (W, H), _c8(AZUL))
                _texto(ImageDraw.Draw(im), (40, 30, W - 40, H - 30), ROTULO, (255, 255, 255), 150)
                im.save(ruta)
            clave = "".join(ch for ch in unicodedata.normalize("NFKD", ROTULO) if ch.isalnum())[:24]
            rutas["rotulo"] = _generar(f"rotulo_{clave}_{v}.png", rotulo)
            aspecto["rotulo"] = 1600 / 200

        # --- roll-up: 0.85 m x 2.0 m ---
        def rollup(ruta, W=600):
            H = int(round(W * 2.0 / 0.85))
            t = np.linspace(0, 1, H)[:, None, None]
            img = np.array(AZUL) * (1 - t) + np.array(AZUL_OSCURO) * t
            img = np.repeat(img, W, axis=1)
            yy, xx = np.mgrid[0:H, 0:W]
            img[((xx - yy) % 70) < 2] *= 1.25                   # rayado diagonal sutil
            im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")
            d = ImageDraw.Draw(im)
            d.polygon([(0, 0.06 * H), (W, 0.02 * H), (W, 0.045 * H), (0, 0.085 * H)], fill=_c8(NARANJA))
            d.polygon([(0, 0.93 * H), (W, 0.87 * H), (W, H), (0, H)], fill=_c8(NARANJA))
            d.polygon([(0, 0.955 * H), (W, 0.895 * H), (W, 0.90 * H), (0, 0.96 * H)], fill=(255, 255, 255))
            # logo de Elecnor hacia el 37 % de la altura: a la altura de la vista del visor
            _pegar(im, logo_e, (0.07 * W, 0.30 * H, 0.93 * W, 0.45 * H), radio=18)
            if LEMA_ROLLUP:
                _texto(d, (0.08 * W, 0.50 * H, 0.92 * W, 0.56 * H), LEMA_ROLLUP, (255, 255, 255), 64)
                d.rectangle((0.38 * W, 0.575 * H, 0.62 * W, 0.578 * H), fill=_c8(NARANJA))
            _pegar(im, logo_i, (0.15 * W, 0.62 * H, 0.85 * W, 0.72 * H), radio=14)
            im.save(ruta)
        rutas["rollup"] = _generar(f"rollup_{v}.png", rollup)

        # --- atril de la zona de montaje: 0.60 m x 0.45 m, dos caras ---
        def cartel(ruta, W=960, H=720):
            im = Image.new("RGB", (W, H), (250, 250, 250))
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, W, 0.24 * H), fill=_c8(AZUL))
            d.rectangle((0, 0.24 * H, W, 0.265 * H), fill=_c8(NARANJA))
            _texto(d, (40, 0.03 * H, W - 40, 0.21 * H), "ZONA DE MONTAJE", (255, 255, 255), 110)
            _texto(d, (60, 0.31 * H, W - 60, 0.43 * H), "Instalación de tuberías", _c8(AZUL_OSCURO), 70)
            _texto(d, (60, 0.45 * H, W - 60, 0.53 * H), "Teleoperación · TRON 2", _c8(AZUL_MEDIO), 52, negrita=False)
            d.rectangle((0.06 * W, 0.60 * H, 0.94 * W, 0.603 * H), fill=(200, 202, 206))
            _pegar(im, logo_e, (0.06 * W, 0.65 * H, 0.52 * W, 0.92 * H))
            _pegar(im, logo_i, (0.58 * W, 0.68 * H, 0.94 * W, 0.89 * H))
            d.rectangle((0, 0.96 * H, W, H), fill=_c8(NARANJA))
            im.save(ruta)
        rutas["cartel"] = _generar(f"cartel_montaje_{v}.png", cartel)
        aspecto["cartel"] = 960 / 720
    except Exception as e:                       # sin permiso de escritura, PNG roto...
        print(f"[decoracion] no se pudieron generar todas las texturas ({e}): lo que falte, en colores lisos")
    return rutas, aspecto


# =============================== CONSTRUCTOR =================================
class _Deco:
    """Anade geoms solo visuales (sin colision ni masa) a un grupo de visualizacion."""

    def __init__(self, root, grupo, rutas, aspecto):
        self.root, self.grupo, self.rutas, self.aspecto = root, str(grupo), rutas, aspecto
        self.asset = root.find("asset")
        if self.asset is None:
            self.asset = ET.SubElement(root, "asset")
        self.mundo = root.find("worldbody")
        self.nombres = {e.get("name") for e in self.asset if e.get("name")}
        self.mallas = {}

    def _nombre(self, base):
        k, nombre = 0, base
        while nombre in self.nombres:
            k += 1
            nombre = f"{base}_{k}"
        self.nombres.add(nombre)
        return nombre

    def material(self, nombre, color, textura=None, spec=0.2, brillo=0.3, emision=0.0, uniforme=False, repetir=(1, 1)):
        """Material 'deco_<nombre>' (se crea una vez). Con textura (clave de self.rutas) el color
        del material es blanco y manda la imagen. Devuelve (nombre, rgba que debe llevar el geom)."""
        nombre = "deco_" + nombre
        ruta = self.rutas.get(textura) if textura else None
        rgba = "1 1 1 1" if ruta else _rgba(color)
        if nombre not in self.nombres:
            self.nombres.add(nombre)
            atr = {"name": nombre, "rgba": rgba, "specular": str(spec), "shininess": str(brillo),
                   "reflectance": "0"}
            if emision:
                atr["emission"] = str(emision)
            if ruta:
                tex = self._nombre("tex_" + nombre)
                ET.SubElement(self.asset, "texture", {"name": tex, "type": "2d", "file": ruta})
                atr.update(texture=tex, texuniform="true" if uniforme else "false", texrepeat=_f(repetir))
            ET.SubElement(self.asset, "material", atr)
        return nombre, rgba

    def geom(self, material=None, color=None, padre=None, **kw):
        """Geom visual. Lleva SIEMPRE rgba explicito: el <default> del robot es semitransparente
        y un rgba heredado tapa el color del material."""
        atr = {"contype": "0", "conaffinity": "0", "group": self.grupo}
        if material is not None:
            nombre, rgba = material
            atr.update(material=nombre, rgba=rgba)
        else:
            atr["rgba"] = _rgba(color if color is not None else BLANCO)
        for k, v in kw.items():
            atr[k] = v if isinstance(v, str) else _f(v, 6 if k == "quat" else 4)
        return ET.SubElement(self.mundo if padre is None else padre, "geom", atr)

    def caja(self, centro, semi, color=None, material=None, R=None):
        kw = dict(type="box", pos=centro, size=semi)
        if R is not None:
            kw["quat"] = _quat(R)
        return self.geom(material, color, **kw)

    def cilindro(self, a, b, r, color=None, material=None):
        return self.geom(material, color, type="cylinder", fromto=np.r_[a, b], size=r)

    def plano(self, centro, normal, arriba, semiancho, semialto, color):
        """Plano: MuJoCo solo lo dibuja por su cara visible (la de 'normal'): desde fuera de
        la sala las paredes no tapan la camara."""
        return self.geom(None, color, type="plane", pos=centro, quat=_quat(_marco(normal, arriba)),
                         size=(semiancho, semialto, 0.5))

    def malla(self, base, m, centro, R, color=None, material=None):
        atr = m.atributos()
        clave = tuple(atr.values())             # mallas iguales se comparten (por su contenido)
        if clave not in self.mallas:
            self.mallas[clave] = self._nombre("deco_malla_" + base)
            ET.SubElement(self.asset, "mesh", dict(name=self.mallas[clave], **atr))
        return self.geom(material, color, type="mesh", mesh=self.mallas[clave], pos=centro, quat=_quat(R))

    def placa(self, clave, centro, normal, arriba, ancho, alto, grosor=0.004, uv=(0, 0, 1, 1),
              uv_solido=(0.005, 0.005), espejo_detras=False, color=BLANCO, spec=0.15):
        """Placa con la imagen 'clave' en su cara delantera (la que mira a 'normal'). Sin
        textura, una losa del color dado."""
        mat = self.material("img_" + clave, color, textura=clave, spec=spec)
        m = _malla_placa(ancho / 2, alto / 2, grosor / 2, uv, uv_solido, espejo_detras)
        return self.malla(clave, m, centro, _marco(normal, arriba), material=mat)

    def tiene(self, clave):
        return clave in self.rutas

    def medida_logo(self, clave, ancho, alto_max):
        """(ancho, alto) de una placa con el logo, respetando su proporcion."""
        a = self.aspecto.get(clave, 3.0)
        alto = ancho / a
        if alto > alto_max:
            alto, ancho = alto_max, alto_max * a
        return ancho, alto


# =============================== BASE ========================================
def _hay_kuka(root):
    for b in root.iter("body"):
        if b.get("name") == "kuka_soporte":
            return np.array([float(x) for x in b.get("pos", "0 0 0").split()])
    return None


def _suelo(d):
    suelo = next((g for g in d.mundo.findall("geom") if g.get("name") == "ground"), None)
    if suelo is None or not d.tiene("suelo"):
        return
    nombre, rgba = d.material("suelo", COLOR_SUELO, textura="suelo", spec=0.25, brillo=0.4,
                              uniforme=True, repetir=(1, 1))
    for e in d.asset:
        if e.get("name") == nombre and REFLEJO_SUELO:
            e.set("reflectance", str(REFLEJO_SUELO))
    suelo.set("material", nombre)
    suelo.set("rgba", rgba)


def _zona_robot(d):
    """Disco azul con aro naranja bajo el robot, logo en el suelo y cinta de la celda."""
    d.geom(None, NARANJA, type="cylinder", pos=(0, 0, 0.001), size=(R_ZONA + ANCHO_ARO, 0.001))
    d.geom(None, AZUL, type="cylinder", pos=(0, 0, 0.0025), size=(R_ZONA, 0.001))
    if d.tiene("logo_elecnor"):
        ancho, alto = d.medida_logo("logo_elecnor", 0.70, 0.28)
        # Uno a cada lado del torso, dentro del disco, y cada uno legible desde la vista que lo
        # ve: la camara de arranque (delante, mirando a -x) solo ve el suelo de DETRAS del robot
        # (el de delante queda por debajo del encuadre), y la de detras, el de delante.
        for lado in (-1, 1):
            d.placa("logo_elecnor", (lado * (0.40 + alto / 2), 0, 0.005), (0, 0, 1), (lado, 0, 0),
                    ancho, alto, grosor=0.002)
    x0, x1, y0, y1 = CELDA
    a, z = ANCHO_CINTA / 2, 0.001
    for y in (y0, y1):
        d.caja(((x0 + x1) / 2, y, z), ((x1 - x0) / 2 + a, a, 0.001), NARANJA)
    for x in (x0, x1):
        d.caja((x, (y0 + y1) / 2, z), (a, (y1 - y0) / 2 - a, 0.001), NARANJA)


def _pared(d, origen, normal, largo, logos=(), rotulo=None):
    """Pared plana con zocalo, filete y cornisa. origen: punto de la pared a ras de suelo y en
    el centro; logos: [(clave, s, ancho)] con s = desplazamiento a la DERECHA de quien mira la pared."""
    arriba = np.array([0.0, 0.0, 1.0])
    R = _marco(normal, arriba)
    u, n = R[:, 0], R[:, 2]
    o = np.asarray(origen, dtype=float)
    p = lambda s, z, sep: o + u * s + arriba * z + n * sep
    d.plano(p(0, ALTO_SALA / 2, 0), n, arriba, largo / 2, ALTO_SALA / 2, PARED)
    d.plano(p(0, ZOCALO / 2, 0.004), n, arriba, largo / 2, ZOCALO / 2, AZUL)
    d.plano(p(0, ZOCALO + FILETE / 2, 0.004), n, arriba, largo / 2, FILETE / 2, NARANJA)
    d.plano(p(0, ALTO_SALA - CORNISA / 2, 0.004), n, arriba, largo / 2, CORNISA / 2, AZUL)
    d.plano(p(0, ALTO_SALA - CORNISA - FILETE / 4, 0.004), n, arriba, largo / 2, FILETE / 4, NARANJA)
    for clave, s, ancho in logos:
        if not d.tiene(clave):
            continue
        w, h = d.medida_logo(clave, ancho, ALTO_MAX_LOGO)
        d.caja(p(s, Z_LOGOS, 0.012), (w / 2 + 0.035, h / 2 + 0.035, 0.006), AZUL_OSCURO, R=R)  # marco
        d.placa(clave, p(s, Z_LOGOS, 0.0215), n, arriba, w, h, grosor=0.006)
    if rotulo is not None and d.tiene("rotulo"):
        s, ancho = rotulo
        alto = min(ancho / d.aspecto["rotulo"], ZOCALO * 0.6)
        ancho = alto * d.aspecto["rotulo"]
        d.placa("rotulo", p(s, ZOCALO / 2, 0.009), n, arriba, ancho, alto, grosor=0.004,
                color=AZUL, spec=0.2)


def _paredes(d):
    x0, x1 = SALA_X
    y0, y1 = SALA_Y
    xc = (x0 + x1) / 2
    L = LATERAL_LOGOS
    # delante (el fondo de la vista desde detras del robot): su derecha es -y -> Elecnor a su izquierda (+y)
    _pared(d, (x1, 0, 0), (-1, 0, 0), y1 - y0,
           [("logo_elecnor", -L, ANCHO_LOGO_ELECNOR), ("logo_idi", L, ANCHO_LOGO_IDI)],
           rotulo=(-L, 1.6) if ROTULO else None)
    # detras (el fondo de la vista de arranque, tras el KUKA): su derecha es +y
    _pared(d, (x0, 0, 0), (1, 0, 0), y1 - y0,
           [("logo_elecnor", -L, ANCHO_LOGO_ELECNOR), ("logo_idi", L, ANCHO_LOGO_IDI)],
           rotulo=(-L, 1.6) if ROTULO else None)
    # laterales (su derecha: +x en la izquierda del robot, -x en la derecha)
    _pared(d, (xc, y1, 0), (0, -1, 0), x1 - x0, [("logo_elecnor", 0.0, ANCHO_LOGO_ELECNOR)])
    _pared(d, (xc, y0, 0), (0, 1, 0), x1 - x0, [("logo_idi", 0.0, ANCHO_LOGO_IDI)])


def _rollups(d):
    arriba = np.array([0.0, 0.0, 1.0])
    for (x, y), mirar in ROLLUPS:
        pos = np.array([x, y, 0.0])
        n = _unit(np.r_[np.asarray(mirar) - pos[:2], 0.0])
        R = _marco(n, arriba)
        u = R[:, 0]
        d.caja(pos + arriba * 0.05, (0.45, 0.05, 0.06), ALUMINIO, R=R)          # casete
        for lado in (-1, 1):                                                 # patas
            d.caja(pos + arriba * 0.006 + u * lado * 0.38, (0.015, 0.006, 0.16), ACERO, R=R)
        d.cilindro(pos - n * 0.025 + arriba * 0.10, pos - n * 0.025 + arriba * 2.10, 0.008, ALUMINIO)
        d.cilindro(pos + arriba * 2.105 - u * 0.44, pos + arriba * 2.105 + u * 0.44, 0.009, ALUMINIO)
        if d.tiene("rollup"):
            d.placa("rollup", pos + arriba * 1.10, n, arriba, 0.85, 2.0, grosor=0.003,
                    uv_solido=(0.01, 0.30), spec=0.1)
        else:
            d.caja(pos + arriba * 1.10, (0.425, 1.0, 0.0015), AZUL, R=R)


def _zona_kuka(d, base):
    """Placa de anclaje y franja amarillo/negro alrededor de la base del KUKA."""
    bx, by = float(base[0]), float(base[1])
    s = PLACA_KUKA
    d.caja((bx, by, 0.006), (s, s, 0.006), ACERO)
    for ex in (-1, 1):
        for ey in (-1, 1):
            d.geom(None, (0.55, 0.57, 0.60), type="cylinder",
                   pos=(bx + ex * (s - 0.08), by + ey * (s - 0.08), 0.016), size=(0.03, 0.004))
    delante, detras, lado = ZONA_KUKA
    x0, x1, y0, y1 = bx - detras, bx + delante, by - lado, by + lado
    a = ANCHO_FRANJA
    mat = d.material("peligro", (0.98, 0.78, 0.06), textura="peligro", spec=0.1)
    for y in (y0 + a / 2, y1 - a / 2):                 # tramos a lo largo de x (enteros)
        L = x1 - x0
        m = _malla_placa(L / 2, a / 2, 0.001, uv=(0, 0, L / 0.4, 1), uv_solido=(0.01, 0.5))
        d.malla("franja", m, ((x0 + x1) / 2, y, 0.001), _marco((0, 0, 1), (0, 1, 0)), material=mat)
    for x in (x0 + a / 2, x1 - a / 2):                 # tramos a lo largo de y (entre los otros)
        L = y1 - y0 - 2 * a
        m = _malla_placa(L / 2, a / 2, 0.001, uv=(0, 0, L / 0.4, 1), uv_solido=(0.01, 0.5))
        d.malla("franja", m, (x, (y0 + y1) / 2, 0.001), _marco((0, 0, 1), (-1, 0, 0)), material=mat)


def decorar_base(root, rutas=None, aspecto=None):
    """Decoracion corporativa siempre visible (grupo GRUPO_BASE). Llamar con el KUKA ya montado."""
    if not ACTIVA:
        return
    if rutas is None:
        rutas, aspecto = _texturas()
    d = _Deco(root, GRUPO_BASE, rutas, aspecto)
    _suelo(d)
    _zona_robot(d)
    _paredes(d)
    _rollups(d)
    base_kuka = _hay_kuka(root)
    if base_kuka is not None:
        _zona_kuka(d, base_kuka)
    print(f"[decoracion] Elecnor: sala {SALA_X[1] - SALA_X[0]:.1f} x {SALA_Y[1] - SALA_Y[0]:.1f} m"
          + (", zona KUKA" if base_kuka is not None else "")
          + ("" if rutas.get("logo_elecnor") else " | SIN logo de Elecnor"))


# =============================== TUBERIAS ====================================
def _techo(d, esc, z):
    x0, x1, y0, y1 = esc.TECHO_EXTENSION
    techo = next((g for g in d.mundo.findall("geom") if g.get("name") == "techo"), None)
    if techo is not None and d.tiene("techo"):
        nombre, rgba = d.material("techo", COLOR_TECHO, textura="techo", spec=0.05, brillo=0.1,
                                  uniforme=True, repetir=(1 / (2 * PLACA_TECHO), 1 / (2 * PLACA_TECHO)))
        techo.set("material", nombre)
        techo.set("rgba", rgba)
    # paneles LED (empotrados) con su marco de aluminio
    led = d.material("led", (0.97, 0.98, 1.0), emision=0.85, spec=0.0)
    for x, y in LEDS_TECHO:
        if x0 + 0.33 < x < x1 - 0.33 and y0 + 0.33 < y < y1 - 0.33:
            d.caja((x, y, z - 0.006), (0.305, 0.305, 0.006), ALUMINIO)
            d.caja((x, y, z - 0.014), (0.285, 0.285, 0.004), material=led)
    # faldon (tabica) alrededor: azul con filete naranja abajo
    h, e = ALTO_FALDON / 2, 0.015
    zb = z - ALTO_FALDON if z > Z_FALDON_COLGADO else z        # borde de abajo de la tabica
    zc, zf = zb + h, zb + 0.007
    for x in (x0 + e, x1 - e):
        d.caja((x, (y0 + y1) / 2, zc), (e, (y1 - y0) / 2, h), AZUL)
        d.caja((x, (y0 + y1) / 2, zf), (e + 0.0015, (y1 - y0) / 2 + 0.0015, 0.008), NARANJA)
    for y in (y0 + e, y1 - e):
        d.caja(((x0 + x1) / 2, y, zc), ((x1 - x0) / 2 - 2 * e, e, h), AZUL)
        d.caja(((x0 + x1) / 2, y, zf), ((x1 - x0) / 2 - 2 * e, e + 0.0015, 0.008), NARANJA)
    # logos en el faldon: las dos caras del de delante (la de fuera la ve la camara de arranque,
    # que queda fuera del techo; la de dentro, la de detras del robot) y la cara interior del de
    # detras. Elecnor siempre a la izquierda de quien mira.
    alto = ALTO_FALDON - 0.045
    caras = [(x1 + 0.0025, (1, 0, 0), -1), (x1 - 2 * e - 0.0025, (-1, 0, 0), 1), (x0 + 2 * e + 0.0025, (1, 0, 0), -1)]
    for xc, normal, izq in caras:
        for clave, y in (("logo_elecnor", izq * 1.0), ("logo_idi", -izq * 1.0)):
            if d.tiene(clave) and y0 + 0.5 < y < y1 - 0.5:
                w, hh = d.medida_logo(clave, 0.6, alto)
                d.placa(clave, (xc, y, zc + 0.006), normal, (0, 0, 1), w, hh, grosor=0.004)


def _atril(d, pos, mirar):
    """Cartel de pie 'ZONA DE MONTAJE', de dos caras (se lee por las dos), mirando a 'mirar'."""
    x, y = pos
    R = _marco(np.r_[np.asarray(mirar, dtype=float) - (x, y), 0.0], (0, 0, 1))
    d.caja((x, y, 0.008), (0.16, 0.16, 0.008), ACERO)
    d.cilindro((x, y, 0.016), (x, y, 0.95), 0.022, ALUMINIO)
    d.caja((x, y, 0.96), (0.26, 0.012, 0.02), ACERO, R=R)               # soporte del cartel
    ancho = 0.60
    alto = ancho / d.aspecto.get("cartel", 4 / 3)
    centro = (x, y, 0.972 + alto / 2)
    if d.tiene("cartel"):
        d.placa("cartel", centro, R[:, 2], (0, 0, 1), ancho, alto, grosor=0.008,
                uv_solido=(0.02, 0.57), espejo_detras=True, spec=0.15)
    else:
        d.caja(centro, (ancho / 2, alto / 2, 0.004), AZUL, R=R)


def _carro(d, pos):
    """Carro de herramientas azul Elecnor: cajones hacia el robot (+y), logo en los dos
    extremos (+x y -x, uno para cada vista) y encima una caja de accesorios de PVC."""
    x, y = pos
    sx, sy = 0.25, 0.20
    for ex in (-1, 1):                                                    # ruedas
        for ey in (-1, 1):
            cx, cy = x + ex * (sx - 0.05), y + ey * (sy - 0.05)
            d.cilindro((cx, cy - 0.015, 0.035), (cx, cy + 0.015, 0.035), 0.035, NEGRO)
            d.caja((cx, cy, 0.075), (0.02, 0.022, 0.008), ACERO)
    d.caja((x, y, 0.49), (sx, sy, 0.41), AZUL)
    d.caja((x, y, 0.92), (sx + 0.015, sy + 0.015, 0.02), (0.16, 0.17, 0.19))    # encimera
    yf = y + sy
    for k, zc in enumerate((0.20, 0.38, 0.56, 0.74)):
        d.caja((x, yf + 0.005, zc), (sx - 0.02, 0.005, 0.082), AZUL_MEDIO)
        d.caja((x, yf + 0.018, zc + 0.05), (0.11, 0.008, 0.007), NARANJA)
    # asa de empuje en el lado +x
    for ey in (-1, 1):
        d.cilindro((x + sx, y + ey * 0.14, 0.86), (x + sx + 0.06, y + ey * 0.14, 0.86), 0.012, ACERO)
    d.cilindro((x + sx + 0.06, y - 0.15, 0.86), (x + sx + 0.06, y + 0.15, 0.86), 0.015, NARANJA)
    if d.tiene("logo_elecnor"):
        w, h = d.medida_logo("logo_elecnor", 0.34, 0.20)
        for lado in (-1, 1):
            d.placa("logo_elecnor", (x + lado * (sx + 0.0025), y, 0.58), (lado, 0, 0), (0, 0, 1), w, h,
                    grosor=0.004)
    _caja_accesorios(d, (x, y), 0.94)


def _pale(d, pos):
    """Pale europeo (1.2 x 0.8) con tubos de PVC de 63 mm apilados."""
    x, y = pos
    for yy in (-0.35, 0.0, 0.35):                                         # tablas de abajo
        d.caja((x, y + yy, 0.011), (0.60, 0.05, 0.011), MADERA_OSCURA)
    for xx in (-0.55, 0.0, 0.55):
        for yy in (-0.35, 0.0, 0.35):                                     # tacos
            d.caja((x + xx, y + yy, 0.061), (0.05, 0.05, 0.039), MADERA_OSCURA)
        d.caja((x + xx, y, 0.111), (0.05, 0.40, 0.011), MADERA)          # travesanos
    for yy in np.linspace(-0.35, 0.35, 5):                                # tablas de arriba
        d.caja((x, y + yy, 0.133), (0.60, 0.05, 0.011), MADERA)
    r = 0.0315
    try:                         # tubos huecos con su campana (tuberias_pvc.py)
        import tuberias_pvc
        k_pvc = tuberias_pvc.Constructor(d, semilla=9) if tuberias_pvc.ACTIVA else None
    except ImportError:
        k_pvc = None
    circulos = []                # seccion de los tubos (y, z, radio): por ella pasan los flejes
    for capa, n in enumerate((10, 9, 8)):
        z = 0.144 + r + capa * r * np.sqrt(3)
        for k in range(n):
            yy = (k - (n - 1) / 2) * 2 * r
            circulos.append((yy, z, r))
            if k_pvc is None:
                d.cilindro((x - 0.59, y + yy, z), (x + 0.59, y + yy, z), r - 0.0005, PVC)
                continue
            s = 1.0 if (k + capa) % 2 else -1.0          # campanas alternadas, como se apilan
            R = r - 0.0005
            boca = np.array([x + s * 0.59, y + yy, z])
            k_pvc.tubo(np.array([x - s * 0.59, y + yy, z]), boca - np.array([s * 0.075, 0, 0]), R)
            k_pvc.campana(boca, (s, 0.0, 0.0), R, L=0.09)
    # flejes naranjas: rodean el haz de tubos tensados (se apoyan en los tubos de fuera y bajan
    # en diagonal) y pasan por debajo de la tabla de arriba del pale, entre los tacos
    tablero = (0.40, 0.122, 0.144)           # semiancho en y, cara de abajo y de arriba
    for xx in (-0.35, 0.35):
        _fleje(d, (x + xx, y), circulos, tablero, sello_y=0.09 if xx > 0 else -0.09)


FLEJE = dict(ancho=0.013, grosor=0.0011, holgura=0.0007)   # cinta de PP de 13 mm


def _envolvente(puntos):
    """Envolvente convexa 2D (cadena monotona), en sentido antihorario."""
    P = sorted(set(map(tuple, np.round(puntos, 7))))
    cruz = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    bajo, alto = [], []
    for p in P:
        while len(bajo) >= 2 and cruz(bajo[-2], bajo[-1], p) <= 0:
            bajo.pop()
        bajo.append(p)
    for p in reversed(P):
        while len(alto) >= 2 and cruz(alto[-2], alto[-1], p) <= 0:
            alto.pop()
        alto.append(p)
    return np.array(bajo[:-1] + alto[:-1])


def _fleje(d, pos, circulos, tablero, sello_y=0.0):
    """Fleje cerrado en el plano y-z en x = pos[0]: la envolvente tensa de los tubos y del
    tablero del pale (como queda una cinta apretada), hecha malla con algo de grosor, y su
    sello metalico encima."""
    x0, y0 = pos
    h, w, t = FLEJE["holgura"], FLEJE["ancho"] / 2, FLEJE["grosor"]
    ang = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    pts = [np.c_[cy + (r + h) * np.cos(ang), cz + (r + h) * np.sin(ang)] for cy, cz, r in circulos]
    sy, z_bajo, z_alto = tablero
    pts.append(np.array([[-sy - h, z_bajo - h], [sy + h, z_bajo - h], [sy + h, z_alto], [-sy - h, z_alto]]))
    lazo = _envolvente(np.vstack(pts))
    # normales hacia fuera en cada punto (media de los dos tramos que llegan a el)
    n = len(lazo)
    tan = np.array([_unit(np.r_[lazo[(i + 1) % n] - lazo[i], 0.0])[:2] for i in range(n)])
    normal_seg = np.c_[tan[:, 1], -tan[:, 0]]               # hacia fuera (lazo antihorario)
    normal = np.array([_unit(np.r_[normal_seg[i - 1] + normal_seg[i], 0.0])[:2] for i in range(n)])
    m = _Malla()

    def cara(p, nrm):                  # cuatro vertices con su normal propia (cara suave)
        b = len(m.V)
        for q, nq in zip(p, nrm):
            m.V.append(np.asarray(q, dtype=float))
            m.N.append(_unit(nq))
            m.T.append((0.5, 0.5))
        a0, a1, a2 = (m.V[b + k] for k in range(3))
        media = sum(nrm) / 4
        if np.dot(np.cross(a1 - a0, a2 - a0), media) < 0:
            m.F += [(b, b + 2, b + 1), (b, b + 3, b + 2)]
        else:
            m.F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]

    P3 = lambda i, ex, off: np.array([ex, lazo[i % n][0] + normal[i % n][0] * off,
                                      lazo[i % n][1] + normal[i % n][1] * off])
    N3 = lambda i, s: np.array([0.0, s * normal[i % n][0], s * normal[i % n][1]])
    for i in range(n):
        j = i + 1
        cara([P3(i, -w, t), P3(j, -w, t), P3(j, w, t), P3(i, w, t)], [N3(i, 1), N3(j, 1), N3(j, 1), N3(i, 1)])
        cara([P3(i, -w, 0), P3(i, w, 0), P3(j, w, 0), P3(j, -w, 0)], [N3(i, -1), N3(i, -1), N3(j, -1), N3(j, -1)])
        for ex in (-w, w):
            nx = np.array([np.sign(ex), 0.0, 0.0])
            cara([P3(i, ex, 0), P3(j, ex, 0), P3(j, ex, t), P3(i, ex, t)], [nx] * 4)
    mat = d.material("fleje", NARANJA, spec=0.45, brillo=0.6)
    d.malla("fleje", m, (x0, y0, 0.0), np.eye(3), material=mat)
    # sello metalico: donde el fleje pasa por encima, cerca de sello_y
    arriba = [i for i in range(n) if normal[i][1] > 0.5]
    if arriba:
        i = min(arriba, key=lambda k: abs(lazo[k][0] - sello_y))
        tg = np.array([0.0, tan[i][0], tan[i][1]])
        nn = np.array([0.0, normal[i][0], normal[i][1]])
        R = np.column_stack([[1.0, 0.0, 0.0], tg, nn])
        centro = np.array([x0, y0 + lazo[i][0], lazo[i][1]]) + nn * (t + 0.0012)
        d.caja(centro, (w + 0.0015, 0.011, 0.0012), (0.70, 0.72, 0.74), R=R)


def _caja_accesorios(d, pos, z0=0.0):
    """Caja de carton de 40 x 30 x 24 cm con el logo en la cara +x y en la +y."""
    x, y = pos
    sx, sy, sz = 0.20, 0.15, 0.12
    d.caja((x, y, z0 + sz), (sx, sy, sz), CARTON)
    d.caja((x, y, z0 + 2 * sz + 0.0012), (sx + 0.003, 0.025, 0.0012), (0.78, 0.66, 0.50))  # cinta
    if d.tiene("logo_elecnor"):
        w, h = d.medida_logo("logo_elecnor", 0.30, 0.12)
        d.placa("logo_elecnor", (x, y + sy + 0.0015, z0 + sz), (0, 1, 0), (0, 0, 1), w, h, grosor=0.003)
        w, h = d.medida_logo("logo_elecnor", 0.24, 0.12)
        d.placa("logo_elecnor", (x + sx + 0.0015, y, z0 + sz), (1, 0, 0), (0, 0, 1), w, h, grosor=0.003)


def _conos(d):
    alto, r_base, r_punta = 0.50, 0.14, 0.022
    z0 = 0.03
    cono = _malla_revolucion([(r_base, z0), (r_punta, z0 + alto)])
    radio = lambda z: r_base + (r_punta - r_base) * (z - z0) / alto
    bandas = [_malla_revolucion([(radio(a) + 0.002, a), (radio(b) + 0.002, b)]) for a, b in ((0.22, 0.29), (0.35, 0.40))]
    for x, y in CONOS:
        d.caja((x, y, 0.015), (0.18, 0.18, 0.015), NEGRO)
        d.malla("cono", cono, (x, y, 0), np.eye(3), NARANJA)
        for m in bandas:
            d.malla("banda_cono", m, (x, y, 0), np.eye(3), BLANCO)


def decorar_tuberias(root, esc, rutas=None, aspecto=None):
    """Decoracion del escenario de tuberias (en su grupo: solo se ve con el escenario activo).
    esc: el modulo escenario_tuberias, despues de anadir_al_modelo (usa z_techo, la mesa...)."""
    if not ACTIVA:
        return
    if rutas is None:
        rutas, aspecto = _texturas()
    d = _Deco(root, esc.GRUPO_ESCENARIO, rutas, aspecto)
    z = float(esc.z_techo)
    cx, cy = esc.MESA_CENTRO
    _techo(d, esc, z)
    _atril(d, ATRIL, CAMARA_FRENTE)
    _carro(d, (cx + CARRO_DESDE_MESA[0], cy + CARRO_DESDE_MESA[1]))
    _pale(d, PALE)
    _conos(d)


def con_decoracion(escenario=None):
    """Envuelve la funcion del escenario (escenario_tuberias.anadir_al_modelo): decora la
    base, monta el escenario y lo decora. Se pasa a RobotTron2 en lugar de la funcion."""
    def extra(root):
        rutas, aspecto = _texturas() if ACTIVA else ({}, {})
        decorar_base(root, rutas, aspecto)
        if escenario is None:
            return
        escenario(root)
        modulo = sys.modules.get(getattr(escenario, "__module__", ""))
        if modulo is not None and hasattr(modulo, "TECHO_EXTENSION") and hasattr(modulo, "z_techo"):
            decorar_tuberias(root, modulo, rutas, aspecto)
    return extra


# =============================== PRUEBA ======================================
if __name__ == "__main__":
    import mujoco
    import mujoco.viewer

    import escenario_tuberias
    from robot_tron2 import SOPORTE_KUKA, quitar_base_flotante

    xml = sys.argv[1] if len(sys.argv) > 1 else "tron2a/DACH_TRON2A/xml/robot_elecnor.xml"
    extra = con_decoracion(escenario_tuberias.anadir_al_modelo)
    if SOPORTE_KUKA:
        import montar_kuka
        extra = montar_kuka.con_kuka(extra)
    m = mujoco.MjModel.from_xml_path(quitar_base_flotante(xml, extra))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"[decoracion] modelo listo: {m.ngeom} geoms. Grupo {escenario_tuberias.GRUPO_ESCENARIO} "
          f"(tuberias) a la vista: tecla {escenario_tuberias.GRUPO_ESCENARIO} del visor para ocultarlo")
    with mujoco.viewer.launch_passive(m, d) as v:
        with v.lock():                  # la misma vista con la que arranca teleop_tron2.py
            v.opt.geomgroup[escenario_tuberias.GRUPO_ESCENARIO] = 1
            v.cam.lookat[:] = (0.3, 0.0, 1.06)
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.8, 180.0, -25.0
        while v.is_running():
            v.sync()
            import time
            time.sleep(0.05)
