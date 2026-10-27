"""
Tuberias de PVC con detalle para el escenario de montaje (solo estetica: la logica de
escenario_tuberias.py sigue usando TUBOS_TECHO, CONEXION_PUNTO, EXTREMOS... como siempre).

Antes cada tubo era un cilindro macizo de MuJoCo y los codos, esferas. Ahora:
  - TUBOS HUECOS: pared de 3 mm con el interior oscuro y bisel en los extremos. El extremo
    libre del punto de conexion se ve abierto, como un tubo de verdad.
  - Marcaje impreso a lo largo del tubo ("PVC-U  O40 x 3,0  UNE-EN 1329-1 ..."), como el de
    fabrica, y un leve rayado de extrusion. Va en los dos costados para que se vea siempre.
  - CODOS de verdad (no esferas): cuerpo curvo con una copa en cada lado en la que entran los
    tubos. Las copas tienen hombro y boca achaflanada.
  - UNIONES: las barras miden BARRA m; en los tramos mas largos hay una copa de union.
  - SUBIDAS: los extremos libres de la red (salvo el punto de conexion) giran con un codo y
    suben al forjado, con un embellecedor en el techo (antes acababan cortados en el aire).
  - ABRAZADERAS isofonicas: banda naranja Elecnor con su goma negra, orejetas con tornillo a
    los lados, tuerca soldada arriba, varilla roscada cincada con contratuerca y arandela + tuerca
    en el techo.
  - ETIQUETAS de identificacion "EVACUACION" con la flecha del sentido del flujo.
  - La pieza suelta (si es el codo de PVC) se construye igual: codo, tramos y manguitos huecos.

Todo son mallas con sus coordenadas de textura (como los logos de decoracion_elecnor.py, del que
reutiliza utilidades y la carpeta decoracion/generadas/). Sin Pillow, mismas formas en colores
lisos. Para volver a los cilindros de antes: TRON2_TUBERIAS_DETALLE=0
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

import decoracion_elecnor as de

ACTIVA = os.environ.get("TRON2_TUBERIAS_DETALLE", "1") != "0"
VERSION = 3                       # va en el nombre de las texturas: subirlo las regenera

# =============================== MEDIDAS =====================================
ESPESOR = 0.003                   # pared del tubo (m)
N_LADOS = 40                      # segmentos alrededor del tubo
BARRA = 3.0                       # m: largo de una barra de tubo (en los tramos largos hay uniones)
HOLGURA_FONDO = 0.002             # el tubo se queda a esto del fondo de la copa
ETIQUETA_MIN = 0.6                # m: los tramos mas cortos no llevan etiqueta
L_ETIQUETA = 0.12                 # m de largo de cada etiqueta
TEXTO_ETIQUETA = "EVACUACIÓN"
MARCAJE = "PVC-U  Ø{d} x 3,0  UNE-EN 1329-1  B  ·  2026/10"   # {d}: diametro en mm
SUBIR_AL_TECHO = True             # los extremos libres suben al forjado (si no, tapon)


def radio_copa(R):
    """Radio exterior de copas, codos y manguitos."""
    return R + max(0.0032, 0.15 * R)


def largo_copa(R):
    return max(0.030, 1.8 * R)


def radio_curva(R):
    """Radio del eje del codo (codo corto de evacuacion)."""
    return 2.5 * R


# =============================== COLORES =====================================
PVC_GRIS = (0.60, 0.61, 0.60)     # PVC de evacuacion (gris)
ACCESORIO = (0.53, 0.545, 0.54)   # codos y copas, un punto mas oscuros
INTERIOR = (0.20, 0.21, 0.21)     # dentro del tubo (en sombra)
TINTA = (0.13, 0.14, 0.15)        # marcaje impreso
GOMA = (0.07, 0.07, 0.08)
ZINC = (0.72, 0.74, 0.76)
ZINC_OSCURO = (0.56, 0.58, 0.61)
BLANCO = (0.95, 0.95, 0.94)

# Regiones de la textura del tubo (v; 0 = fila de arriba de la imagen)
V0_EXT, V1_EXT = 0.03, 0.71       # superficie exterior impresa (da la vuelta entera al tubo)
V_LISO = 0.76                     # PVC sin imprimir (biseles, caras, campanas)
V_ACC = 0.86                      # accesorios
V_INT = 0.955                     # interior oscuro
L_TEX = 0.5                       # m de tubo que cubre la imagen a lo largo
PX_M = 2048                       # pixeles por metro


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / max(np.linalg.norm(v), 1e-12)


def _perp(v, ref):
    """Parte de 'ref' perpendicular a v (unitaria); si no hay, otra cualquiera."""
    v = _unit(v)
    for r in (ref, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        p = np.asarray(r, dtype=float) - v * np.dot(r, v)
        if np.linalg.norm(p) > 1e-3:
            return _unit(p)
    return np.array([0.0, 0.0, 1.0])


def marco_tubo(eje, arriba=(0.0, 0.0, 1.0)):
    """Rotacion con columnas (x, y, z): z a lo largo del tubo y x lo mas hacia arriba posible
    (asi el marcaje, a 90 grados de x, queda en los costados)."""
    z = _unit(eje)
    x = _perp(z, arriba)
    return np.column_stack([x, np.cross(z, x), z])


# =============================== MALLAS ======================================
class MallaPVC:
    """Malla con normales y coordenadas de textura por vertice. Las caras se orientan solas
    hacia sus normales al exportar (asi el volumen sale positivo y MuJoCo la acepta)."""

    def __init__(self):
        self.V, self.N, self.T, self.F = [], [], [], []

    def quad(self, p, n, t):
        b = len(self.V)
        self.V += [tuple(x) for x in p]
        self.N += [tuple(_unit(x)) for x in n]
        self.T += [tuple(x) for x in t]
        self.F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]

    def atributos(self):
        V, N = np.array(self.V), np.array(self.N)
        T, F = np.array(self.T, dtype=float), np.array(self.F)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        cruz = np.cross(b - a, c - a)
        ok = np.linalg.norm(cruz, axis=1) > 1e-14          # fuera triangulos degenerados
        F, cruz = F[ok], cruz[ok]
        girar = np.einsum("ij,ij->i", cruz, N[F].sum(axis=1)) < 0
        F[girar] = F[girar][:, [0, 2, 1]]
        if de.INVERTIR_V:
            T[:, 1] = 1.0 - T[:, 1]
        unir = lambda filas, fmt: " ".join(fmt.format(*f) for f in filas)
        return {"vertex": unir(V, "{:.5f} {:.5f} {:.5f}"), "normal": unir(N, "{:.4f} {:.4f} {:.4f}"),
                "texcoord": unir(T, "{:.5f} {:.5f}"), "face": unir(F, "{} {} {}")}


def revolucion(perfil, uvf, n=N_LADOS, facetado=False, theta0=0.0):
    """Solido de revolucion alrededor del eje z local. perfil: lazo CERRADO [(r, z, etiqueta)]
    en sentido antihorario en el plano (r a la derecha, z arriba); la etiqueta de cada punto es
    la de la arista que sale de el. uvf(etiqueta, z, f) -> (u, v), con f = angulo / 2 pi.
    facetado: caras planas (tuercas hexagonales con n = 6)."""
    P = [(float(r), float(z), t) for r, z, t in perfil]
    area = 0.5 * sum(P[i - 1][0] * P[i][1] - P[i][0] * P[i - 1][1] for i in range(len(P)))   # shoelace
    if area <= 0:
        raise ValueError("revolucion: el perfil debe ir en sentido antihorario")
    th = theta0 + np.linspace(0.0, 2 * np.pi, n + 1)
    c, s = np.cos(th), np.sin(th)
    m = MallaPVC()
    for i, (r0, z0, tag) in enumerate(P):
        r1, z1, _ = P[(i + 1) % len(P)]
        L = np.hypot(r1 - r0, z1 - z0)
        if L < 1e-9:
            continue
        nr, nz = (z1 - z0) / L, -(r1 - r0) / L
        for k in range(n):
            if facetado:
                tm = 0.5 * (th[k] + th[k + 1])
                nk = nk1 = (nr * np.cos(tm), nr * np.sin(tm), nz)
            else:
                nk, nk1 = (nr * c[k], nr * s[k], nz), (nr * c[k + 1], nr * s[k + 1], nz)
            f0, f1 = k / n, (k + 1) / n
            m.quad([(r0 * c[k], r0 * s[k], z0), (r0 * c[k + 1], r0 * s[k + 1], z0),
                    (r1 * c[k + 1], r1 * s[k + 1], z1), (r1 * c[k], r1 * s[k], z1)],
                   [nk, nk1, nk1, nk],
                   [uvf(tag, z0, f0), uvf(tag, z0, f1), uvf(tag, z1, f1), uvf(tag, z1, f0)])
    return m


def arco(rb, phi, r_ext, r_int, uvf, n=N_LADOS):
    """Codo hueco: tubo de radios r_ext / r_int curvado con radio de eje rb un angulo phi.
    Empieza en el origen hacia +z y gira hacia +x (en el plano x-z). Termina en
    (rb (1 - cos phi), 0, rb sin phi) con direccion (sin phi, 0, cos phi)."""
    pasos = max(6, int(np.ceil(np.degrees(phi) / 6.0)))
    fi = np.linspace(0.0, phi, pasos + 1)
    th = np.linspace(0.0, 2 * np.pi, n + 1)
    C = np.stack([rb * (1 - np.cos(fi)), np.zeros_like(fi), rb * np.sin(fi)], axis=1)
    A = np.stack([np.cos(fi), np.zeros_like(fi), -np.sin(fi)], axis=1)
    B = np.array([0.0, 1.0, 0.0])
    Tg = np.stack([np.sin(fi), np.zeros_like(fi), np.cos(fi)], axis=1)
    m = MallaPVC()
    radial = lambda j, k: np.cos(th[k]) * A[j] + np.sin(th[k]) * B
    for r, signo, tag in ((r_ext, 1.0, "acc"), (r_int, -1.0, "int")):
        for j in range(pasos):
            for k in range(n):
                idx = [(j, k), (j, k + 1), (j + 1, k + 1), (j + 1, k)]
                m.quad([C[a] + r * radial(a, b) for a, b in idx], [signo * radial(a, b) for a, b in idx],
                       [uvf(tag, rb * fi[a], b / n) for a, b in idx])
    for j, signo in ((0, -1.0), (pasos, 1.0)):            # tapas (coronas) de los extremos
        for k in range(n):
            m.quad([C[j] + r_int * radial(j, k), C[j] + r_int * radial(j, k + 1),
                    C[j] + r_ext * radial(j, k + 1), C[j] + r_ext * radial(j, k)],
                   [signo * Tg[j]] * 4, [uvf("acc", 0.0, 0.0)] * 4)
    return m


# --------------------------- perfiles (r, z, etiqueta) ---------------------------
def perfil_tubo(R, L, bisel=0.0012):
    ri, cz = R - ESPESOR, bisel * 2.5
    return [(ri, 0.0, "liso"), (R - bisel, 0.0, "liso"), (R, cz, "ext"), (R, L - cz, "liso"),
            (R - bisel, L, "liso"), (ri, L, "int")]


def perfil_copa(R, L, r_ext=None, cerrada=False):
    """Copa (boca en z = L, hombro en z = 0) en la que entra un tubo de radio R."""
    ri, ro, ch = R + 0.0003, (r_ext or radio_copa(R)), 0.0015
    if cerrada:                                   # tapon: fondo cerrado en z = 0
        return [(0.0006, -0.004, "acc"), (ro - ch, -0.004, "acc"), (ro, -0.004 + ch, "acc"), (ro, L - ch, "acc"),
                (ro - ch, L, "acc"), (ri + 0.0006, L, "int"), (ri, L - 0.0015, "int"), (ri, 0.0, "int"),
                (0.0006, 0.0, "int")]
    return [(ri, 0.0, "acc"), (ro, 0.0, "acc"), (ro, L - ch, "acc"), (ro - ch, L, "acc"),
            (ri + 0.0006, L, "int"), (ri, L - 0.0015, "int")]


def perfil_campana(R, L):
    """Campana (copa integral) del extremo de una barra: base conica en z = 0, boca en z = L."""
    ro, ri, cono, ch = radio_copa(R) - 0.001, R + 0.0003, 0.010, 0.0015
    return [(R - 0.0012, 0.0, "liso"), (R + 0.0002, 0.0, "liso"), (ro, cono, "liso"), (ro, L - ch, "liso"),
            (ro - ch, L, "liso"), (ri, L, "int"), (ri, cono, "liso"), (R - 0.0012, cono, "int")]


def perfil_anillo(r_int, r_ext, z0, z1, tag_ext="x", tag="x"):
    return [(r_int, z0, tag), (r_ext, z0, tag_ext), (r_ext, z1, tag), (r_int, z1, tag)]


# =============================== TEXTURAS ====================================
def _pegar_texto(im, caja, texto, color, fuente, girado=False):
    """Escribe 'texto' centrado en caja (x0, y0, x1, y1) de la imagen PIL 'im'. girado: 180
    grados (el texto de la otra cara del tubo, para que se lea derecho desde ese lado)."""
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = (int(round(v)) for v in caja)
    capa = Image.new("L", (x1 - x0, y1 - y0), 0)
    d = ImageDraw.Draw(capa)
    l, t, r, b = d.textbbox((0, 0), texto, font=fuente)
    d.text(((x1 - x0 - (r - l)) / 2 - l, (y1 - y0 - (b - t)) / 2 - t), texto, font=fuente, fill=255)
    if girado:
        capa = capa.rotate(180)
    im.paste(Image.new("RGB", capa.size, color), (x0, y0), capa)


def _fuente_que_cabe(texto, ancho, alto, tam_max, negrita=True):
    from PIL import Image, ImageDraw
    d = ImageDraw.Draw(Image.new("L", (8, 8)))
    tam = tam_max
    while tam > 8:
        f = de._fuente(tam, negrita)
        l, t, r, b = d.textbbox((0, 0), texto, font=f)
        if r - l <= ancho and b - t <= alto:
            return f
        tam = int(tam * 0.92)
    return de._fuente(tam, negrita)


def _textura_tubo(R):
    """Marcaje impreso + regiones lisa / accesorio / interior (ver V_*). None sin Pillow."""
    D = int(round(2000 * R))
    ruta = os.path.join(de.CARPETA_GENERADAS, f"tubo_pvc_D{D}_v{VERSION}.png")
    if os.path.isfile(ruta):
        return ruta
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    try:
        os.makedirs(de.CARPETA_GENERADAS, exist_ok=True)
        W = int(round(L_TEX * PX_M))
        H = int(round(2 * np.pi * R * PX_M / (V1_EXT - V0_EXT)))
        fila = lambda v: int(round(v * H))
        rng = np.random.default_rng(D)
        vetas = np.convolve(rng.normal(0, 1, H), np.ones(7) / 7, "same")     # lineas de extrusion
        img = np.ones((H, W, 3)) * np.array(PVC_GRIS)
        img *= (1 + 0.018 * vetas)[:, None, None] * (1 + 0.006 * rng.normal(0, 1, (H, W)))[..., None]
        img[:fila(V0_EXT)] = PVC_GRIS
        img[fila(V1_EXT):fila(0.81)] = PVC_GRIS
        img[fila(0.81):fila(0.905)] = ACCESORIO
        img[fila(0.905):] = INTERIOR
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")
        alto = int(0.010 * PX_M)
        f = de._fuente(max(9, int(0.0065 * PX_M)), negrita=False)
        texto = MARCAJE.format(d=D)
        # en los dos costados (ver marco_tubo); el de la otra cara, girado para leerse derecho
        for frac, girado in ((0.25, False), (0.75, True)):
            y = fila(V0_EXT + frac * (V1_EXT - V0_EXT))
            for x0 in (0, W // 2):
                _pegar_texto(im, (x0 + 8, y - alto / 2, x0 + W // 2 - 8, y + alto / 2), texto,
                             de._c8(TINTA), f, girado)
        im.save(ruta)
        return ruta
    except Exception as e:
        print(f"[tuberias] no se pudo generar la textura del tubo ({e}): color liso")
        return None


def _textura_etiqueta(R):
    D = int(round(2000 * R))
    ruta = os.path.join(de.CARPETA_GENERADAS, f"etiqueta_D{D}_v{VERSION}.png")
    if os.path.isfile(ruta):
        return ruta
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    try:
        os.makedirs(de.CARPETA_GENERADAS, exist_ok=True)
        W = 480
        H = int(round(2 * np.pi * (R + 0.0006) * W / L_ETIQUETA))
        im = Image.new("RGB", (W, H), (250, 250, 248))
        d = ImageDraw.Draw(im)
        az, na = de._c8(de.AZUL), de._c8(de.NARANJA)
        d.rectangle((0, 0, int(0.05 * W), H), fill=na)
        d.rectangle((W - int(0.05 * W), 0, W, H), fill=na)
        d.rectangle((int(0.05 * W), 0, int(0.065 * W), H), fill=az)
        d.rectangle((W - int(0.065 * W), 0, W - int(0.05 * W), H), fill=az)
        h = H / 2
        f = _fuente_que_cabe(TEXTO_ETIQUETA, 0.60 * W, 0.44 * h, 90)
        for mitad in (0, 1):                     # dos veces alrededor: se lee desde los dos lados
            yc = (mitad + 0.5) * h
            # en la segunda mitad el texto va girado 180 grados (se lee derecho desde ese lado)
            _pegar_texto(im, (0.10 * W, yc - 0.22 * h, 0.70 * W, yc + 0.22 * h), TEXTO_ETIQUETA,
                         az, f, girado=bool(mitad))
            for x0 in (0.74, 0.83):              # flechas: sentido del flujo (+u = de a hacia b)
                d.polygon([(x0 * W, yc - 0.16 * h), ((x0 + 0.07) * W, yc), (x0 * W, yc + 0.16 * h)], fill=az)
        im.save(ruta)
        return ruta
    except Exception as e:
        print(f"[tuberias] no se pudo generar la textura de la etiqueta ({e})")
        return None


def _uv_tubo(u0=0.0):
    v_tag = {"liso": V_LISO, "acc": V_ACC, "int": V_INT, "x": V_LISO}

    def uvf(tag, z, f):
        if tag == "ext":
            return (u0 + z / L_TEX, V0_EXT + f * (V1_EXT - V0_EXT))
        return (u0 + z / L_TEX, v_tag[tag])
    return uvf


def _uv_etiqueta(largo):
    return lambda tag, z, f: (z / largo, f) if tag == "etq" else (0.5, 0.004)


def _uv_nada(tag, z, f):
    return (0.5, 0.5)


# =============================== CONSTRUCTOR =================================
class Constructor:
    """Anade las piezas al modelo con un decoracion_elecnor._Deco (materiales, mallas compartidas
    por contenido, grupo de visualizacion, sin colisiones). padre: cuerpo donde van (None =
    worldbody); extra: atributos para cada geom (p.ej. {"mass": "0"} en un cuerpo mocap)."""

    def __init__(self, deco, padre=None, extra=None, semilla=3):
        self.d, self.padre, self.extra = deco, padre, dict(extra or {})
        self.rng = np.random.default_rng(semilla)
        m = self.d.material
        self.m_banda = m("pvc_banda", de.NARANJA, spec=0.5, brillo=0.5)
        self.m_goma = m("pvc_goma", GOMA, spec=0.05, brillo=0.1)
        self.m_zinc = m("pvc_zinc", ZINC, spec=0.8, brillo=0.7)
        self.m_varilla = m("pvc_varilla", ZINC_OSCURO, spec=0.6, brillo=0.5)
        self.m_escudo = m("pvc_escudo", BLANCO, spec=0.2, brillo=0.3)

    # ---------------- materiales con textura ----------------
    def _con_textura(self, clave, fabricar, color, **kw):
        if clave not in self.d.rutas:
            ruta = fabricar()
            if ruta:
                self.d.rutas[clave] = ruta
        return self.d.material(clave, color, textura=clave, **kw)

    def mat_tubo(self, R):
        D = int(round(2000 * R))
        return self._con_textura(f"pvc_tubo{D}", lambda: _textura_tubo(R), PVC_GRIS, spec=0.35, brillo=0.55)

    def mat_etiqueta(self, R):
        D = int(round(2000 * R))
        return self._con_textura(f"pvc_etiqueta{D}", lambda: _textura_etiqueta(R), BLANCO, spec=0.25, brillo=0.4)

    # ---------------- utilidades ----------------
    def _malla(self, base, m):
        atr = m.atributos()
        clave = tuple(atr.values())
        if clave not in self.d.mallas:
            self.d.mallas[clave] = self.d._nombre("pvc_" + base)
            ET.SubElement(self.d.asset, "mesh", dict(name=self.d.mallas[clave], **atr))
        return self.d.mallas[clave]

    def pieza(self, base, m, pos, R, material, nombre=None):
        kw = dict(type="mesh", mesh=self._malla(base, m), pos=np.asarray(pos, dtype=float), quat=de._quat(R))
        kw.update(self.extra)
        if nombre:
            kw["name"] = nombre
        return self.d.geom(material, None, padre=self.padre, **kw)

    def _primitiva(self, material, nombre=None, **kw):
        kw.update(self.extra)
        if nombre:
            kw["name"] = nombre
        return self.d.geom(material, None, padre=self.padre, **kw)

    # ---------------- piezas de PVC ----------------
    def tubo(self, a, b, R, nombre=None):
        """Tubo hueco de a a b (bisel en los dos extremos)."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        L = float(np.linalg.norm(b - a))
        if L < 0.005:
            return None
        uvf = _uv_tubo(float(self.rng.uniform(0, 1)))
        return self.pieza("tubo", revolucion(perfil_tubo(R, L), uvf), a, marco_tubo(b - a), self.mat_tubo(R), nombre)

    def copa(self, base, direccion, R, L, r_ext=None, nombre=None, cerrada=False):
        """Copa (manguito) con el hombro en 'base' y la boca en base + direccion * L."""
        m = revolucion(perfil_copa(R, L, r_ext, cerrada), _uv_tubo())
        return self.pieza("copa", m, base, marco_tubo(direccion), self.mat_tubo(R), nombre)

    def campana(self, boca, direccion_boca, R, L=None, nombre=None):
        """Campana (copa integral de una barra) con la boca en 'boca', abierta hacia direccion_boca."""
        L = L or largo_copa(R) + 0.012
        d = _unit(direccion_boca)
        m = revolucion(perfil_campana(R, L), _uv_tubo())
        return self.pieza("campana", m, np.asarray(boca, dtype=float) - d * L, marco_tubo(d), self.mat_tubo(R), nombre)

    def codo(self, esquina, d1, d2, R, nombre=None):
        """Codo en la esquina donde un tubo que llega en direccion d1 sale en direccion d2.
        Devuelve (T1, T2): fondos de las dos copas (donde acaban los tubos), o None si es recto."""
        c, d1, d2 = np.asarray(esquina, dtype=float), _unit(d1), _unit(d2)
        phi = float(np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0)))
        if phi < np.radians(5):
            return None
        rb = radio_curva(R)
        t = rb * np.tan(phi / 2)
        T1, T2 = c - d1 * t, c + d2 * t
        x = _perp(d1, d2)
        Rm = np.column_stack([x, np.cross(d1, x), d1])
        r_ext = R + 0.0024
        m = arco(rb, phi, r_ext, R - ESPESOR, _uv_tubo())
        mat = self.mat_tubo(R)
        self.pieza("codo", m, T1, Rm, mat, nombre)
        Lc = largo_copa(R)
        self.copa(T1, -d1, R, Lc, nombre=None if nombre is None else nombre + "_copa0")
        self.copa(T2, d2, R, Lc, nombre=None if nombre is None else nombre + "_copa1")
        return T1 - d1 * HOLGURA_FONDO, T2 + d2 * HOLGURA_FONDO

    def etiqueta(self, centro, eje, R):
        """Etiqueta adhesiva alrededor del tubo; la flecha apunta hacia +eje."""
        eje = _unit(eje)
        m = revolucion(perfil_anillo(R - 0.0004, R + 0.0006, 0.0, L_ETIQUETA, tag_ext="etq"),
                       _uv_etiqueta(L_ETIQUETA))
        return self.pieza("etiqueta", m, np.asarray(centro, dtype=float) - eje * L_ETIQUETA / 2,
                          marco_tubo(eje), self.mat_etiqueta(R))

    # ---------------- soportes ----------------
    def tuerca(self, base, eje, r=0.0075, alto=0.0065):
        """Tuerca hexagonal con la cara de abajo en 'base' y el eje hacia 'eje'."""
        m = revolucion(perfil_anillo(0.0035, r, 0.0, alto, tag_ext="x"), _uv_nada, n=6, facetado=True,
                       theta0=np.pi / 6)
        return self.pieza("tuerca", m, base, marco_tubo(eje, (1.0, 0.0, 0.0)), self.m_zinc)

    def abrazadera(self, centro, eje, R, z_techo):
        """Abrazadera isofonica colgada del techo con varilla roscada."""
        c = np.asarray(centro, dtype=float)
        Rm = marco_tubo(eje)
        arriba, lado = Rm[:, 0], Rm[:, 1]
        r_goma, r_banda = R + 0.003, R + 0.0045
        self.pieza("goma", revolucion(perfil_anillo(R - 0.0004, r_goma, -0.0125, 0.0125), _uv_nada),
                   c, Rm, self.m_goma)
        self.pieza("banda", revolucion(perfil_anillo(r_goma - 0.0002, r_banda, -0.0105, 0.0105), _uv_nada),
                   c, Rm, self.m_banda)
        Rl = np.column_stack([lado, Rm[:, 2], arriba])      # caja: x hacia el lado, z hacia arriba
        for s in (-1.0, 1.0):                               # orejetas con su tornillo
            p = c + lado * s * (r_banda + 0.008)
            self._primitiva(self.m_banda, type="box", pos=p, size=(0.0095, 0.0105, 0.006), quat=de._quat(Rl))
            self._primitiva(self.m_goma, type="box", pos=p - lado * s * 0.001, size=(0.0088, 0.0125, 0.0008),
                            quat=de._quat(Rl))
            self._primitiva(self.m_zinc, type="cylinder", fromto=np.r_[p - arriba * 0.011, p + arriba * 0.006],
                            size=0.0021)
            self.tuerca(p + arriba * 0.006, arriba, r=0.0042, alto=0.003)
        tope = c + arriba * r_banda
        self.tuerca(tope - arriba * 0.001, arriba)                    # tuerca soldada
        self.tuerca(tope + arriba * 0.016, arriba)                    # contratuerca
        if z_techo is not None and abs(arriba[2]) > 0.9 and z_techo > tope[2] + 0.03:
            arriba_v = np.array([0.0, 0.0, 1.0])
            techo = np.array([tope[0], tope[1], z_techo])
            self._primitiva(self.m_varilla, type="cylinder", fromto=np.r_[tope, techo], size=0.004)
            self.tuerca(techo - arriba_v * 0.0095, arriba_v)
            self._primitiva(self.m_zinc, type="cylinder", fromto=np.r_[techo - arriba_v * 0.003, techo],
                            size=0.016)

    def escudo(self, centro_techo, R):
        """Embellecedor del paso del tubo por el techo (la cara de abajo se ve desde el suelo)."""
        ro = max(2.4 * R, R + 0.02)
        perfil = [(R + 0.0005, 0.0, "x"), (ro - 0.004, 0.0, "x"), (ro, 0.003, "x"), (ro, 0.0055, "x"),
                  (R + 0.0005, 0.0055, "x")]
        p = np.asarray(centro_techo, dtype=float) - np.array([0.0, 0.0, 0.0057])
        return self.pieza("escudo", revolucion(perfil, _uv_nada), p, np.eye(3), self.m_escudo)


# =============================== RED DEL TECHO ===============================
def anadir_red(root, tubos, codos, z_techo, punto_libre, grupo, separacion=0.6):
    """Construye la red del techo. tubos: [(a, b, r)]; codos: [(punto, r)] donde se unen dos
    tubos; punto_libre: extremo que se queda abierto (el punto de conexion).
    Devuelve los tramos que se han anadido (subidas al techo) como [(a, b, r)], para que el
    escenario los tenga en cuenta (p.ej. al comprobar que el KUKA no choca con la red)."""
    k = Constructor(de._Deco(root, grupo, {}, {}))
    T = [(np.asarray(a, dtype=float), np.asarray(b, dtype=float), float(r)) for a, b, r in tubos]
    libre = None if punto_libre is None else np.asarray(punto_libre, dtype=float)
    fin = {}                                   # (tubo, extremo 0/1) -> donde acaba de verdad
    extra = []
    arriba = np.array([0.0, 0.0, 1.0])

    def otro(i, j):
        return T[i][1 - j]

    # codos donde se juntan dos tubos
    for c, _ in codos:
        c = np.asarray(c, dtype=float)
        ext = [(i, j) for i, t in enumerate(T) for j in (0, 1) if np.linalg.norm(t[j] - c) < 1e-4]
        if len(ext) != 2 or abs(T[ext[0][0]][2] - T[ext[1][0]][2]) > 1e-6:
            r = max(T[i][2] for i, _ in ext) if ext else 0.02
            k._primitiva(k.mat_tubo(r), type="sphere", pos=c, size=radio_copa(r) + 0.002)
            continue
        (i, ji), (j, jj) = ext
        res = k.codo(c, c - otro(i, ji), otro(j, jj) - c, T[i][2])
        if res is not None:
            fin[(i, ji)], fin[(j, jj)] = res

    # extremos libres: el punto de conexion se queda abierto; los demas suben al techo
    for i, (a, b, r) in enumerate(T):
        for j, p in enumerate((a, b)):
            if (i, j) in fin:
                continue
            if libre is not None and np.linalg.norm(p - libre) < 1e-3:
                continue
            d1 = _unit(p - otro(i, j))
            rb = radio_curva(r)
            if SUBIR_AL_TECHO and abs(d1[2]) < 0.3 and z_techo is not None and z_techo - p[2] > rb + 0.01:
                res = k.codo(p, d1, arriba, r)
                fin[(i, j)] = res[0]
                cima = np.array([p[0], p[1], z_techo + 0.02])
                k.tubo(res[1], cima, r)
                k.escudo(np.array([p[0], p[1], z_techo]), r)
                extra.append((res[1].copy(), cima, r))
            else:                                               # tapon
                Lc = largo_copa(r)
                k.copa(p + d1 * HOLGURA_FONDO, -d1, r, Lc, cerrada=True)

    # tubos, abrazaderas, uniones y etiquetas
    for i, (a, b, r) in enumerate(T):
        k.tubo(fin.get((i, 0), a), fin.get((i, 1), b), r, nombre=f"tubo_techo_{i}")
        largo = float(np.linalg.norm(b - a))
        e = _unit(b - a)
        n = max(1, int(largo // separacion))
        for kk in range(n):
            k.abrazadera(a + e * largo * (kk + 0.5) / n, e, r, z_techo)
        f_etq = 0.25 if n == 1 else 1.0 / n
        if largo >= ETIQUETA_MIN:
            k.etiqueta(a + e * largo * f_etq, e, r)
        huecos = [kk / n for kk in range(1, n) if abs(kk / n - f_etq) > 1e-9]
        for barra in range(1, int(largo // BARRA) + 1):
            if huecos:
                f = min(huecos, key=lambda x: abs(x * largo - barra * BARRA))
                huecos.remove(f)
                k.campana(a + e * largo * f, -e, r)       # la boca mira aguas arriba (hacia a)
    return extra


# =============================== PIEZA SUELTA ================================
def pieza_codo(root, cuerpo, R, r_copa, largo_manguito, extremos, grupo, prefijo="tubo_suelto_"):
    """La pieza suelta de PVC (codo con un tramo y un manguito en cada lado) dentro de 'cuerpo'
    (el mocap). extremos: [(punto, direccion hacia fuera)] x 2, en coordenadas de la pieza, con
    el codo en el origen. Los geoms se llaman prefijo + ... (la logica los busca asi)."""
    k = Constructor(de._Deco(root, grupo, {}, {}), padre=cuerpo, extra={"mass": "0"})
    (pa, ea), (pb, eb) = [(np.asarray(p, dtype=float), _unit(e)) for p, e in extremos]
    res = k.codo(np.zeros(3), -eb, ea, R, nombre=prefijo + "codo")
    T1, T2 = res if res is not None else (np.zeros(3), np.zeros(3))
    for idx, (p, e, inicio) in enumerate(((pa, ea, T2), (pb, eb, T1))):
        Lm = largo_manguito
        k.tubo(inicio, p - e * (Lm - HOLGURA_FONDO), R, nombre=f"{prefijo}tramo{idx}")
        k.copa(p - e * Lm, e, R, Lm, r_ext=r_copa, nombre=f"{prefijo}copa{idx}")
