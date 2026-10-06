"""
Geometria del brazo humano que comparten seguimiento_brazos.py (una camara) y fusion.py
(varias camaras).

codo_en_circulo: cuando NADIE ve el codo (brazo apuntando a la camara, oclusion), antes se
daba el brazo por RECTO de hombro a muneca, lo que hacia saltar la muneca del robot 20-30 cm
cuando el codo estaba doblado. Con las longitudes del operador (LongitudesHumano) el codo
esta en una CIRCUNFERENCIA alrededor de la recta hombro-muneca; se elige el punto mas
cercano al ultimo codo conocido (mismo "giro" del codo), y si no hay ninguno, el codo hacia
abajo y hacia fuera, que es como cuelga un codo humano relajado.
"""
import numpy as np


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def codo_en_circulo(hombro, muneca, L_brazo, L_antebrazo, codo_previo=None, abajo=None, fuera=None):
    """Posicion plausible del codo con el hombro y la muneca conocidos.
    Devuelve (codo, reconstruido): reconstruido=False si la geometria obliga a brazo recto."""
    S, W = np.asarray(hombro, float), np.asarray(muneca, float)
    d = W - S
    D = float(np.linalg.norm(d))
    if D < 1e-6:
        return S + np.array([0.0, 0.0, -L_brazo]), False
    u = d / D
    if D >= L_brazo + L_antebrazo - 0.005:          # estirado del todo: recto
        return S + L_brazo * u, False
    if D <= abs(L_brazo - L_antebrazo) + 0.005:     # imposible (muneca pegada al hombro): recto hacia la muneca
        return S + L_brazo * u, False
    a = (L_brazo ** 2 - L_antebrazo ** 2 + D ** 2) / (2.0 * D)
    r = float(np.sqrt(max(L_brazo ** 2 - a ** 2, 1e-9)))
    c = S + a * u
    # referencia del giro del codo: el codo anterior; si no, hacia abajo (+ un poco hacia fuera)
    v = None
    if codo_previo is not None:
        v = np.asarray(codo_previo, float) - c
        v = v - u * np.dot(v, u)
        if np.linalg.norm(v) < 0.02:
            v = None
    if v is None:
        ref = np.array([0.0, 0.0, -1.0]) if abajo is None else -np.asarray(abajo, float)
        if fuera is not None:
            ref = ref + 0.5 * np.asarray(fuera, float)
        v = ref - u * np.dot(ref, u)
        if np.linalg.norm(v) < 1e-3:                # recta vertical: cualquier perpendicular
            v = np.cross(u, [1.0, 0.0, 0.0])
    return c + r * _unit(v), True


def direcciones(hombro, codo, muneca):
    """(direccion del brazo, direccion del antebrazo) unitarias, o (None, None)."""
    d_b, d_a = _unit(np.asarray(codo, float) - np.asarray(hombro, float)), _unit(np.asarray(muneca, float) - np.asarray(codo, float))
    return d_b, d_a


# ---------------------------------------------------------------- coherencia del codo (oct-2026)
# Con una sola camara de frente la profundidad del codo es la peor medida: si sale unos cm
# adelantado, el brazo parece flexionado hacia delante y el antebrazo doblado HACIA ATRAS. Eso
# exige girar el humero ~150-180 grados, algo que ni una persona ni el robot pueden hacer, y la
# IK se queda con un error de codo enorme o retuerce el hombro. En la sesion de prueba, con solo
# la camara A, el 55 % de las fotos del brazo izquierdo con el codo doblado eran asi (0 % con A+B).
# limitar_rotacion_humeral() gira el codo alrededor de la recta hombro-muneca (la muneca NO se
# mueve y la flexion del codo tampoco) hasta la rotacion humeral factible mas cercana.

_ABAJO = np.array([0.0, 0.0, -1.0])


def _cruz(a, b):
    """np.cross sin su coste fijo (aqui se llama muchas veces con arrays pequenos)."""
    a, b = np.broadcast_arrays(a, b)
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)
_DELANTE = np.array([1.0, 0.0, 0.0])


def _swing(db, v):
    """v girado con la rotacion MINIMA que lleva 'abajo' a db (vectorizado: db y v (N, 3))."""
    k = _cruz(_ABAJO, db)
    s = np.linalg.norm(k, axis=-1, keepdims=True)
    c = np.sum(db * _ABAJO, axis=-1, keepdims=True)
    k = np.where(s > 1e-9, k / np.maximum(s, 1e-12), np.array([1.0, 0.0, 0.0]))
    s = np.where(s > 1e-9, s, 0.0)
    c = np.where(s > 0, c, np.sign(c) + (c == 0))
    v = np.broadcast_to(v, db.shape)
    return v * c + _cruz(k, v) * s + k * np.sum(k * v, axis=-1, keepdims=True) * (1 - c)


def rotacion_humeral(db, da, fuera):
    """Rotacion humeral (grados) en el marco del robot (x delante, y izquierda, z arriba), con el
    mismo convenio que robot_tron2: 0 = antebrazo hacia delante con el brazo colgando y el codo a
    90, + = externa (antebrazo hacia fuera). 'fuera': +y (izquierdo) o -y (derecho). Vectorizado."""
    db = np.atleast_2d(np.asarray(db, float))
    da = np.atleast_2d(np.asarray(da, float))
    p = da - db * np.sum(da * db, axis=-1, keepdims=True)
    ref, out = _swing(db, _DELANTE), _swing(db, np.asarray(fuera, float))
    return np.degrees(np.arctan2(np.sum(p * out, axis=-1), np.sum(p * ref, axis=-1)))


def _girar(v, eje, ang):
    """Rodrigues vectorizado: v (3,) girado 'ang' (N,) rad alrededor de 'eje' unitario (3,)."""
    ang = np.asarray(ang, float)[:, None]
    return v * np.cos(ang) + _cruz(eje, v) * np.sin(ang) + eje * np.dot(eje, v) * (1 - np.cos(ang))


def limitar_rotacion_humeral(d_b, d_a, fuera, rango, L_b=0.30, L_a=0.27, psi_previo=0.0,
                             flex_min=12.0, elev_max=160.0, paso=6.0):
    """Gira el codo alrededor de la recta hombro-muneca hasta que la rotacion humeral quede en
    rango(elevacion) -> (min, max) en grados (vectorizado: con un array de elevaciones devuelve
    dos arrays). Devuelve (d_b, d_a, psi en grados): psi = cuanto se
    ha girado el codo (0 = nada que corregir). Con el codo casi recto o el brazo casi vertical la
    rotacion no esta definida y no se toca nada. psi_previo: correccion de la foto anterior; entre
    dos soluciones parecidas se prefiere la del mismo lado (sin saltos de un lado a otro)."""
    d_b = np.asarray(d_b, float) / np.linalg.norm(d_b)
    d_a = np.asarray(d_a, float) / np.linalg.norm(d_a)
    flex = np.degrees(np.arccos(np.clip(np.dot(d_b, d_a), -1.0, 1.0)))
    elev = np.degrees(np.arccos(np.clip(-d_b[2], -1.0, 1.0)))
    # cuanto se aplica la correccion: 0 con el codo casi recto o el brazo casi vertical (ahi la
    # rotacion no esta definida) y 1 con margen; la rampa evita saltos al cruzar los umbrales
    alfa = _rampa((flex - flex_min) / 10.0) * _rampa((elev_max - elev) / 15.0)
    if alfa <= 0.0:
        return d_b, d_a, 0.0
    rmin, rmax = rango(elev)
    rot = float(rotacion_humeral(d_b, d_a, fuera)[0])
    if rmin <= rot <= rmax:
        return d_b, d_a, 0.0
    c = L_b * d_b
    w = c + L_a * d_a
    u = w / np.linalg.norm(w)

    def evaluar(psis):
        cs = _girar(c, u, np.radians(psis))
        dbs = cs / L_b
        das = (w - cs) / L_a
        el = np.degrees(np.arccos(np.clip(-dbs[:, 2], -1.0, 1.0)))
        rs = rotacion_humeral(dbs, das, fuera)
        lo_r, hi_r = rango(el)
        return dbs, das, (rs >= lo_r) & (rs <= hi_r)

    psis = np.arange(-180.0, 180.0 + 1e-9, paso)
    _, _, ok = evaluar(psis)
    if not ok.any():
        return d_b, d_a, 0.0
    coste = np.abs(psis) - np.where(np.sign(psis) == np.sign(psi_previo), 20.0, 0.0) * (psi_previo != 0)
    coste[~ok] = np.inf
    psi_ok = psis[int(np.argmin(coste))]
    # afinar el borde: biseccion entre el ultimo infactible y el primero factible hacia psi = 0
    lo, hi = psi_ok - np.sign(psi_ok) * paso, psi_ok
    for _ in range(5):
        mid = 0.5 * (lo + hi)
        if evaluar(np.array([mid]))[2][0]:
            hi = mid
        else:
            lo = mid
    psi = alfa * hi
    dbs, das, _ = evaluar(np.array([psi]))
    return dbs[0], das[0], float(psi)


def _rampa(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)
