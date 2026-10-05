"""
Geometria 3D de la mano a partir de lo que ven una o dos camaras.

Hand Landmarker da, por camara, 21 PIXELES muy precisos y 21 puntos 3D "world" cuya
profundidad relativa es mala (esta aplastada: PRONO_Z_ESCALA del proyecto ARCTOS). Con
dos OAK-D situadas una respecto a la otra (calibrar_extrinseca.py) se puede TRIANGULAR
cada landmark: 3D metrico real, sin la z inventada, y la mano no se pierde si una
camara deja de verla. Con una sola camara, el "2.5D": pixeles + profundidad de la
palma (OAK-D) + diferencias de profundidad del modelo.

Marcos: R_i (3x3) lleva del marco de la camara i al marco W; t_i es el origen de la
camara i en W (los R_op / t de fusion.FusionCamaras). K_i intrinsecas de la imagen en
la que estan los pixeles.

Funciones puras (sin MediaPipe): se prueban en prueba_manos3d() con una mano sintetica.
"""
import numpy as np

PALMA_PUNTOS = (0, 5, 9, 13, 17)
MUNECA, INDICE_MCP, CORAZON_MCP, MENIQUE_MCP = 0, 5, 9, 17
# tamano plausible de una mano adulta: distancia muneca -> nudillo del corazon (m)
PALMA_MIN, PALMA_MAX = 0.055, 0.13
REPROY_MAX_PX = 6.0        # error de reproyeccion mediano maximo (px a 640 de ancho) para creerse la triangulacion
ANGULO_RAYOS_MIN = 8.0     # grados entre los rayos de las dos camaras: por debajo, triangular es inestable


class Punto3D:
    """Punto con .x .y .z (lo que espera retargeting.py en los landmarks 'mundo')."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


def a_puntos(P):
    return [Punto3D(*p) for p in np.asarray(P, float)]


def a_array(mundo):
    lista = getattr(mundo, "landmark", mundo)
    return np.array([[l.x, l.y, l.z] for l in lista], dtype=float)


def rayos(px, K):
    """Direcciones unitarias (en el marco de la camara) de los pixeles px (N x 2)."""
    px = np.asarray(px, float)
    r = np.column_stack([(px[:, 0] - K[0, 2]) / K[0, 0], (px[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(px))])
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def proyectar(P_cam, K):
    """Pixeles de puntos en el marco de la camara (N x 3) -> (N x 2)."""
    P_cam = np.asarray(P_cam, float)
    z = np.where(np.abs(P_cam[:, 2]) < 1e-6, 1e-6, P_cam[:, 2])
    return np.column_stack([K[0, 0] * P_cam[:, 0] / z + K[0, 2], K[1, 1] * P_cam[:, 1] / z + K[1, 2]])


def levantar_25d(px, K, z_palma, mundo):
    """Puntos 3D (marco de la camara, metros) de los 21 landmarks con la profundidad de la
    OAK-D en la palma y las diferencias de profundidad del modelo (hand_world_landmarks)
    para el resto. Las coordenadas laterales salen de los PIXELES (exactas a esa z)."""
    px = np.asarray(px, float)
    W = a_array(mundo)
    dz = W[:, 2] - W[list(PALMA_PUNTOS), 2].mean()
    z = z_palma + dz
    z = np.maximum(z, 0.05)
    X = (px[:, 0] - K[0, 2]) * z / K[0, 0]
    Y = (px[:, 1] - K[1, 2]) * z / K[1, 1]
    return np.column_stack([X, Y, z])


def triangular_punto(origenes, direcciones):
    """Punto mas cercano (minimos cuadrados) a varias rectas (origen + s * direccion)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origenes, direcciones):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.solve(A + 1e-9 * np.eye(3), b)


def triangular(vistas):
    """vistas: [(px (21x2), K, R_cam_a_W, t_cam_en_W), ...] de la MISMA mano en el mismo instante.
    Devuelve (P (21x3) en W, error de reproyeccion mediano por vista [px], angulo entre
    rayos [deg]) o None si no hay al menos dos vistas."""
    if len(vistas) < 2:
        return None
    n = len(vistas[0][0])
    dirs_w = [(np.asarray(R, float) @ rayos(px, K).T).T for px, K, R, t in vistas]
    origenes = [np.asarray(t, float) for _, _, _, t in vistas]
    P = np.array([triangular_punto(origenes, [d[k] for d in dirs_w]) for k in range(n)])
    errores = []
    for (px, K, R, t), d in zip(vistas, dirs_w):
        R = np.asarray(R, float)
        P_cam = (P - np.asarray(t, float)) @ R          # W -> camara (R^T p)
        errores.append(float(np.median(np.linalg.norm(proyectar(P_cam, K) - px, axis=1))))
    # angulo entre los rayos del centro de la palma de las dos primeras vistas
    c0 = dirs_w[0][list(PALMA_PUNTOS)].mean(axis=0)
    c1 = dirs_w[1][list(PALMA_PUNTOS)].mean(axis=0)
    ang = float(np.degrees(np.arccos(np.clip(np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1)), -1, 1))))
    return P, errores, ang


def tamano_palma(P):
    P = np.asarray(P, float)
    return float(np.linalg.norm(P[CORAZON_MCP] - P[MUNECA]))


def triangulacion_valida(P, errores, ang, max_px=REPROY_MAX_PX):
    """True si la mano triangulada es creible: reproyecta bien en las dos camaras, los rayos
    no son casi paralelos y la palma tiene un tamano humano."""
    if P is None or not np.all(np.isfinite(P)):
        return False
    if max(errores) > max_px or ang < ANGULO_RAYOS_MIN:
        return False
    return PALMA_MIN < tamano_palma(P) < PALMA_MAX


MCPS = (5, 9, 13, 17)
PLANO_PALMA = (0, 1, 2, 5, 9, 13, 17)


def marco_de_puntos(P, robusto=True):
    """Marco semantico (columnas: dedos, ancho indice->menique, normal) de 21 puntos 3D.
    dedos = de la muneca al centro de los 4 nudillos MCP; ancho = del MCP del indice al del
    menique; normal = dedos x ancho. Con robusto=True la normal sale del PLANO de la palma
    (PCA de 7 puntos: mucho menos ruido que el producto de dos vectores cortos) y los otros
    dos ejes se proyectan sobre el. Es la misma construccion que SeguidorManos.marco() y
    que el marco del robot en mano_orca.ManoOrca."""
    P = np.asarray(P, float)
    f = P[list(MCPS)].mean(axis=0) - P[MUNECA]
    a = P[MENIQUE_MCP] - P[INDICE_MCP]
    nf = np.linalg.norm(f)
    if nf < 1e-6 or np.linalg.norm(a) < 1e-6:
        return None
    f = f / nf
    n = np.cross(f, a)
    if np.linalg.norm(n) < 1e-9:
        return None
    n /= np.linalg.norm(n)
    if robusto:
        Q = P[list(PLANO_PALMA)]
        Q = Q - Q.mean(axis=0)
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        n_pca = Vt[-1]
        if np.dot(n_pca, n) < 0:
            n_pca = -n_pca
        n = n_pca
        f = f - n * np.dot(f, n)
        f /= np.linalg.norm(f)
    a = np.cross(n, f)
    return np.column_stack([f, a, n])


def centrar(P):
    """Puntos relativos a su centro geometrico (como los hand_world_landmarks)."""
    P = np.asarray(P, float)
    return P - P.mean(axis=0)


def extrapolar_px(px, px_prev, t, t_prev, t_obj, factor=0.7, max_dt=0.06):
    """Adelanta los pixeles de t a t_obj con la velocidad medida entre t_prev y t."""
    if px_prev is None or t_prev is None or not (1e-3 < t - t_prev < 0.2):
        return np.asarray(px, float)
    dt = float(np.clip(t_obj - t, 0.0, max_dt))
    return np.asarray(px, float) + factor * (np.asarray(px, float) - np.asarray(px_prev, float)) / (t - t_prev) * dt


# ------------------------------------------------------------------ prueba
def _mano_sintetica():
    """21 puntos (m) de una mano derecha abierta, palma en el plano z=0, dedos hacia +y."""
    P = np.zeros((21, 3))
    P[0] = (0.0, 0.0, 0.0)
    bases = {5: (0.03, 0.085), 9: (0.01, 0.09), 13: (-0.01, 0.087), 17: (-0.03, 0.08)}
    largos = {5: 0.075, 9: 0.085, 13: 0.08, 17: 0.065}
    for b, (x, y) in bases.items():
        P[b] = (x, y, 0.0)
        for k in range(1, 4):
            P[b + k] = (x, y + largos[b] * k / 3.0, -0.004 * k)
    P[1] = (0.035, 0.02, 0.0)
    P[2] = (0.055, 0.04, 0.005)
    P[3] = (0.07, 0.055, 0.01)
    P[4] = (0.085, 0.07, 0.012)
    return P


def prueba_manos3d(ruido_px=1.0, semilla=0, verbose=True):
    """Dos camaras virtuales a 2 m, 60 grados entre ellas, ven una mano girada al azar:
    la triangulacion debe recuperar los puntos con error de milimetros y el marco con
    error de pocos grados. Devuelve (error_mm, error_marco_deg)."""
    rng = np.random.default_rng(semilla)
    K = np.array([[465.0, 0.0, 320.0], [0.0, 465.0, 240.0], [0.0, 0.0, 1.0]])
    # orientacion aleatoria de la mano y posicion delante de las camaras
    w = rng.normal(size=3)
    w = w / np.linalg.norm(w) * rng.uniform(0, np.pi)
    from suavizado import exp_rot
    R_mano = exp_rot(w)
    P_w = _mano_sintetica() @ R_mano.T + np.array([0.1, -0.2, 0.3])
    vistas, pxs = [], []
    for ang in (-30.0, 30.0):
        a = np.radians(ang)
        # camara mirando al origen desde 2 m, girada 'ang' alrededor de la vertical (y de W = y camara)
        t = np.array([2.0 * np.sin(a), 0.0, -2.0 * np.cos(a)])
        z_cam = -t / np.linalg.norm(t)                        # mira al origen
        x_cam = np.cross([0.0, 1.0, 0.0], z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        R = np.column_stack([x_cam, y_cam, z_cam])           # camara -> W
        P_cam = (P_w - t) @ R
        px = proyectar(P_cam, K) + rng.normal(scale=ruido_px, size=(21, 2))
        vistas.append((px, K, R, t))
        pxs.append(px)
    P, err, ang = triangular(vistas)
    e_mm = float(np.mean(np.linalg.norm(P - P_w, axis=1))) * 1000
    Rm = marco_de_puntos(P)
    Rv = marco_de_puntos(P_w)
    e_deg = float(np.degrees(np.arccos(np.clip((np.trace(Rm @ Rv.T) - 1) / 2, -1, 1))))
    if verbose:
        print(f"triangulacion: error medio {e_mm:.1f} mm | reproyeccion {np.round(err, 2)} px | "
              f"angulo entre rayos {ang:.0f} deg | marco {e_deg:.1f} deg | valida: "
              f"{triangulacion_valida(P, err, ang)}")
    return e_mm, e_deg


if __name__ == "__main__":
    for s in range(5):
        prueba_manos3d(semilla=s)
