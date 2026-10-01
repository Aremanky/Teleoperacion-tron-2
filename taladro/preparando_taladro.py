#!/usr/bin/env python3
"""
Prepara el taladro (drill.obj) para MuJoCo y genera escena_taladro.xml.

Ejecutar UNA vez desde la raíz del proyecto (o cada vez que cambies algo aquí):
    python taladro/preparar_taladro.py

Qué hace:
  1. Lee taladro/drill.obj (centímetros, eje Y hacia arriba, broca hacia +Z).
  2. Lo pasa a METROS y a los ejes de MuJoCo:
        x = hacia donde apunta la broca · y = lateral · z = arriba (empuñadura abajo)
     con el origen en el centro de la EMPUÑADURA (así es fácil colocarlo y cogerlo).
  3. Lo separa en 5 piezas para darles color: cuerpo, empuñadura, batería,
     portabrocas y broca (estas dos giran juntas).
  4. Calcula colisiones SIMPLES (cápsulas y cajas): MuJoCo usaría la envolvente
     convexa del mallado entero, que es un "bloque" imposible de agarrar.
  5. Escribe escena_taladro.xml junto a arctos_mjcf.xml, con rutas ABSOLUTAS a
     las piezas (así no depende del meshdir del modelo).

Luego hay que añadir UNA línea en arctos_mjcf.xml, junto al include de la mesa:
    <include file="escena_taladro.xml"/>
"""
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
try:
    from config import RUTA_XML            # para saber dónde está arctos_mjcf.xml
except Exception:
    RUTA_XML = None

OBJ_ENTRADA = os.path.join(AQUI, "drill.obj")
DIR_PIEZAS = os.path.join(AQUI, "piezas")
XML_SALIDA = os.path.join(AQUI, "escena_taladro.xml")

MASA_TALADRO = 0.6      # kg (uno real pesa ~1.2 kg; más ligero = más fácil de sujetar en simulación)
N_MARCAS     = 8        # agujeros que se pueden dejar marcados en el techo

# pieza → (grupos del .obj, color rgba)
PIEZAS = {
    "cuerpo":      (("C_body_geo", "C_torque_geo", "C_rearCap_geo", "C_gearSelector_geo",
                     "L_motorPadLarge_geo", "L_motorPadSmall_geo", "R_motorPadLarge_geo",
                     "R_motorPadSmall_geo", "L_cooling_geo", "L_ScrewN_geo", "R_screwA_geo",
                     "R_screwB_geo", "R_screwC_geo", "L_signReverse_geo", "R_signForward_geo",
                     "C_reverseButton_geo", "C_screwDrive_geo"), "0.95 0.72 0.10 1"),
    "empunadura": (("C_gripBack_geo", "C_gripFront_geo", "C_trigger_geo"), "0.12 0.12 0.12 1"),
    "bateria":    (("C_battery_geo", "L_batteryRelease_geo", "R_batteryRelease_geo"), "0.25 0.25 0.27 1"),
    "portabrocas": (("C_chuckAdjuster_geo", "C_torqueRing_geo", "C_chuckRing_geo",
                     "C_chuckA_geo", "C_chuckB_geo", "C_chuckC_geo"), "0.10 0.10 0.10 1"),
    "broca":      (("C_bit_geo",), "0.78 0.78 0.80 1"),
}
GIRAN = ("portabrocas", "broca")        # van en un cuerpo hijo con bisagra (la broca gira)


def leer_obj(ruta):
    V, caras, grupo = [], {}, None
    with open(ruta) as f:
        for linea in f:
            if linea.startswith("v "):
                V.append([float(t) for t in linea.split()[1:4]])
            elif linea.startswith("g "):
                grupo = linea.split()[1]
                caras.setdefault(grupo, [])
            elif linea.startswith("f "):
                idx = [int(t.split("/")[0]) for t in linea.split()[1:]]
                idx = [i - 1 if i > 0 else len(V) + i for i in idx]
                for k in range(1, len(idx) - 1):            # triangulación en abanico
                    caras.setdefault(grupo, []).append((idx[0], idx[k], idx[k + 1]))
    return np.array(V), caras


def a_mujoco(P_cm):
    """(x, y, z) del .obj en cm → metros con x = broca, y = lateral, z = arriba."""
    P = np.asarray(P_cm) * 0.01
    return np.column_stack([P[:, 2], P[:, 0], P[:, 1]])


def capsula_pca(P, pct_radio=70):
    """Cápsula que envuelve la nube P siguiendo su dirección principal."""
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    eje = vt[0]
    t = (P - c) @ eje
    d = np.linalg.norm((P - c) - np.outer(t, eje), axis=1)
    r = float(np.percentile(d, pct_radio))
    a, b = c + eje * (t.min() + r), c + eje * (t.max() - r)
    return a, b, r


def escribir_obj(ruta, V, tris):
    usados = sorted({i for t in tris for i in t})
    nuevo = {v: k + 1 for k, v in enumerate(usados)}
    with open(ruta, "w") as f:
        f.write("# Pieza del taladro — metros, ejes MuJoCo (generado por preparar_taladro.py)\n")
        for v in usados:
            f.write(f"v {V[v, 0]:.6f} {V[v, 1]:.6f} {V[v, 2]:.6f}\n")
        for a, b, c in tris:
            f.write(f"f {nuevo[a]} {nuevo[b]} {nuevo[c]}\n")


def f3(v):
    return " ".join(f"{x:.4f}" for x in v)


def main():
    if not os.path.exists(OBJ_ENTRADA):
        sys.exit(f"No encuentro {OBJ_ENTRADA}")
    V_cm, caras = leer_obj(OBJ_ENTRADA)
    V = a_mujoco(V_cm)

    def puntos(grupos):
        idx = sorted({i for g in grupos for t in caras.get(g, []) for i in t})
        return V[idx]

    # Origen = centro de la empuñadura
    origen = puntos(PIEZAS["empunadura"][0]).mean(axis=0)
    V = V - origen
    # Eje de giro de la broca: su centro en (y, z); pivote en el inicio del portabrocas
    Pb = puntos(PIEZAS["broca"][0])
    Pch = puntos(PIEZAS["portabrocas"][0])
    pivote = np.array([Pch[:, 0].min(), Pb[:, 1].mean(), Pb[:, 2].mean()])
    punta = np.array([Pb[:, 0].max(), pivote[1], pivote[2]])

    os.makedirs(DIR_PIEZAS, exist_ok=True)
    for nombre, (grupos, _) in PIEZAS.items():
        tris = [t for g in grupos for t in caras.get(g, [])]
        Vp = V - pivote if nombre in GIRAN else V       # las que giran, relativas al pivote
        escribir_obj(os.path.join(DIR_PIEZAS, f"{nombre}.obj"), Vp, tris)

    # ── Colisiones simples ───────────────────────────────────────────────────
    ea, eb, er = capsula_pca(puntos(PIEZAS["empunadura"][0]), 60)
    Pc = puntos(("C_rearCap_geo", "C_torque_geo"))
    ca = np.array([Pc[:, 0].min(), pivote[1], Pc[:, 2].mean()])
    cb = np.array([pivote[0], pivote[1], Pc[:, 2].mean()])
    cr = 0.5 * float(np.percentile(Pc[:, 2], 95) - np.percentile(Pc[:, 2], 5)) * 0.9
    ca[0] += cr; cb[0] -= cr * 0.2
    Pbat = puntos(PIEZAS["bateria"][0])
    bat_c = (Pbat.min(0) + Pbat.max(0)) / 2
    bat_m = (Pbat.max(0) - Pbat.min(0)) / 2
    Pchuck = Pch - pivote
    ch_r = float(np.percentile(np.linalg.norm(Pchuck[:, 1:], axis=1), 90))
    ch_len = float(Pchuck[:, 0].max())
    broca_len = float(punta[0] - pivote[0])

    rgba = {n: c for n, (_, c) in PIEZAS.items()}
    pz = lambda n: os.path.join(DIR_PIEZAS, f"{n}.obj")

    marcas = "\n".join(
        f'''    <body name="taladro_agujero_{k}" mocap="true" pos="0 0 -10">
      <geom type="cylinder" size="0.006 0.0015" rgba="0.05 0.05 0.05 1"
            contype="0" conaffinity="0" group="2" mass="0"/>
    </body>''' for k in range(N_MARCAS))

    xml = f'''<!-- GENERADO por taladro/preparar_taladro.py — no editar a mano (vuelve a generarlo). -->
<mujoco>
  <asset>
    <mesh name="taladro_cuerpo"      file="{pz('cuerpo')}"/>
    <mesh name="taladro_empunadura"  file="{pz('empunadura')}"/>
    <mesh name="taladro_bateria"     file="{pz('bateria')}"/>
    <mesh name="taladro_portabrocas" file="{pz('portabrocas')}"/>
    <mesh name="taladro_broca"       file="{pz('broca')}"/>
  </asset>

  <worldbody>
    <!-- Techo donde se taladra (lo coloca [4] a la altura adecuada) -->
    <body name="taladro_techo" mocap="true" pos="0 0 -10">
      <geom name="taladro_techo_geom" type="box" size="0.60 0.60 0.015"
            rgba="0.85 0.85 0.82 1" friction="1 0.01 0.001"
            contype="1" conaffinity="1" group="0"/>
    </body>

    <!-- Peana donde espera el taladro (de pie, apoyado en la batería) -->
    <body name="taladro_soporte" mocap="true" pos="0 0 -10">
      <geom name="taladro_soporte_geom" type="box" size="0.08 0.08 0.01"
            rgba="0.35 0.30 0.25 1" friction="1 0.01 0.001"
            contype="1" conaffinity="1" group="0"/>
    </body>

    <!-- El taladro -->
    <body name="taladro" pos="0 0 -10">
      <freejoint name="taladro_libre"/>
      <inertial pos="{f3(0.45 * bat_c + 0.55 * (ca + cb) / 2)}" mass="{MASA_TALADRO}"
                diaginertia="0.0025 0.0030 0.0012"/>
      <!-- visual (no choca) -->
      <geom type="mesh" mesh="taladro_cuerpo"     rgba="{rgba['cuerpo']}"     contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="mesh" mesh="taladro_empunadura" rgba="{rgba['empunadura']}" contype="0" conaffinity="0" group="2" mass="0"/>
      <geom type="mesh" mesh="taladro_bateria"    rgba="{rgba['bateria']}"    contype="0" conaffinity="0" group="2" mass="0"/>
      <!-- colisiones simples (invisibles, grupo 3) -->
      <geom name="taladro_col_empunadura" type="capsule" fromto="{f3(ea)} {f3(eb)}" size="{er:.4f}"
            contype="1" conaffinity="1" group="3" rgba="1 0 0 0.3" condim="4" friction="1.5 0.02 0.001" mass="0"/>
      <geom name="taladro_col_cabeza" type="capsule" fromto="{f3(ca)} {f3(cb)}" size="{cr:.4f}"
            contype="1" conaffinity="1" group="3" rgba="1 0 0 0.3" condim="4" friction="1.2 0.02 0.001" mass="0"/>
      <geom name="taladro_col_bateria" type="box" pos="{f3(bat_c)}" size="{f3(bat_m)}"
            contype="1" conaffinity="1" group="3" rgba="1 0 0 0.3" condim="4" friction="1.2 0.02 0.001" mass="0"/>

      <!-- Portabrocas + broca: GIRAN alrededor del eje de la broca -->
      <body name="taladro_rotor" pos="{f3(pivote)}">
        <joint name="taladro_giro" type="hinge" axis="1 0 0" damping="0.0005"/>
        <inertial pos="{ch_len / 2:.4f} 0 0" mass="0.02" diaginertia="0.000005 0.00001 0.00001"/>
        <geom type="mesh" mesh="taladro_portabrocas" rgba="{rgba['portabrocas']}" contype="0" conaffinity="0" group="2" mass="0"/>
        <geom type="mesh" mesh="taladro_broca"       rgba="{rgba['broca']}"       contype="0" conaffinity="0" group="2" mass="0"/>
        <geom name="taladro_col_portabrocas" type="capsule" fromto="0.004 0 0 {ch_len - ch_r:.4f} 0 0"
              size="{ch_r:.4f}" contype="1" conaffinity="1" group="3" rgba="1 0 0 0.3" condim="4" mass="0"/>
        <!-- punta de la broca: es la que "toca" el techo -->
        <geom name="taladro_punta" type="sphere" pos="{broca_len:.4f} 0 0" size="0.004"
              contype="1" conaffinity="1" group="3" rgba="1 0 0 0.6" condim="3" mass="0"/>
      </body>
    </body>

    <!-- Marcas de los agujeros hechos (visuales) -->
{marcas}
  </worldbody>
</mujoco>
'''
    with open(XML_SALIDA, "w") as f:
        f.write(xml)

    tam = V.max(0) - V.min(0)
    print(f"✅ Piezas en: {DIR_PIEZAS}")
    print(f"✅ Escena:    {XML_SALIDA}")
    print(f"   Tamaño del taladro: {tam[0]*100:.1f} (largo) × {tam[1]*100:.1f} (ancho) × "
          f"{tam[2]*100:.1f} (alto) cm · masa {MASA_TALADRO} kg")
    print(f"   Empuñadura: cápsula de radio {er*100:.1f} cm · broca {broca_len*100:.1f} cm desde el portabrocas")
    print(f"   Base de la batería a {-(V[:, 2].min())*100:.1f} cm por debajo del centro de la empuñadura")
    print('\n👉 Añade en arctos_mjcf.xml, junto al include de la mesa:\n'
          '       <include file="escena_taladro.xml"/>')


if __name__ == "__main__":
    main()