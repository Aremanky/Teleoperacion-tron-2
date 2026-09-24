"""
Paso 1 - Cargar el TRON 2 (variante de brazos) en MuJoCo con la base fija.

Uso:
  git clone https://github.com/limxdynamics/tron2-robot-description.git
  pip install "mujoco>=3.2" numpy
  python ver_tron2.py tron2a/DACH_TRON2A/xml/robot.xml
  (ejecutar desde la raíz del repo; también existe robot_grasper.xml y la carpeta tron2b)
"""
import os
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else \
    "tron2a/DACH_TRON2A/xml/robot.xml"

# ---------------------------------------------------------------
# 1) Quitar la base flotante para fijar el torso al mundo
# ---------------------------------------------------------------
tree = ET.parse(src)
root = tree.getroot()
removed = 0
for parent in root.iter():
    for child in list(parent):
        if child.tag == "freejoint" or (child.tag == "joint" and child.get("type") == "free"):
            parent.remove(child)
            removed += 1

# Si había keyframes, quitar los 7 valores de qpos (y 6 de qvel) de la base libre
if removed:
    for key in root.iter("key"):
        if key.get("qpos"):
            key.set("qpos", " ".join(key.get("qpos").split()[7:]))
        if key.get("qvel"):
            key.set("qvel", " ".join(key.get("qvel").split()[6:]))

# Se guarda junto al original para que las rutas relativas de mallas/includes sigan funcionando
fixed = os.path.join(os.path.dirname(os.path.abspath(src)), "robot_fixed.xml")
tree.write(fixed)
print(f"Free joints eliminados: {removed}  ->  {fixed}")

# ---------------------------------------------------------------
# 2) Cargar e inspeccionar
# ---------------------------------------------------------------
m = mujoco.MjModel.from_xml_path(fixed)
d = mujoco.MjData(m)
name = lambda obj, i: mujoco.mj_id2name(m, obj, i)

print(f"\nnq={m.nq}  nv={m.nv}  nu(actuadores)={m.nu}  ncam={m.ncam}\n")
print("ARTICULACIONES:")
for j in range(m.njnt):
    lo, hi = m.jnt_range[j]
    tipo = ["free", "ball", "slide", "hinge"][m.jnt_type[j]]
    print(f"  {j:2d}  {name(mujoco.mjtObj.mjOBJ_JOINT, j):32s} {tipo:6s} [{lo:+.2f}, {hi:+.2f}]")

print("\nACTUADORES:")
for a in range(m.nu):
    print(f"  {a:2d}  {name(mujoco.mjtObj.mjOBJ_ACTUATOR, a)}")

print("\nCÁMARAS:", [name(mujoco.mjtObj.mjOBJ_CAMERA, c) for c in range(m.ncam)])
print("SITES:  ", [name(mujoco.mjtObj.mjOBJ_SITE, s) for s in range(m.nsite)])

# ---------------------------------------------------------------
# 3) Visor en modo cinemático: se escribe qpos y se llama a mj_forward
#    (sin física). Todas las bisagras oscilan un poco para comprobar
#    que el modelo responde.
# ---------------------------------------------------------------
hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
t0 = time.time()
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        t = time.time() - t0
        for j in hinges:
            q = 0.4 * np.sin(t + 0.3 * j)
            if m.jnt_limited[j]:
                q = np.clip(q, *m.jnt_range[j])
            d.qpos[m.jnt_qposadr[j]] = q
        mujoco.mj_forward(m, d)
        v.sync()
        time.sleep(0.01)