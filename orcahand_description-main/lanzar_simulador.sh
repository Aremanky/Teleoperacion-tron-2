#!/bin/bash
# Navegar a la carpeta del proyecto
cd "$(dirname "$0")"
# Ejecutar el visor usando directamente el python del entorno virtual
./.venv/bin/python -m mujoco.viewer --mjcf="$(pwd)/v2/scene_right.xml"
