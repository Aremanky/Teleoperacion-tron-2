#!/bin/bash

# Comprobar si el directorio .venv no existe
if [ ! -d ".venv" ]; then
    echo "El entorno virtual no existe. Creando .venv con python3.12..."
    python3.12 -m venv .venv
else
    echo "El entorno virtual .venv ya existe."
fi

# Activar el entorno virtual
echo "Activando el entorno virtual..."
source .venv/bin/activate

# Instalar dependencias si existe el archivo requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Instalando dependencias desde requirements.txt..."
    pip install "mujoco>=3.2" numpy "mujoco>=3.2" "depthai>=3" mediapipe
else
    echo "Advertencia: No se encontró requirements.txt, saltando la instalación."
fi

# Ejecutar el script principal
echo "Arrancando teleop_tron2.py..."
python teleop_tron2.py

# Desactivar el entorno al finalizar (opcional, pero buena práctica si se corre con 'source')
deactivate
