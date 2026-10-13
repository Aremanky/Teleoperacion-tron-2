#!/usr/bin/env bash
# =============================================================================
#  arranque.sh - Arranque de la teleoperacion del TRON 2 (teleop_tron2.py)
#                version SIN guantes MANUS
#
#  1) Prepara el entorno virtual .venv/ (lo crea con python3.12 si no existe e
#     instala las dependencias).
#  2) Muestra un menu con las opciones de arranque: soporte (KUKA o columna
#     original), camaras, manos, escenario... Se cambian con su numero.
#  3) Enseña el comando final, pide confirmacion y lanza teleop_tron2.py.
#
#  Uso (desde la raiz del proyecto o desde cualquier sitio):
#     ./arranque.sh
# =============================================================================
set -u

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RAIZ" || exit 1
VENV="$RAIZ/.venv"
PY="$VENV/bin/python"
PAQUETES=("mujoco>=3.2" numpy "depthai>=3" mediapipe opencv-python)

# ----------------------------------------------------------------- colores
if [ -t 1 ]; then
    N=$'\e[1m'; D=$'\e[2m'; R=$'\e[0m'; V=$'\e[32m'; A=$'\e[33m'; C=$'\e[36m'; E=$'\e[31m'
else
    N=""; D=""; R=""; V=""; A=""; C=""; E=""
fi
info()  { echo "${C}==>${R} $*"; }
aviso() { echo "${A}[aviso]${R} $*"; }
error() { echo "${E}[error]${R} $*" >&2; }

# =============================================================================
# 1) ENTORNO VIRTUAL
# =============================================================================
if [ ! -d "$VENV" ]; then
    info "No existe .venv/: lo creo con python3.12"
    if ! command -v python3.12 >/dev/null 2>&1; then
        error "No encuentro python3.12. Instalalo (p.ej. sudo apt install python3.12 python3.12-venv)"
        exit 1
    fi
    python3.12 -m venv "$VENV" || { error "No se ha podido crear el entorno .venv/"; exit 1; }
    "$PY" -m pip install --upgrade pip
    info "Instalando: ${PAQUETES[*]}"
    "$PY" -m pip install "${PAQUETES[@]}" || { error "Fallo al instalar las dependencias"; exit 1; }
else
    info "Entorno .venv/ encontrado"
    # por si se creo a medias o alguien lo ha tocado
    if ! "$PY" -c "import mujoco, numpy, depthai, mediapipe, cv2" >/dev/null 2>&1; then
        aviso "Faltan dependencias en .venv/: las instalo"
        "$PY" -m pip install "${PAQUETES[@]}" || { error "Fallo al instalar las dependencias"; exit 1; }
    fi
fi

# =============================================================================
# 2) MENU
# =============================================================================
# Valores por defecto: lo habitual (dos OAK-D + OrcaHand, sobre el KUKA)
SOPORTE="kuka"          # kuka | columna
HAY_KUKA=0
[ -f "$RAIZ/montar_kuka.py" ] && HAY_KUKA=1
[ "$HAY_KUKA" = 0 ] && SOPORTE="columna"
CAMARAS="dos"           # una | dos | webcam | reproducir
WEBCAM_IDX=0
SESION=""               # con CAMARAS=reproducir
VELOCIDAD="1.0"
ORCA=1
ESCENARIO=0
MODELO="full"           # lite | full | heavy
GRABAR=""
GRABAR_VIDEO=0
SOLO_MEDIAPIPE=0
SIN_MANOS=0
SIN_GIRO=0
SIN_IMU=0
PROYECTOR="0.7"
SIN_HD=0
SIN_RECORTE=0
SIN_MANOS_B=0
SIN_ESTEREO_FINO=0
SIN_EXTRINSECAS=0

si_no() { [ "$1" = 1 ] && echo "${V}SI${R}" || echo "${D}no${R}"; }

texto_camaras() {
    case "$CAMARAS" in
        una)        echo "una OAK-D" ;;
        dos)        echo "dos OAK-D fusionadas" ;;
        webcam)     echo "webcam $WEBCAM_IDX (sin OAK-D)" ;;
        reproducir) echo "reproducir ${SESION:-<sin archivo>} (x$VELOCIDAD, sin camaras)" ;;
    esac
}

mostrar_menu() {
    clear 2>/dev/null || true
    cat <<EOF
${N}=====================================================================
   TELEOPERACION TRON 2 - opciones de arranque
=====================================================================${R}
 ${N}Robot${R}
  1) Soporte del TRON 2 ............ ${V}${SOPORTE}${R}
       ${D}kuka: montado sobre el KUKA KR210 | columna: soporte original del DACH_TRON2A${R}$([ "$HAY_KUKA" = 0 ] && echo "
       ${A}(sin montar_kuka.py en este proyecto: solo columna)${R}")
  2) Manos del robot ............... $([ "$ORCA" = 1 ] && echo "${V}OrcaHand${R}" || echo "${V}pinzas originales${R}")
       ${D}OrcaHand: dedos uno a uno (tecla g = pinzas predefinidas) | pinzas: solo abrir/cerrar${R}
  3) Escenario de tuberias al inicio  $(si_no $ESCENARIO)
       ${D}si no, aparece con la tecla 1 dentro del teleop${R}

 ${N}Entrada del operador${R}
  4) Camaras ....................... ${V}$(texto_camaras)${R}
       ${D}dos OAK-D: lo habitual, mejor profundidad y menos oclusiones${R}
  5) Modelo MediaPipe .............. ${V}${MODELO}${R}
       ${D}lite: mas rapido y menos preciso | full: equilibrio | heavy: mas preciso y lento${R}
  6) Grabar la sesion .............. $([ -n "$GRABAR" ] && echo "${V}${GRABAR}${R} (video $(si_no $GRABAR_VIDEO))" || echo "${D}no${R}")
       ${D}guarda un .pkl para reproducirlo o medir la imitacion (metricas_imitacion.py)${R}

  7) Opciones avanzadas  ${D}(profundidad, IMU, proyector IR, HD, extrinsecas...)${R}

  ${N}a)${R} ARRANCAR      ${N}q)${R} salir
EOF
}

menu_avanzado() {
    while true; do
        clear 2>/dev/null || true
        cat <<EOF
${N}---------------------- opciones avanzadas ----------------------${R}
  1) Solo MediaPipe (sin profundidad de la OAK-D) .. $(si_no $SOLO_MEDIAPIPE)
       ${D}util si la profundidad falla; los brazos salen peor en 3D${R}
  2) No seguir las manos ........................... $(si_no $SIN_MANOS)
       ${D}va mas rapido; el robot no mueve dedos ni orienta la palma${R}
  3) No orientar la palma con la mano .............. $(si_no $SIN_GIRO)
       ${D}la mano solo abre y cierra${R}
  4) No usar la IMU de las OAK-D ................... $(si_no $SIN_IMU)
       ${D}la IMU corrige la inclinacion de la camara${R}
  5) Proyector IR (0 = apagado, 0-1) ............... ${V}${PROYECTOR}${R}
       ${D}ayuda a la profundidad de las OAK-D Pro; apagalo si molesta a otra camara${R}
  6) Sin fotograma HD para las manos ............... $(si_no $SIN_HD)
       ${D}si el USB no da ancho de banda suficiente${R}
  7) Manos sobre la imagen entera (sin recorte) .... $(si_no $SIN_RECORTE)
       ${D}como antes; el recorte suele ser mas preciso${R}
  8) La camara B no sigue las manos ................ $(si_no $SIN_MANOS_B)
       ${D}solo con dos camaras: ahorra CPU${R}
  9) Sin estereo fino .............................. $(si_no $SIN_ESTEREO_FINO)
       ${D}profundidad menos precisa pero mas ligera${R}
 10) Sin extrinsecas (ignorar camaras_extrinsecas.json) $(si_no $SIN_EXTRINSECAS)
       ${D}solo con dos camaras: se autocalibran moviendo los brazos${R}

  v) volver
EOF
        read -r -p "Opcion: " op
        case "$op" in
            1) SOLO_MEDIAPIPE=$((1 - SOLO_MEDIAPIPE)) ;;
            2) SIN_MANOS=$((1 - SIN_MANOS)) ;;
            3) SIN_GIRO=$((1 - SIN_GIRO)) ;;
            4) SIN_IMU=$((1 - SIN_IMU)) ;;
            5) read -r -p "Intensidad del proyector (0-1) [$PROYECTOR]: " x
               if [[ "${x:-}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then PROYECTOR="$x"
               elif [ -n "${x:-}" ]; then aviso "valor no valido"; sleep 1; fi ;;
            6) SIN_HD=$((1 - SIN_HD)) ;;
            7) SIN_RECORTE=$((1 - SIN_RECORTE)) ;;
            8) SIN_MANOS_B=$((1 - SIN_MANOS_B)) ;;
            9) SIN_ESTEREO_FINO=$((1 - SIN_ESTEREO_FINO)) ;;
            10) SIN_EXTRINSECAS=$((1 - SIN_EXTRINSECAS)) ;;
            v|V|"") return ;;
        esac
    done
}

elegir_camaras() {
    echo
    echo "  1) una OAK-D"
    echo "  2) dos OAK-D fusionadas"
    echo "  3) webcam normal (sin OAK-D)"
    echo "  4) reproducir una sesion grabada (sin camaras)"
    read -r -p "Camaras: " x
    case "$x" in
        1) CAMARAS="una" ;;
        2) CAMARAS="dos" ;;
        3) CAMARAS="webcam"
           read -r -p "Indice de la webcam [$WEBCAM_IDX]: " i
           [[ "${i:-}" =~ ^[0-9]+$ ]] && WEBCAM_IDX="$i" ;;
        4) CAMARAS="reproducir"
           read -r -p "Archivo .pkl [${SESION:-sesion.pkl}]: " s
           SESION="${s:-${SESION:-sesion.pkl}}"
           [ -f "$SESION" ] || { aviso "no existe $SESION"; sleep 1; }
           read -r -p "Velocidad (veces el tiempo real) [$VELOCIDAD]: " s
           [[ "${s:-}" =~ ^[0-9]+(\.[0-9]+)?$ ]] && VELOCIDAD="$s" ;;
    esac
}

elegir_grabacion() {
    if [ -n "$GRABAR" ]; then GRABAR=""; GRABAR_VIDEO=0; return; fi
    local def="sesion_$(date +%Y%m%d_%H%M).pkl"
    read -r -p "Archivo de la grabacion [$def]: " x
    GRABAR="${x:-$def}"
    read -r -p "Grabar tambien los fotogramas (JPEG, ocupa mucho)? (s/N): " x
    [[ "${x:-n}" =~ ^[sS] ]] && GRABAR_VIDEO=1 || GRABAR_VIDEO=0
}

construir_args() {
    ARGS=()
    case "$CAMARAS" in
        dos)        ARGS+=(--camara2) ;;
        webcam)     ARGS+=(--webcam "$WEBCAM_IDX") ;;
        reproducir) ARGS+=(--reproducir "$SESION" --velocidad "$VELOCIDAD") ;;
    esac
    [ "$ORCA" = 1 ] && ARGS+=(--orca)
    [ "$ESCENARIO" = 1 ] && ARGS+=(--escenario tuberias)
    [ "$MODELO" != "full" ] && ARGS+=(--modelo "$MODELO")
    if [ -n "$GRABAR" ] && [ "$CAMARAS" != "reproducir" ]; then
        ARGS+=(--grabar "$GRABAR")
        [ "$GRABAR_VIDEO" = 1 ] && ARGS+=(--grabar-video)
    fi
    [ "$SOLO_MEDIAPIPE" = 1 ] && ARGS+=(--solo-mediapipe)
    [ "$SIN_MANOS" = 1 ] && ARGS+=(--sin-manos)
    [ "$SIN_GIRO" = 1 ] && ARGS+=(--sin-giro)
    [ "$SIN_IMU" = 1 ] && ARGS+=(--sin-imu)
    [ "$PROYECTOR" != "0.7" ] && ARGS+=(--proyector "$PROYECTOR")
    [ "$SIN_HD" = 1 ] && ARGS+=(--sin-hd)
    [ "$SIN_RECORTE" = 1 ] && ARGS+=(--sin-recorte)
    if [ "$CAMARAS" = "dos" ]; then
        [ "$SIN_MANOS_B" = 1 ] && ARGS+=(--sin-manos-b)
        [ "$SIN_EXTRINSECAS" = 1 ] && ARGS+=(--sin-extrinsecas)
    fi
    [ "$SIN_ESTEREO_FINO" = 1 ] && ARGS+=(--sin-estereo-fino)
}

while true; do
    mostrar_menu
    read -r -p "Opcion: " op
    case "$op" in
        1) if [ "$HAY_KUKA" = 0 ]; then
               aviso "Este proyecto no tiene montar_kuka.py: solo hay columna original"; sleep 2
           else
               [ "$SOPORTE" = "kuka" ] && SOPORTE="columna" || SOPORTE="kuka"
           fi ;;
        2) ORCA=$((1 - ORCA)) ;;
        3) ESCENARIO=$((1 - ESCENARIO)) ;;
        4) elegir_camaras ;;
        5) case "$MODELO" in lite) MODELO="full" ;; full) MODELO="heavy" ;; *) MODELO="lite" ;; esac ;;
        6) elegir_grabacion ;;
        7) menu_avanzado ;;
        q|Q) echo "Hasta luego"; exit 0 ;;
        a|A)
            if [ "$CAMARAS" = "reproducir" ] && [ ! -f "$SESION" ]; then
                aviso "No existe la sesion '$SESION': elige otra en la opcion 4"; sleep 2; continue
            fi
            construir_args
            echo
            echo "${N}Resumen${R}"
            echo "  Soporte: ${V}${SOPORTE}${R} | Manos: $([ "$ORCA" = 1 ] && echo OrcaHand || echo pinzas) | Camaras: $(texto_camaras)"
            echo "  Comando: ${C}TRON2_SOPORTE=$SOPORTE python teleop_tron2.py ${ARGS[*]}${R}"
            echo
            read -r -p "Arrancar? (S/n): " ok
            [[ "${ok:-s}" =~ ^[nN] ]] && continue
            break ;;
    esac
done

# =============================================================================
# 3) ARRANQUE
# =============================================================================
if [ "$SOPORTE" = "kuka" ] && [ ! -f "kuka_kr210_support/meshes/kr210l150/visual_mujoco/colores.json" ]; then
    # la primera vez montar_kuka.py convierte las mallas .dae del KUKA (necesita trimesh)
    if ! "$PY" -c "import trimesh, collada" >/dev/null 2>&1; then
        info "Primera vez con el KUKA: instalo trimesh y pycollada para convertir sus mallas"
        "$PY" -m pip install trimesh pycollada || { error "No se ha podido instalar trimesh"; exit 1; }
    fi
fi

export TRON2_SOPORTE="$SOPORTE"
info "Arrancando teleop_tron2.py ..."
exec "$PY" teleop_tron2.py "${ARGS[@]}"