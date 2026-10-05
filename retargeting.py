"""
Retargeting de la mano: landmarks 3D → ctrl[] de la OrcaHand (17 valores).

Origen: Skill_Shield_Nivel_6.py del proyecto OrcaHand/ARCTOS. Cambios para el TRON 2:
  - importa de config_mano.py (no de config.py);
  - IMITACION_PURA (estado["pinza_habilitada"] = False): los dedos copian a los tuyos y
    NO se sustituyen por las posturas de pinza predefinidas (PINZA_CTRL_CONTACTO);
  - el suavizado de dedos es un One Euro (suavizado.FiltroDedos) en vez de una media movil;
  - fuera la comprobacion de contacto yema-pulgar por nombres de cuerpo (con los prefijos
    orcaL_/orcaR_ nunca encontraba nada) y el print de depuracion por datos.time.
"""
import numpy as np

from config_mano import (FUERZA_ABD, FUERZA_NUDILLOS, FUERZA_PUNTAS, GANANCIA_DEDOS, LIMITES_CTRL, N_ACT,
                         OFFSET_MANO, PINZA_DIST_MAX, PINZA_UMBRAL_OFF, PINZA_UMBRAL_ON, PULGAR_GANANCIA_IP,
                         PULGAR_GANANCIA_MCP, PUNO_COLECTIVO, UMBRAL_BLOQUEO, UMBRAL_EXT_PULGAR_IP,
                         UMBRAL_EXT_PULGAR_MCP, UMBRAL_LIBERACION, UMBRAL_PUNO)
from config_mano import (UMBRAL_COMBO2_OFF as _UMBRAL_COMBO2_OFF, UMBRAL_COMBO2_ON as _UMBRAL_COMBO2_ON,
                         UMBRAL_COMBO3_OFF as _UMBRAL_COMBO3_OFF, UMBRAL_COMBO3_ON as _UMBRAL_COMBO3_ON,
                         UMBRAL_COMBO_OFF as _UMBRAL_COMBO_OFF, UMBRAL_COMBO_ON as _UMBRAL_COMBO_ON)
from biomecanica import calcular_abertura, calcular_distancia, calcular_flexion, calcular_flexion_muneca


# ══════════════════════════════════════════════════════════════════════════════
# RETARGETING: LANDMARKS 3D → ctrl[]
# ══════════════════════════════════════════════════════════════════════════════

def retarget_a_ctrl(modelo, datos, lm3d: list, hand_world_lm, estado: dict, suav_dedos) -> dict:
    """
    Calcula y escribe datos.ctrl[0..N_ACT-1] a partir de landmarks 3D.

    Flujo:
      1. Aberturas laterales (abducción entre dedos)
      2. Detección de pinza (distancia pulgar→cada dedo)
      3. Control de muñeca con latch de seguridad
      4. Flexión de cada dedo (MCP + PIP/DIP)
      5. Pose del pulgar (oposición + abducción)
      6. Override de pinza (cierre dirigido a un dedo concreto)
      7. Suavizado de dedos (media móvil)
      8. Clip de seguridad anatómica

    Devuelve el estado actualizado.
    """
    # ctrl_mano: vista sobre la porción de mano de datos.ctrl[], desplazada
    # OFFSET_MANO posiciones porque el brazo ocupa ahora los primeros índices.
    # Todo lo de abajo sigue usando los índices 0..16 de siempre.
    ctrl_mano = datos.ctrl[OFFSET_MANO: OFFSET_MANO + N_ACT]

    # — 1. Aberturas (Con umbrales calibrados y Convergencia Activa) —
    
    # 1.1 Vectores largos (MCP -> DIP)
    ang_indice  = calcular_abertura(lm3d[9],  lm3d[11], lm3d[5],  lm3d[7])
    ang_anular  = calcular_abertura(lm3d[9],  lm3d[11], lm3d[13], lm3d[15])
    ang_menique = calcular_abertura(lm3d[13], lm3d[15], lm3d[17], lm3d[19])

    # 1.2 Distancias físicas entre las articulaciones medias (PIP)
    dist_pip_indice_corazon = calcular_distancia(lm3d[6], lm3d[10])
    dist_pip_corazon_anular = calcular_distancia(lm3d[10], lm3d[14])
    dist_pip_anular_menique = calcular_distancia(lm3d[14], lm3d[18])

    # 1.3 Zonas Muertas INDIVIDUALES (en metros)
    UMBRAL_INDICE_CORAZON = 0.028  
    UMBRAL_CORAZON_ANULAR = 0.022  
    UMBRAL_ANULAR_MENIQUE = 0.020  

    # 1.4 Ángulos de Convergencia (Radianes)
    # Valores que fuerzan a los dedos a inclinarse hacia el corazón (dedo ancla).
    # Ajusta estos valores si la colisión en MuJoCo es demasiado fuerte o se queda corta.
    CONV_INDICE  = -0.18  # Fuerza al índice a inclinarse hacia el centro
    CONV_ANULAR  = -0.15  # Fuerza al anular a inclinarse hacia el centro
    CONV_MENIQUE = -0.18  # Fuerza al meñique a pegarse al anular

    # 1.5 Aplicar la compuerta lógica con convergencia
    sep_indice  = CONV_INDICE  if dist_pip_indice_corazon < UMBRAL_INDICE_CORAZON else ang_indice
    sep_anular  = CONV_ANULAR  if dist_pip_corazon_anular < UMBRAL_CORAZON_ANULAR else ang_anular
    sep_menique = CONV_MENIQUE if dist_pip_anular_menique < UMBRAL_ANULAR_MENIQUE else ang_menique

    # — 2. Detección de pinza con histéresis y ratio continuo —
    dist_pulgar = {
        "indice":  calcular_distancia(lm3d[4], lm3d[8]),
        "corazon": calcular_distancia(lm3d[4], lm3d[12]),
        "anular":  calcular_distancia(lm3d[4], lm3d[16]),
        "menique": calcular_distancia(lm3d[4], lm3d[20]),
    }

    # — Detección de combos multi-dedo ────────────────────────────────────────
    # Se evalúan ANTES que la búsqueda de dedo individual para tener prioridad.
    # Prioridad: indice_corazon > corazon_anular > anular_menique
    # Cada combo se activa cuando AMBOS dedos entran en su umbral propio,
    # y se libera cuando CUALQUIERA de los dos supera su umbral de salida.
    dedo_bloqueado_actual = estado.get("dedo_bloqueado")

    # — Combo índice + corazón —
    combo2_activa = False
    if not estado["modo_pinza"] or dedo_bloqueado_actual == "indice_corazon":
        if not estado["modo_pinza"]:
            combo2_activa = (
                dist_pulgar["indice"]  < _UMBRAL_COMBO2_ON["indice"] and
                dist_pulgar["corazon"] < _UMBRAL_COMBO2_ON["corazon"]
            )
        else:
            combo2_activa = not (
                dist_pulgar["indice"]  > _UMBRAL_COMBO2_OFF["indice"] or
                dist_pulgar["corazon"] > _UMBRAL_COMBO2_OFF["corazon"]
            )

    # — Combo corazón + anular —
    combo1_activa = False
    if not combo2_activa and (not estado["modo_pinza"] or
                               dedo_bloqueado_actual == "corazon_anular"):
        if not estado["modo_pinza"]:
            combo1_activa = (
                dist_pulgar["corazon"] < _UMBRAL_COMBO_ON["corazon"] and
                dist_pulgar["anular"]  < _UMBRAL_COMBO_ON["anular"]
            )
        else:
            combo1_activa = not (
                dist_pulgar["corazon"] > _UMBRAL_COMBO_OFF["corazon"] or
                dist_pulgar["anular"]  > _UMBRAL_COMBO_OFF["anular"]
            )

    # — Combo anular + meñique —
    combo3_activa = False
    if not combo2_activa and not combo1_activa and (not estado["modo_pinza"] or
                                                     dedo_bloqueado_actual == "anular_menique"):
        if not estado["modo_pinza"]:
            combo3_activa = (
                dist_pulgar["anular"]  < _UMBRAL_COMBO3_ON["anular"] and
                dist_pulgar["menique"] < _UMBRAL_COMBO3_ON["menique"]
            )
        else:
            combo3_activa = not (
                dist_pulgar["anular"]  > _UMBRAL_COMBO3_OFF["anular"] or
                dist_pulgar["menique"] > _UMBRAL_COMBO3_OFF["menique"]
            )

    combo_activa = combo2_activa or combo1_activa or combo3_activa

    # — Detección de dedo individual (histéresis) ─────────────────────────────
    dedo_candidato = min(dist_pulgar, key=dist_pulgar.get)
    dist_min       = dist_pulgar[dedo_candidato]

    if combo2_activa:
        modo_pinza = True
        estado["dedo_bloqueado"] = "indice_corazon"
    elif combo1_activa:
        modo_pinza = True
        estado["dedo_bloqueado"] = "corazon_anular"
    elif combo3_activa:
        modo_pinza = True
        estado["dedo_bloqueado"] = "anular_menique"
    elif not estado["modo_pinza"]:
        if dist_min < PINZA_UMBRAL_ON[dedo_candidato]:
            modo_pinza               = True
            estado["dedo_bloqueado"] = dedo_candidato
        else:
            modo_pinza = False
    else:
        if dedo_bloqueado_actual in ("corazon_anular", "indice_corazon", "anular_menique"):
            modo_pinza               = False
            estado["dedo_bloqueado"] = None
        else:
            dist_dedo_bloqueado = dist_pulgar[dedo_bloqueado_actual]
            if dist_dedo_bloqueado > PINZA_UMBRAL_OFF[dedo_bloqueado_actual]:
                modo_pinza               = False
                estado["dedo_bloqueado"] = None
            else:
                modo_pinza = True

# Si las pinzas están deshabilitadas por teclado, forzamos la desactivación.
    if not estado.get("pinza_habilitada", True):
        modo_pinza = False
        estado["dedo_bloqueado"] = None

    dedo_activo = estado["dedo_bloqueado"] if modo_pinza else dedo_candidato

    # Ratio continuo de cierre — para combos se usa el promedio de distancias
    if dedo_activo == "corazon_anular":
        dist_ref         = (dist_pulgar["corazon"] + dist_pulgar["anular"]) / 2.0
        umbral_on_activo = (_UMBRAL_COMBO_ON["corazon"] + _UMBRAL_COMBO_ON["anular"]) / 2.0
    elif dedo_activo == "indice_corazon":
        dist_ref         = (dist_pulgar["indice"] + dist_pulgar["corazon"]) / 2.0
        umbral_on_activo = (_UMBRAL_COMBO2_ON["indice"] + _UMBRAL_COMBO2_ON["corazon"]) / 2.0
    elif dedo_activo == "anular_menique":
        dist_ref         = (dist_pulgar["anular"] + dist_pulgar["menique"]) / 2.0
        umbral_on_activo = (_UMBRAL_COMBO3_ON["anular"] + _UMBRAL_COMBO3_ON["menique"]) / 2.0
    else:
        dist_ref         = dist_pulgar[dedo_activo]
        umbral_on_activo = PINZA_UMBRAL_ON[dedo_activo]
    ratio_pinza = float(np.clip(
        1.0 - (dist_ref - umbral_on_activo) / (PINZA_DIST_MAX - umbral_on_activo),
        0.0, 1.0
    ))

# — 3. Muñeca (Estabilizada) —
    calc_muneca = 0.0
    if estado["vec_neutro"] is not None:
        flexion_raw = calcular_flexion_muneca(
            hand_world_lm, estado["vec_neutro"], estado["eje_lateral"])
        
        ZONA_MUERTA = 0.25  # Aumentado para absorber el ruido Z de MediaPipe
        GANANCIA    = 1.40  # Reducido (antes 2.2) para evitar amplificar temblores
        
        if abs(flexion_raw) < ZONA_MUERTA:
            calc_muneca = 0.0
        else:
            calc_muneca = np.sign(flexion_raw) * (abs(flexion_raw) - ZONA_MUERTA) * GANANCIA
            
        calc_muneca = np.clip(calc_muneca, -1.134, 0.610)

    if modo_pinza:
        if not estado["pinza_previa"]:
            estado["ultima_pos_palma"] = ctrl_mano[0]
        if abs(calc_muneca - estado["ultima_pos_palma"]) > 0.45:
            estado["ultima_pos_palma"] += (
                calc_muneca - estado["ultima_pos_palma"]) * 0.15  # Más suave en pinza
        ctrl_mano[0] = float(estado["ultima_pos_palma"])
    else:
        if estado["muneca_bloqueada"]:
            ctrl_mano[0] = 0.610
            if calc_muneca < UMBRAL_LIBERACION:
                estado["muneca_bloqueada"] = False
        else:
            # Filtro de suavizado reforzado: 96% valor anterior, 4% nuevo valor
            ctrl_mano[0] = float(np.clip(
                ctrl_mano[0] * 0.96 + calc_muneca * 0.04, -1.134, 0.610))
            if calc_muneca >= UMBRAL_BLOQUEO:
                estado["muneca_bloqueada"] = True

    # — 4 + 5. Dedos y pulgar (valores brutos, sin suavizar aún) —
    ctrl_raw = np.array([ctrl_mano[i] for i in range(N_ACT)], dtype=float)

    # Meñique  — lm: 17=MCP  18=PIP  19=DIP  20=punta
    ctrl_raw[1] = -sep_menique * FUERZA_ABD + 0.15
    ctrl_raw[2] = calcular_flexion(lm3d[0],  lm3d[17], lm3d[18]) * FUERZA_NUDILLOS * GANANCIA_DEDOS["menique_mcp"]
    ctrl_raw[3] = calcular_flexion(lm3d[17], lm3d[18], lm3d[19]) * FUERZA_PUNTAS   * GANANCIA_DEDOS["menique_pip"]  # 19=DIP, no 20=punta
    # Anular  — lm: 13=MCP  14=PIP  15=DIP  16=punta
    ctrl_raw[4] = -sep_anular * FUERZA_ABD
    ctrl_raw[5] = calcular_flexion(lm3d[0],  lm3d[13], lm3d[14]) * FUERZA_NUDILLOS * GANANCIA_DEDOS["anular_mcp"]
    ctrl_raw[6] = calcular_flexion(lm3d[13], lm3d[14], lm3d[15]) * FUERZA_PUNTAS   * GANANCIA_DEDOS["anular_pip"]   # 15=DIP, no 16=punta
    # Corazón  — lm: 9=MCP  10=PIP  11=DIP  12=punta  (abducción fija)
    ctrl_raw[7] = 0.0
    ctrl_raw[8] = calcular_flexion(lm3d[0],  lm3d[9],  lm3d[10]) * FUERZA_NUDILLOS * GANANCIA_DEDOS["corazon_mcp"]
    ctrl_raw[9] = calcular_flexion(lm3d[9],  lm3d[10], lm3d[11]) * FUERZA_PUNTAS   * GANANCIA_DEDOS["corazon_pip"]  # 11=DIP, no 12=punta
    # Índice  — lm: 5=MCP  6=PIP  7=DIP  8=punta
    ctrl_raw[10] = sep_indice * FUERZA_ABD
    ctrl_raw[11] = calcular_flexion(lm3d[0],  lm3d[5],  lm3d[6])  * FUERZA_NUDILLOS * GANANCIA_DEDOS["indice_mcp"]
    ctrl_raw[12] = calcular_flexion(lm3d[5],  lm3d[6],  lm3d[7])  * FUERZA_PUNTAS   * GANANCIA_DEDOS["indice_pip"]   # 7=DIP, no 8=punta

    # ── Seguimiento colectivo en zona de puño ─────────────────────────────────
    # Cuando el índice cierra, arrastra al resto progresivamente.
    # Compensa la oclusión de MediaPipe: al cerrar el puño las yemas del anular
    # y meñique desaparecen de la vista y MediaPipe los pierde.
    # El suavizador se reduce a ventana=1 (sin dilución) durante el puño.
    en_puno = PUNO_COLECTIVO and (ctrl_raw[11] > UMBRAL_PUNO or ctrl_raw[5] > UMBRAL_PUNO)
    if en_puno:
        suav_dedos.set_ventana(1)   # sin suavizado: el cierre llega al máximo
        factor = max(
            (ctrl_raw[11] - UMBRAL_PUNO) / (1.80 - UMBRAL_PUNO),
            (ctrl_raw[5]  - UMBRAL_PUNO) / (1.80 - UMBRAL_PUNO),
        )
        factor = float(np.clip(factor, 0.0, 1.0))
        # MCP de cada dedo: arrastra con ganancia anatómica
        for idx_ctrl, ref_ctrl, g in [
            (2,  11, 1.05),   # meñique mcp
            (5,  11, 1.08),   # anular  mcp  ← ya tiene señal directa, refuerza
            (8,  11, 1.03),   # corazón mcp
        ]:
            ctrl_raw[idx_ctrl] = max(ctrl_raw[idx_ctrl],
                                     ctrl_raw[ref_ctrl] * g * factor)
        # PIP de cada dedo
        for idx_ctrl, ref_ctrl, g in [
            (3,  12, 1.10),   # meñique pip
            (6,  12, 1.12),   # anular  pip
            (9,  12, 1.05),   # corazón pip
        ]:
            ctrl_raw[idx_ctrl] = max(ctrl_raw[idx_ctrl],
                                     ctrl_raw[ref_ctrl] * g * factor)
    else:
        suav_dedos.set_ventana(4)   # suavizado normal fuera del puño

    # ── Pulgar — Oposición (CMC) y Abducción (ABD) por Proyección Transversal ──
    def _lm_w(i):
        l = hand_world_lm.landmark[i]
        return np.array([l.x, l.y, l.z])

    ix_mcp = _lm_w(5)   # Nudillo Índice
    pi_mcp = _lm_w(17)  # Nudillo Meñique
    th_tip = _lm_w(4)   # Punta del pulgar (la parte que realmente viaja)

    # 1. Creamos un "carril" que va desde el índice hasta el meñique
    v_palma = pi_mcp - ix_mcp
    ancho_palma = np.linalg.norm(v_palma)

    if ancho_palma > 1e-6:
        dir_palma = v_palma / ancho_palma
        
        # 2. Vector desde el índice hasta donde está el pulgar actualmente
        v_pulgar = th_tip - ix_mcp
        
        # 3. Proyectamos el pulgar sobre el carril de la palma.
        # ratio_cruce = 0.0 -> pulgar en reposo (alineado con el índice)
        # ratio_cruce = 1.0 -> pulgar tocando la base del meñique
        ratio_cruce = np.dot(v_pulgar, dir_palma) / ancho_palma
        
        # 4. Mapeo Directo a los motores (Basado en los valores de tu captura)
        
        # CMC (Oposición): En reposo = -0.1. Cruzado = 0.576.
        # Usamos un multiplicador agresivo (1.3) para garantizar que alcance el 0.576
        calc_cmc = (ratio_cruce * 1.3) - 0.10
        ctrl_raw[13] = float(np.clip(calc_cmc, LIMITES_CTRL[13][0], LIMITES_CTRL[13][1]))
        
        # ABD (Abducción): En reposo = ~0.65. Cruzado = -0.314.
        # Relación inversa: Cuanto más cruzas, más negativo se vuelve para aplastarse contra la palma
        calc_abd = 0.65 - (ratio_cruce * 1.3)
        ctrl_raw[14] = float(np.clip(calc_abd, LIMITES_CTRL[14][0], LIMITES_CTRL[14][1]))
    else:
        ctrl_raw[13], ctrl_raw[14] = 0.0, 0.30

    # ── Lógica Desacoplada para Flexión/Extensión del Pulgar ──
    flex_raw_mcp = calcular_flexion(lm3d[1], lm3d[2], lm3d[3])
    flex_raw_ip  = calcular_flexion(lm3d[2], lm3d[3], lm3d[4])

    # Nudillo del Pulgar (MCP - Joint 15)
    if flex_raw_mcp < UMBRAL_EXT_PULGAR_MCP:
        # Zona de Extensión: Aplanamos el valor hacia 0 para abrir el dedo por completo
        ctrl_raw[15] = (flex_raw_mcp / UMBRAL_EXT_PULGAR_MCP) * 0.05
    else:
        # Zona de Flexión: Aplicamos ganancia para asegurar un buen agarre
        ctrl_raw[15] = 0.05 + (flex_raw_mcp - UMBRAL_EXT_PULGAR_MCP) * PULGAR_GANANCIA_MCP

    # Punta del Pulgar (IP - Joint 16)
    if flex_raw_ip < UMBRAL_EXT_PULGAR_IP:
        # Zona de Extensión: Filtra el "temblor" de la punta cuando la mano está abierta
        ctrl_raw[16] = (flex_raw_ip / UMBRAL_EXT_PULGAR_IP) * 0.05
    else:
        # Zona de Flexión: Ganancia agresiva para que la yema baje correctamente
        ctrl_raw[16] = 0.05 + (flex_raw_ip - UMBRAL_EXT_PULGAR_IP) * PULGAR_GANANCIA_IP

    # — 6. Suavizado de dedos (ctrl[1..16]) — solo para movimiento libre —————
    # Los joints de pinza se sobreescriben después del filtro (ver paso 7).
    # Al entrar en pinza se vacía el buffer para no mezclar valores anteriores.
    if modo_pinza and not estado["pinza_previa"]:
        suav_dedos.reset()
    ctrl_raw[1:] = suav_dedos.actualizar(ctrl_raw[1:])

    # — 7. Clip de seguridad anatómica —
    for i, (lo, hi) in enumerate(LIMITES_CTRL):
        ctrl_mano[i] = float(np.clip(ctrl_raw[i], lo, hi))

    # — 8. Override de pinza POST-suavizado — escribe directo a datos.ctrl ————
    #
    # CRÍTICO: se aplica DESPUÉS del suavizador y del clip para que los valores
    # lleguen a MuJoCo exactamente como se calibraron en el visor, sin dilución
    # por el filtro de media móvil.
    #
    # Valores calibrados manualmente en el visor MuJoCo para cada pinza:
    #   Se leen directamente del panel "Control" con la pose de contacto activa.
    #   Para calibrar una nueva pinza: alcanza la pose en el visor → copia los
    #   valores del panel aquí → reinicia el script.
    #
    # Mapeo ctrl[] → joint del panel:
    #   ctrl[1]=p-abd  ctrl[2]=p-mcp  ctrl[3]=p-pip   (meñique)
    #   ctrl[4]=r-abd  ctrl[5]=r-mcp  ctrl[6]=r-pip   (anular)
    #   ctrl[7]=m-abd  ctrl[8]=m-mcp  ctrl[9]=m-pip   (corazón)
    #   ctrl[10]=i-abd ctrl[11]=i-mcp ctrl[12]=i-pip  (índice)
    #   ctrl[13]=t-cmc ctrl[14]=t-abd ctrl[15]=t-mcp ctrl[16]=t-pip (pulgar)
    PINZA_CTRL_CONTACTO = {
        # ── Pulgar ↔ Índice ───────────────────────────────────────────────────
        "indice": {
            10: 0.000,   # índice ABD
            11: 0.491,   # índice MCP
            12: 1.250,   # índice PIP
            13: 0.000,   # pulgar CMC
            14: 0.469,   # pulgar ABD
            15: 0.469,   # pulgar MCP
            16: 0.888,   # pulgar IP
        },

        # ── Pulgar ↔ Corazón ──────────────────────────────────────────────────
        "corazon": {
            8:  0.916,   # corazón MCP
            9:  1.540,   # corazón PIP
            13: 0.426,   # pulgar CMC
            14: 0.113,   # pulgar ABD
            15: 0.174,   # pulgar MCP
            16: 1.720,   # pulgar IP
        },

        # ── Pulgar ↔ Anular ───────────────────────────────────────────────────
        "anular": {
            4: -0.0565,  # anular ABD
            5:  1.230,   # anular MCP
            6:  0.675,   # anular PIP
            13: 0.000,   # pulgar CMC
            14: -0.282,  # pulgar ABD 
            15: 0.905,   # pulgar MCP
            16: 0.526,   # pulgar IP
        },

        # ── Pulgar ↔ Meñique ──────────────────────────────────────────────────
        "menique": {
            1:  0.0995,  # meñique ABD
            2:  1.130,   # meñique MCP
            3:  0.366,   # meñique PIP
            13: 0.576,   # pulgar CMC
            14: -0.301,  # pulgar ABD
            15: 0.207,   # pulgar MCP
            16: 0.888,   # pulgar IP
        },

        # ── Pulgar ↔ Corazón + Anular (perro / sombra) ────────────────────────
        "corazon_anular": {
            4:  0.066,   # anular ABD
            5:  1.040,   # anular MCP
            6:  0.867,   # anular PIP
            7:  0.000,   # corazón ABD (fijo)
            8:  1.130,   # corazón MCP
            9:  0.888,   # corazón PIP
            13: 0.406,   # pulgar CMC
            14: 0.0171,  # pulgar ABD
            15: 0.229,   # pulgar MCP
            16: 0.217,   # pulgar IP
        },

        # ── Pulgar ↔ Índice + Corazón ─────────────────────────────────────────
        "indice_corazon": {
            7:  0.0565,  # corazón ABD
            8:  0.916,   # corazón MCP
            9:  0.835,   # corazón PIP
            10: -0.0571, # índice ABD
            11: 0.840,   # índice MCP
            12: 0.643,   # índice PIP
            13: 0.419,   # pulgar CMC
            14: 0.291,   # pulgar ABD
            15: 0.000,   # pulgar MCP
            16: 0.526,   # pulgar IP
        },

        # ── Pulgar ↔ Anular + Meñique ─────────────────────────────────────────
        "anular_menique": {
            1:  0.194,   # meñique ABD
            2:  0.982,   # meñique MCP
            3:  0.739,   # meñique PIP
            4: -0.0377,  # anular ABD
            5:  0.916,   # anular MCP
            6:  0.984,   # anular PIP
            13: 0.236,   # pulgar CMC
            14: -0.212,  # pulgar ABD
            15: 0.982,   # pulgar MCP
            16: 0.292,   # pulgar IP
        },
    }

    if modo_pinza:
        for idx, val in PINZA_CTRL_CONTACTO[dedo_activo].items():
            lo, hi = LIMITES_CTRL[idx]
            ctrl_mano[idx] = float(np.clip(val, lo, hi))
    estado["pinza_tocando"] = False   # (sin fisica no hay contactos que comprobar)

    estado["pinza_previa"]  = modo_pinza
    estado["calc_muneca"]   = calc_muneca
    estado["dedo_activo"]   = dedo_activo
    estado["modo_pinza"]    = modo_pinza
    estado["dist_pulgar"]   = dist_pulgar      # para el diagnóstico de pinzas
    return estado
