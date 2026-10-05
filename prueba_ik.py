"""
Prueba de la IK del TRON 2 SIN camaras: un "operador sintetico" mueve los brazos y la
palma con trayectorias suaves y se mide lo bien que el robot los copia.

Separa los fallos de la IK (robot_tron2.Brazo) de los fallos de vision: si esta prueba
sale bien y la teleoperacion no, el problema esta en las camaras/fusion, no en el robot.

  python prueba_ik.py                      # TRON 2 con OrcaHand (como teleop --orca)
  python prueba_ik.py --sin-orca           # con las pinzas originales
  python prueba_ik.py --visor              # ademas lo muestra en el visor de MuJoCo
  python prueba_ik.py --duracion 20

Imprime, por brazo: error de codo y muneca (mediana / p95, en cm), error de orientacion de
la palma (grados), velocidad articular maxima y si hubo rescates. Criterios de "bien":
muneca < 1.5 cm (mediana), orientacion < 8 grados (mediana) con el objetivo alcanzable.
"""
import argparse
import time

import numpy as np

import escenario_tuberias
from robot_tron2 import RobotTron2, marco_semantico
from suavizado import ObjetivoSuave, RotacionSuave, comprimir_alcance, exp_rot, log_rot

FREC = 60.0


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def operador(t, lado):
    """Direcciones (brazo, antebrazo) y marco de la palma de un operador sintetico, en el
    marco del robot (x delante, y izquierda, z arriba). Movimientos amplios y suaves:
    levantar el brazo al frente, abrirlo hacia fuera, doblar el codo (hacia delante/arriba,
    como un codo humano), girar la palma (pronosupinacion) y doblar un poco la muneca."""
    s = 1.0 if lado == "L" else -1.0
    # brazo: elevacion 15..100 grados respecto a la vertical, azimut desde delante (0) hacia fuera
    elev = np.radians(57 + 43 * np.sin(0.35 * t))
    az = np.radians(35 + 35 * np.sin(0.21 * t + 1.0))
    d_b = unit([np.cos(az) * np.sin(elev), s * np.sin(az) * np.sin(elev), -np.cos(elev)])
    # codo: eje lateral (perpendicular al brazo y a "delante+arriba"); flexion 10..110 grados
    eje = unit(np.cross(d_b, [1.0, 0.0, 1.0]))
    flex = np.radians(60 + 50 * np.sin(0.5 * t + 0.7))
    d_a = exp_rot(eje * flex) @ d_b
    # muneca: dedos siguiendo el antebrazo con algo de flexion; normal neutra = palma hacia el
    # cuerpo (como cuelga una mano relajada: en la derecha la palma mira a +y; en la izquierda
    # el 3er eje del marco es el dorso, que tambien mira a +y), girada +-60 grados (pronosupinacion)
    flex_m = np.radians(20 * np.sin(0.6 * t + 2.0))
    f = exp_rot(eje * flex_m) @ d_a
    y = np.array([0.0, 1.0, 0.0])
    n0 = y - f * np.dot(y, f)
    if np.linalg.norm(n0) < 0.2:                 # antebrazo casi lateral: hacia atras
        n0 = np.array([-1.0, 0.0, 0.0]) - f * np.dot([-1.0, 0.0, 0.0], f)
    n0 = unit(n0)
    giro = np.radians(60 * np.sin(0.8 * t))
    n = exp_rot(f * giro) @ n0
    a = np.cross(n, f)                           # ancho tal que f x a = n
    return d_b, d_a, marco_semantico(f, a)


def correr(robot, duracion, visor=None, omega=14.0, omega_rot=12.0, w_codo=0.6):
    dt = 1.0 / FREC
    n = int(duracion * FREC)
    suaves = {l: dict(c=ObjetivoSuave(omega, 0.004), m=ObjetivoSuave(omega, 0.004), R=RotacionSuave(omega_rot))
              for l in ("L", "R")}
    reg = {l: dict(ec=[], em=[], eo=[], vel=[], q_ant=None) for l in ("L", "R")}
    rescates_antes = 0
    t_ini = time.monotonic()
    for k in range(n):
        t = k * dt
        for lado, brazo in robot.brazos.items():
            d_b, d_a, R_h = operador(t, lado)
            codo = brazo.hombro + brazo.L_brazo * d_b
            muneca = codo + brazo.L_antebrazo * d_a
            muneca = comprimir_alcance(brazo.hombro, muneca, brazo.L_brazo + brazo.L_antebrazo)
            s = suaves[lado]
            c_s, m_s = s["c"].actualizar(codo, dt), s["m"].actualizar(muneca, dt)
            s["R"].fijar_objetivo(R_h)
            R_s = s["R"].actualizar(dt)
            ec, em, eo = brazo.seguir(c_s, m_s, dt, w_codo, R_obj=R_s)
            q = robot.d.qpos[brazo.qadr].copy()
            if reg[lado]["q_ant"] is not None:
                reg[lado]["vel"].append(float(np.abs(q - reg[lado]["q_ant"]).max() / dt))
            reg[lado]["q_ant"] = q
            if k > FREC:   # el primer segundo es el arranque desde el reposo
                reg[lado]["ec"].append(ec)
                reg[lado]["em"].append(em)
                reg[lado]["eo"].append(eo)
        robot.actualizar()
        if visor is not None:
            visor.sync()
            resto = (t_ini + (k + 1) * dt) - time.monotonic()
            if resto > 0:
                time.sleep(resto)
    return reg


def informe(reg):
    ok = True
    for lado, r in reg.items():
        ec, em, eo, vel = (np.array(r[k]) for k in ("ec", "em", "eo", "vel"))
        print(f"Brazo {lado}: codo {np.median(ec) * 100:.1f} / {np.percentile(ec, 95) * 100:.1f} cm | "
              f"muneca {np.median(em) * 100:.1f} / {np.percentile(em, 95) * 100:.1f} cm | "
              f"palma {np.degrees(np.median(eo)):.1f} / {np.degrees(np.percentile(eo, 95)):.1f} deg | "
              f"vel. articular max {vel.max():.2f} rad/s (mediana / p95)")
        ok &= np.median(em) < 0.015 and np.degrees(np.median(eo)) < 8.0 and np.all(np.isfinite(em))
    print("RESULTADO:", "BIEN" if ok else "REVISAR (ver criterios en la cabecera)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default="tron2a/DACH_TRON2A/xml/robot_elecnor.xml")
    ap.add_argument("--sin-orca", action="store_true")
    ap.add_argument("--visor", action="store_true")
    ap.add_argument("--duracion", type=float, default=15.0)
    args = ap.parse_args()
    robot = RobotTron2(args.xml, escenario_tuberias.anadir_al_modelo, orca=not args.sin_orca)
    robot.resumen()
    if args.visor:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(robot.m, robot.d, show_left_ui=False, show_right_ui=False) as v:
            reg = correr(robot, args.duracion, visor=v)
    else:
        t0 = time.monotonic()
        reg = correr(robot, args.duracion)
        print(f"({args.duracion:.0f} s simulados en {time.monotonic() - t0:.1f} s de calculo)")
    informe(reg)


if __name__ == "__main__":
    main()
