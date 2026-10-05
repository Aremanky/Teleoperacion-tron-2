"""
Metricas de IMITACION: cuanto se parece la postura del robot a la del operador.

Por fotograma y brazo se compara, en el marco del robot:
  - angulo entre la direccion del BRAZO del operador y la del robot (hombro -> codo)
  - angulo entre la direccion del ANTEBRAZO del operador y la del robot (codo -> muneca)
  - flexion del codo del operador frente a la del robot
  - angulo entre el marco de la PALMA del operador y el del robot (orientacion completa)
  - error de posicion de la IK (codo y muneca, m) y retraso captura -> robot (s)
Se usan desde el teleop (clase Metricas, actualizada en cada ciclo) y desde la linea
de comandos sobre una sesion grabada, sin visor y a toda velocidad:

  python metricas_imitacion.py sesion.pkl [--orca] [--csv salida.csv] [--velocidad 50]

Objetivo razonable con dos camaras calibradas: brazo y antebrazo < 6 grados (mediana),
palma < 12 grados (mediana), retraso < 0.2 s. Si el brazo sale bien y la palma mal, el
problema esta en las manos (recorte, triangulacion); si sale mal el brazo, en la
profundidad / fusion; si la IK tiene error grande, en robot_tron2 (ver prueba_ik.py).
"""
import argparse
import csv
import time

import numpy as np

from suavizado import log_rot


def _ang(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))


class Metricas:
    def __init__(self, ruta_csv=None):
        self.filas = []
        self.ruta_csv = ruta_csv
        self._f = open(ruta_csv, "w", newline="", encoding="utf-8") if ruta_csv else None
        self._w = csv.writer(self._f) if self._f else None
        if self._w:
            self._w.writerow(["t_s", "lado", "foto", "fuente", "fuente_mano", "brazo_deg", "antebrazo_deg",
                              "codo_humano_deg", "codo_robot_deg", "palma_deg", "err_codo_m", "err_muneca_m",
                              "retraso_s"])
        self.t0 = None

    def registrar(self, t, lado_r, d_b_h, d_a_h, R_h, brazo, errores, retraso, foto, fuente, fuente_mano):
        """d_b_h, d_a_h: direcciones del operador YA en el marco del robot (unitarias).
        R_h: marco de la palma del operador en el marco del robot, o None."""
        if self.t0 is None:
            self.t0 = t
        p_c, p_m = brazo.puntos()
        d_b_r = p_c - brazo.hombro
        d_a_r = p_m - p_c
        fila = dict(t=t - self.t0, lado=lado_r, foto=foto, fuente=fuente, fuente_mano=fuente_mano or "",
                    brazo=_ang(d_b_h, d_b_r), antebrazo=_ang(d_a_h, d_a_r),
                    codo_h=_ang(d_b_h, d_a_h), codo_r=_ang(d_b_r, d_a_r),
                    palma=(np.nan if R_h is None else float(np.degrees(np.linalg.norm(log_rot(np.asarray(R_h) @ brazo.orientacion().T))))),
                    ec=float(errores[0]) if errores is not None else np.nan,
                    em=float(errores[1]) if errores is not None else np.nan,
                    retraso=float(retraso))
        self.filas.append(fila)
        if self._w:
            self._w.writerow([f"{fila['t']:.3f}", lado_r, foto, fuente, fila["fuente_mano"], f"{fila['brazo']:.2f}",
                              f"{fila['antebrazo']:.2f}", f"{fila['codo_h']:.1f}", f"{fila['codo_r']:.1f}",
                              f"{fila['palma']:.2f}", f"{fila['ec']:.4f}", f"{fila['em']:.4f}", f"{fila['retraso']:.3f}"])

    def cerrar(self):
        if self._f:
            self._f.close()
            self._f = None

    def resumen(self):
        """Texto con medianas y percentiles 95 por brazo."""
        lineas = []
        for lado in ("L", "R"):
            f = [x for x in self.filas if x["lado"] == lado]
            if not f:
                lineas.append(f"Brazo {lado}: sin datos")
                continue

            def est(k, filtro=None):
                v = np.array([x[k] for x in f if filtro is None or filtro(x)], dtype=float)
                v = v[np.isfinite(v)]
                return (np.nan, np.nan) if v.size == 0 else (float(np.median(v)), float(np.percentile(v, 95)))

            b, a, p, ec, em, r = est("brazo"), est("antebrazo"), est("palma"), est("ec"), est("em"), est("retraso")
            dc = est("codo_h")
            lineas.append(f"Brazo {lado} ({len(f)} fotos): brazo {b[0]:.1f}/{b[1]:.1f} deg | antebrazo {a[0]:.1f}/{a[1]:.1f} deg | "
                          f"palma {p[0]:.1f}/{p[1]:.1f} deg | IK codo {ec[0] * 100:.1f} cm, muneca {em[0] * 100:.1f} cm | "
                          f"retraso {r[0] * 1000:.0f}/{r[1] * 1000:.0f} ms   (mediana/p95)")
            fuentes = {}
            for x in f:
                fuentes[x["fuente_mano"] or "sin mano"] = fuentes.get(x["fuente_mano"] or "sin mano", 0) + 1
            lineas.append("   manos: " + ", ".join(f"{k} {100 * v / len(f):.0f}%" for k, v in sorted(fuentes.items(), key=lambda kv: -kv[1])))
        return "\n".join(lineas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sesion", help="archivo .pkl grabado con teleop_tron2.py --grabar")
    ap.add_argument("--xml", default="tron2a/DACH_TRON2A/xml/robot_elecnor.xml")
    ap.add_argument("--orca", action="store_true", help="OrcaHand (como en la grabacion)")
    ap.add_argument("--csv", default=None, help="guardar las metricas por fotograma")
    ap.add_argument("--velocidad", type=float, default=50.0, help="veces mas rapido que el tiempo real")
    ap.add_argument("--sin-extrinsecas", action="store_true")
    args = ap.parse_args()

    import teleop_tron2
    t0 = time.monotonic()
    tele = teleop_tron2.preparar_reproduccion(args.sesion, args.xml, args.orca, velocidad=args.velocidad,
                                              metricas=Metricas(args.csv), sin_extrinsecas=args.sin_extrinsecas)
    tele.correr_sin_visor()
    print(f"\n({time.monotonic() - t0:.1f} s de calculo)")
    print(tele.metricas.resumen())
    tele.metricas.cerrar()


if __name__ == "__main__":
    main()
