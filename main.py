"""
Punto de entrada principal del proyecto Filtro de Kalman para tráfico aéreo.

Ejecuta el proceso completo de simulación y genera las visualizaciones.
"""

import src.graficas as grf


def main():
    """Ejecuta la simulación y genera todas las gráficas de resultados."""
    grf.ejecutar()

    # --- GENERAR MEDIDAS RADAR CON RUIDO ---
    medidas_radar = generar_radar(medidas_reales, sigma_rho, sigma_theta)

if __name__ == "__main__":
    main()