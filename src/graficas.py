"""
Módulo de visualización de resultados del filtro de Kalman.

Genera gráficas para analizar:
- Ubicación de nodos y radar
- Trayectorias reales vs estimadas
- Comparación de errores CV vs CA
- Medidas de radar con ruido
- Evolución de velocidades estimadas
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import src.config as cnfg
import src.objetivos as obj
import src.generar_datos as gdat

# Configuración global de seaborn para consistencia
sns.set_theme(style="whitegrid", palette="husl")
#: Paleta de colores consistente para todas las gráficas.
PALETTE = sns.color_palette("husl", 8)


def guardar(fig, nombre):
    """
    Guarda una figura en los directorios 'resultados/png' y 'resultados/svg'.

    Genera archivos en ambos formatos para uso en visualización (PNG)
    y para inclusion en informes (SVG).

    Args:
        fig: Figura de matplotlib.
        nombre: Nombre del archivo (sin ruta, con extension .png).
    """
    os.makedirs("resultados/png", exist_ok=True)
    os.makedirs("resultados/svg", exist_ok=True)

    # Guardar en PNG (para visualización rápida)
    fig.savefig(f"resultados/png/{nombre}", dpi=150, bbox_inches="tight")

    # Guardar en SVG (para inclusion en informes)
    nombre_svg = nombre.replace('.png', '.svg')
    fig.savefig(f"resultados/svg/{nombre_svg}", format='svg', bbox_inches="tight")

    plt.close(fig)


def plot_nodos():
    """
    Genera gráfica con la ubicación del radar y los 3 waypoints.

    Sigue la simbología de cartas de navegación aérea:
    - Fix points (waypoints): triángulos negros
    - Radar: hexágonos

    Guarda la gráfica como 'resultados/nodos.png'.
    """
    fijos = gdat.calcular_puntos_fijos()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Radar VALDES (origen del sistema local) - hexágono como en cartas de navegación
    ax.plot(0, 0, marker='h', markersize=22, color=PALETTE[5],
            label='Radar VALDES', linestyle='None', zorder=10,
            markeredgecolor='black', markeredgewidth=1.5)
    ax.annotate('RADAR\nVALDES', (0, 0), xytext=(18, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', color=PALETTE[5])

    # Waypoints RATAS, NUBLO, ROVAK - triángulos negros como fix points en cartas de navegación
    # Solo el primero lleva label para la leyenda ("Fix points")
    ax.plot(fijos[0, 0], fijos[0, 1], marker='^', markersize=14,
            color='black', label='Fix points', linestyle='None', zorder=10,
            markeredgecolor='black', markeredgewidth=1.2)
    ax.annotate('RATAS', (fijos[0, 0], fijos[0, 1]), xytext=(14, 14),
                textcoords='offset points', fontsize=10, fontweight='bold')

    ax.plot(fijos[1, 0], fijos[1, 1], marker='^', markersize=14,
            color='black', linestyle='None', zorder=10,
            markeredgecolor='black', markeredgewidth=1.2)
    ax.annotate('NUBLO', (fijos[1, 0], fijos[1, 1]), xytext=(14, 14),
                textcoords='offset points', fontsize=10, fontweight='bold')

    ax.plot(fijos[2, 0], fijos[2, 1], marker='^', markersize=14,
            color='black', linestyle='None', zorder=10,
            markeredgecolor='black', markeredgewidth=1.2)
    ax.annotate('ROVAK', (fijos[2, 0], fijos[2, 1]), xytext=(14, 14),
                textcoords='offset points', fontsize=10, fontweight='bold')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Ubicacion de nodos y radar")
    # Aumentar espaciado entre elementos de la leyenda para evitar solapamiento
    ax.legend(loc='upper left', labelspacing=1.2)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    guardar(fig, "nodos.png")


def plot_trayectorias(pos_reales, medidas_radar, estados_cv, estados_ca):
    """
    Genera gráfica comparando trayectorias real, medidas radar y estimaciones Kalman.

    Args:
        pos_reales: Matriz de tamaño n×2 con posiciones reales.
        medidas_radar: Matriz de tamaño n×2 con medidas [rho, theta].
        estados_cv: Matriz de tamaño n×4 con estimaciones del modelo CV.
        estados_ca: Matriz de tamaño n×6 con estimaciones del modelo CA.
    """
    # Convertir medidas radar de polares magnéticas a Cartesianas geográficas (sumando declinación)
    theta_geo = medidas_radar[:, 1] + cnfg.DECLINACION_MAG
    x_radar = medidas_radar[:, 0] * np.sin(theta_geo)
    y_radar = medidas_radar[:, 0] * np.cos(theta_geo)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Líneas limpias sin marcadores para mejor legibilidad
    ax.plot(pos_reales[:, 0], pos_reales[:, 1],
            label="Trayectoria real", color=PALETTE[0], linewidth=2.5)
    ax.plot(x_radar, y_radar, '.', alpha=0.4, markersize=3,
            label='Medidas radar (con ruido)', color=PALETTE[5])
    ax.plot(estados_cv[:, 0], estados_cv[:, 1], '--',
            label="Kalman CV", color=PALETTE[1], linewidth=1.8)
    ax.plot(estados_ca[:, 0], estados_ca[:, 1], '--',
            label="Kalman CA", color=PALETTE[2], linewidth=1.8)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trayectorias: real, medidas radar y estimaciones Kalman")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.margins(0.05)
    guardar(fig, "trayectorias.png")


def plot_trayectoria_comparacion(pos_reales, medidas_radar, estados_cv, estados_ca):
    """
    Genera gráfica de trayectorias para comparar ambos filtros Kalman.

    Args:
        pos_reales: Matriz de tamaño n×2 con posiciones reales.
        medidas_radar: Matriz de tamaño n×2 con medidas [rho, theta].
        estados_cv: Matriz de tamaño n×4 con estimaciones del modelo CV.
        estados_ca: Matriz de tamaño n×6 con estimaciones del modelo CA.
    """
    # Convertir medidas radar de polares magnéticas a Cartesianas geográficas (sumando declinación)
    theta_geo = medidas_radar[:, 1] + cnfg.DECLINACION_MAG
    x_radar = medidas_radar[:, 0] * np.sin(theta_geo)
    y_radar = medidas_radar[:, 0] * np.cos(theta_geo)

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.plot(pos_reales[:, 0], pos_reales[:, 1],
            label="Trayectoria real", color=PALETTE[0], linewidth=2.5)
    ax.plot(x_radar, y_radar, '.', alpha=0.4, markersize=3,
            label='Medidas radar (con ruido)', color=PALETTE[5])
    ax.plot(estados_cv[:, 0], estados_cv[:, 1], '--',
            label="Kalman CV", color=PALETTE[1], linewidth=1.5, alpha=0.7)
    ax.plot(estados_ca[:, 0], estados_ca[:, 1], '--',
            label="Kalman CA", color=PALETTE[2], linewidth=1.5, alpha=0.7)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trayectorias: real, medidas radar y estimaciones Kalman")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.margins(0.05)
    guardar(fig, "trayectoria_comparacion.png")



def plot_medidas_con_ruido(pos_reales, medidas_radar):
    """
    Genera gráficas comparando valores reales vs medidos de rho y theta.

    Args:
        pos_reales: Matriz de tamaño n×2 con posiciones reales.
        medidas_radar: Matriz de tamaño n×2 con medidas [rho, theta].
    """
    x_real = pos_reales[:, 0]
    y_real = pos_reales[:, 1]
    rho_real = np.sqrt(x_real**2 + y_real**2)
    # Azimut real: ángulo desde el Norte (eje Y) en sentido horario
    theta_real = np.arctan2(x_real, y_real)

    rho_medido = medidas_radar[:, 0]
    theta_medido = medidas_radar[:, 1] + cnfg.DECLINACION_MAG

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(rho_real, label='Rho real', color=PALETTE[0], linewidth=2)
    axes[0].plot(rho_medido, '.', alpha=0.5, markersize=3,
                 label='Rho medido (con ruido)', color=PALETTE[5])
    axes[0].set_xlabel("Paso de tiempo")
    axes[0].set_ylabel("Distancia (m)")
    axes[0].set_title("Rho: valor real vs medido")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.degrees(theta_real), label='Theta real', color=PALETTE[0], linewidth=2)
    axes[1].plot(np.degrees(theta_medido), '.', alpha=0.5, markersize=3,
                 label='Theta medido (con ruido)', color=PALETTE[6])
    axes[1].set_xlabel("Paso de tiempo")
    axes[1].set_ylabel("Angulo (grados)")
    axes[1].set_title("Theta: valor real vs medido")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    guardar(fig, "medidas_con_ruido.png")


def plot_error_posicion(pos_reales, estados_cv, estados_ca):
    """
    Genera gráficas del error de posición en función del tiempo para ambos filtros.

    Args:
        pos_reales: Matriz de tamaño n×2 con posiciones reales.
        estados_cv: Matriz de tamaño n×4 con estimaciones del modelo CV.
        estados_ca: Matriz de tamaño n×6 con estimaciones del modelo CA.
    """
    err_cv = np.linalg.norm(pos_reales[:, :2] - estados_cv[:, :2], axis=1)
    err_ca = np.linalg.norm(pos_reales[:, :2] - estados_ca[:, :2], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(err_cv, color=PALETTE[1], label="Error CV", linewidth=1.5)
    axes[0].set_xlabel("Paso de tiempo")
    axes[0].set_ylabel("Error (m)")
    axes[0].set_title("Error de posicion - Modelo CV")
    axes[0].legend()

    axes[1].plot(err_ca, color=PALETTE[2], label="Error CA", linewidth=1.5)
    axes[1].set_xlabel("Paso de tiempo")
    axes[1].set_ylabel("Error (m)")
    axes[1].set_title("Error de posicion - Modelo CA")
    axes[1].legend()
    guardar(fig, "error_posicion.png")





def plot_velocidades(estados_cv, estados_ca):
    """
    Genera gráficas de las componentes de velocidad estimadas por cada filtro.

    Args:
        estados_cv: Matriz de tamaño n×4 con estimaciones [x, y, vx, vy].
        estados_ca: Matriz de tamaño n×6 con estimaciones [x, y, vx, vy, ax, ay].
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(estados_cv[:, 2], label="v_x", color=PALETTE[3], linewidth=1.5)
    axes[0].plot(estados_cv[:, 3], label="v_y", color=PALETTE[4], linewidth=1.5)
    axes[0].set_xlabel("Paso de tiempo")
    axes[0].set_ylabel("Velocidad (m/s)")
    axes[0].set_title("Velocidades estimadas - CV")
    axes[0].legend()

    axes[1].plot(estados_ca[:, 2], label="v_x", color=PALETTE[3], linewidth=1.5)
    axes[1].plot(estados_ca[:, 3], label="v_y", color=PALETTE[4], linewidth=1.5)
    axes[1].set_xlabel("Paso de tiempo")
    axes[1].set_ylabel("Velocidad (m/s)")
    axes[1].set_title("Velocidades estimadas - CA")
    axes[1].legend()
    guardar(fig, "velocidades.png")


#: Factor de conversión de m/s a nudos.
MS_TO_KNOTS = 1.94384


def plot_velocidad_total(vel_reales, estados_cv, estados_ca):
    """
    Genera gráfica comparando la velocidad total real vs estimada de cada filtro.

    Args:
        vel_reales: Vector de tamaño n con velocidades reales en m/s.
        estados_cv: Matriz de tamaño n×4 con estimaciones del modelo CV.
        estados_ca: Matriz de tamaño n×6 con estimaciones del modelo CA.
    """
    n = len(vel_reales)
    tiempos = np.arange(n) * cnfg.DT

    # Calcular velocidad total (magnitud) de cada estimación
    v_total_cv = np.sqrt(estados_cv[:, 2]**2 + estados_cv[:, 3]**2) * MS_TO_KNOTS
    v_total_ca = np.sqrt(estados_ca[:, 2]**2 + estados_ca[:, 3]**2) * MS_TO_KNOTS
    v_real = vel_reales * MS_TO_KNOTS

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tiempos, v_real,
            label="Velocidad real", color=PALETTE[0], linewidth=2.5)
    ax.plot(tiempos, v_total_cv, '--',
            label="Kalman CV", color=PALETTE[1], linewidth=1.5, alpha=0.7)
    ax.plot(tiempos, v_total_ca, '--',
            label="Kalman CA", color=PALETTE[2], linewidth=1.5, alpha=0.7)
    # Líneas horizontales para V1 y V2
    ax.axhline(cnfg.V1_MS * MS_TO_KNOTS, color='gray', linestyle=':', alpha=0.7,
               label=f"V1 = {cnfg.V1_MS * MS_TO_KNOTS:.0f} kt")
    ax.axhline(cnfg.V2_MS * MS_TO_KNOTS, color='gray', linestyle='--', alpha=0.7,
               label=f"V2 = {cnfg.V2_MS * MS_TO_KNOTS:.0f} kt")

    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Velocidad (nudos)")
    ax.set_title("Velocidad total estimada vs real")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    guardar(fig, "velocidad_total.png")


def plot_medidas_radar(medidas_radar):
    """
    Genera gráficas de rho y theta a lo largo del tiempo.

    Args:
        medidas_radar: Matriz de tamaño n×2 con medidas [rho, theta].
    """
    rho = medidas_radar[:, 0]
    theta = np.degrees(medidas_radar[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(rho, color=PALETTE[5], linewidth=1.5)
    axes[0].set_xlabel("Paso de tiempo")
    axes[0].set_ylabel("rho (m)")
    axes[0].set_title("Medidas de distancia (rho)")

    axes[1].plot(theta, color=PALETTE[6], linewidth=1.5)
    axes[1].set_xlabel("Paso de tiempo")
    axes[1].set_ylabel("theta (grados)")
    axes[1].set_title("Medidas de angulo (theta)")
    guardar(fig, "medidas_radar.png")


def plot_comparacion_cv_ca(pos_reales, estados_cv, estados_ca):
    """
    Genera gráfica comparando el error de posición de ambos filtros.

    Args:
        pos_reales: Matriz de tamaño n×2 con posiciones reales.
        estados_cv: Matriz de tamaño n×4 con estimaciones del modelo CV.
        estados_ca: Matriz de tamaño n×6 con estimaciones del modelo CA.
    """
    err_cv = np.linalg.norm(pos_reales[:, :2] - estados_cv[:, :2], axis=1)
    err_ca = np.linalg.norm(pos_reales[:, :2] - estados_ca[:, :2], axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(err_cv, label="Error CV", color=PALETTE[1], linewidth=1.5)
    ax.plot(err_ca, label="Error CA", color=PALETTE[2], linewidth=1.5)
    ax.set_xlabel("Paso de tiempo")
    ax.set_ylabel("Error (m)")
    ax.set_title("Comparacion de error CV vs CA")
    ax.legend()
    guardar(fig, "comparacion_cv_ca.png")
    plt.close(fig)


def plot_trazas_covarianza(trazas_cv, trazas_ca):
    """
    Genera gráfica de la traza de la covarianza P a lo largo del tiempo para ambos modelos.

    Args:
        trazas_cv: Vector con la traza de la covarianza de posición P del modelo CV.
        trazas_ca: Vector con la traza de la covarianza de posición P del modelo CA.
    """
    n = len(trazas_cv)
    tiempos = np.arange(n) * cnfg.DT

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Escala logarítmica para ver la convergencia inicial (10^6) y el régimen permanente
    ax.semilogy(tiempos, trazas_cv, label="Covarianza CV (Traza P_pos)", color=PALETTE[1], linewidth=2.0)
    ax.semilogy(tiempos, trazas_ca, label="Covarianza CA (Traza P_pos)", color=PALETTE[2], linewidth=2.0)
    
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Traza de la covarianza de posicion (m^2)")
    ax.set_title("Evolucion de la traza de la covarianza de posicion P")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    
    guardar(fig, "trazas_covarianza.png")


def plot_ganancias_kalman(gains_cv, gains_ca):
    """
    Genera gráfica comparativa de las ganancias de Kalman (K) para posición y velocidad en ambos filtros.

    Args:
        gains_cv: Secuencia de ganancias de Kalman del modelo CV (n×4×2).
        gains_ca: Secuencia de ganancias de Kalman del modelo CA (n×6×2).
    """
    n = len(gains_cv)
    tiempos = np.arange(n) * cnfg.DT

    # Extraer ganancias de posición en X (K[0,0]) y velocidad en X (K[2,0])
    # Debido a la simetría X-Y en el plano cartesiano, las de Y son análogas
    k_pos_cv = gains_cv[:, 0, 0]
    k_pos_ca = gains_ca[:, 0, 0]

    k_vel_cv = gains_cv[:, 2, 0]
    k_vel_ca = gains_ca[:, 2, 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot de Ganancia de Posición
    axes[0].plot(tiempos, k_pos_cv, label="Ganancia Posición CV", color=PALETTE[1], linewidth=2.0)
    axes[0].plot(tiempos, k_pos_ca, label="Ganancia Posición CA", color=PALETTE[2], linewidth=2.0)
    axes[0].set_xlabel("Tiempo (s)")
    axes[0].set_ylabel("Ganancia de Kalman K_p (m/m)")
    axes[0].set_title("Evolucion de la ganancia de posicion K_p")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot de Ganancia de Velocidad
    axes[1].plot(tiempos, k_vel_cv, label="Ganancia Velocidad CV", color=PALETTE[3], linewidth=2.0)
    axes[1].plot(tiempos, k_vel_ca, label="Ganancia Velocidad CA", color=PALETTE[4], linewidth=2.0)
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel("Ganancia de Kalman K_v (1/s)")
    axes[1].set_title("Evolucion de la ganancia de velocidad K_v")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    guardar(fig, "ganancias_kalman.png")


def ejecutar():
    """
    Ejecuta todas las visualizaciones y guarda las gráficas.

    Genera 9 gráficas en el directorio 'resultados/':
    - nodos.png
    - trayectoria_comparacion.png
    - medidas_con_ruido.png
    - error_posicion.png
    - velocidades.png
    - velocidad_total.png
    - medidas_radar.png
    - comparacion_cv_ca.png
    - trazas_covarianza.png
    - ganancias_kalman.png

    También imprime estadísticas resumen en la consola.
    """
    pos_reales, vel_reales, medidas_radar, estados_cv, estados_ca, trazas_cv, trazas_ca, gains_cv, gains_ca = obj.ejecutar()

    plot_nodos()
    plot_trayectoria_comparacion(pos_reales, medidas_radar, estados_cv, estados_ca)
    plot_medidas_con_ruido(pos_reales, medidas_radar)
    plot_error_posicion(pos_reales, estados_cv, estados_ca)
    plot_velocidades(estados_cv, estados_ca)
    plot_velocidad_total(vel_reales, estados_cv, estados_ca)
    plot_medidas_radar(medidas_radar)
    plot_comparacion_cv_ca(pos_reales, estados_cv, estados_ca)
    plot_trazas_covarianza(trazas_cv, trazas_ca)
    plot_ganancias_kalman(gains_cv, gains_ca)

    print("Graficas guardadas en resultados/")

    n_medidas = len(medidas_radar)
    print(f"Numero de medidas procesadas: {n_medidas}")
    print(f"Ultima posicion estimada CV: x={estados_cv[-1, 0]:.2f}, y={estados_cv[-1, 1]:.2f}")
    print(f"Ultima posicion estimada CA: x={estados_ca[-1, 0]:.2f}, y={estados_ca[-1, 1]:.2f}")
    print(f"Posicion real final: x={pos_reales[-1, 0]:.2f}, y={pos_reales[-1, 1]:.2f}")


if __name__ == "__main__":
    ejecutar()
