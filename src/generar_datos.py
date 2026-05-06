"""
Módulo para generar datos de trayectorias y medidas de radar.

Proporciona funciones para:
- Calcular waypoints fijos en coordenadas locales
- Generar trayectorias de vuelo con velocidades variables
- Simular medidas de radar con ruido gaussiano
"""

import numpy as np
import src.config as cnfg
import src.transformaciones as trns


def generar_trayectoria(fijos):
    """
    Genera una trayectoria de vuelo entre waypoints con aceleración gradual.

    La trayectoria se divide en dos tramos:
    1. Trayecto RATAS -> NUBLO: velocidad constante V1
    2. Trayecto NUBLO -> ROVAK: aceleración de V1 a V2

    Args:
        fijos: Array numpy de shape (3, 2) con coordenadas [x, y] de los 3 waypoints.

    Returns:
        tuple: (trayectoria, velocidades) donde:
            - trayectoria: Matriz de tamaño n×2 con posiciones [x, y] en cada paso.
            - velocidades: Vector de tamaño n con velocidad en m/s en cada paso.
    """
    dist_total_1 = np.linalg.norm(fijos[1] - fijos[0])
    dir_1 = (fijos[1] - fijos[0]) / dist_total_1
    t1_total = dist_total_1 / cnfg.V1_MS

    dist_total_2 = np.linalg.norm(fijos[2] - fijos[1])
    dir_2 = (fijos[2] - fijos[1]) / dist_total_2

    # Cálculos analíticos de tiempos para el tramo 2
    t_accel = (cnfg.V2_MS - cnfg.V1_MS) / cnfg.ACCEL
    dist_accel = cnfg.V1_MS * t_accel + 0.5 * cnfg.ACCEL * (t_accel**2)

    if dist_total_2 >= dist_accel:
        dist_const_2 = dist_total_2 - dist_accel
        t2_const = dist_const_2 / cnfg.V2_MS
        t2_total = t_accel + t2_const
    else:
        # Si la aceleración no se llega a completar antes de ROVAK
        t_accel = (-cnfg.V1_MS + np.sqrt(cnfg.V1_MS**2 + 2*cnfg.ACCEL*dist_total_2)) / cnfg.ACCEL
        t2_total = t_accel

    t_total = t1_total + t2_total

    trayectoria = []
    velocidades = []
    
    t = 0.0
    # Muestreo estricto cada DT, sin saltos anómalos ni frenazos artificiales
    while t <= t_total:
        if t <= t1_total:
            # En el primer tramo
            pos = fijos[0] + dir_1 * (cnfg.V1_MS * t)
            vel = cnfg.V1_MS
        else:
            # En el segundo tramo
            t_in_2 = t - t1_total
            if t_in_2 <= t_accel:
                dist_in_2 = cnfg.V1_MS * t_in_2 + 0.5 * cnfg.ACCEL * (t_in_2**2)
                vel = cnfg.V1_MS + cnfg.ACCEL * t_in_2
            else:
                dist_in_2 = dist_accel + cnfg.V2_MS * (t_in_2 - t_accel)
                vel = cnfg.V2_MS
            pos = fijos[1] + dir_2 * dist_in_2

        trayectoria.append(pos)
        velocidades.append(vel)
        t += cnfg.DT

    return np.array(trayectoria), np.array(velocidades)


def calcular_puntos_fijos():
    """
    Calcula las coordenadas locales de los 3 waypoints respecto al radar.

    Convierte las coordenadas DMS de cada waypoint a coordenadas locales
    (x, y) relativas al radar VALDES, usando el sistema de referencia local
    con origen en el radar.

    Returns:
        numpy.ndarray: Matriz de tamaño 3×2 con las coordenadas [x, y] locales
        de los waypoints en metros: [RATAS, NUBLO, ROVAK].
    """
    puntos = {
        "RATAS": cnfg.PTO_RATAS,
        "NUBLO": cnfg.PTO_NUBLO,
        "ROVAK": cnfg.PTO_ROVAK
    }
    # Obtener coordenadas del radar en radianes
    lat_valdes = trns.dms_to_rad(cnfg.LAT_VALDES)
    lon_valdes = trns.dms_to_rad(cnfg.LON_VALDES)
    fijos = []

    for nombre, coords in puntos.items():
        # Parsear coordenadas DMS
        c = coords.split()
        lat_p = trns.dms_to_rad(c[0])
        lon_p = trns.dms_to_rad(c[1])
        # Convertir a geocéntricas y luego a locales
        P_p_G = trns.geodetic_to_geocentric(lat_p, lon_p, cnfg.H_VUELO)
        P_local = trns.geocentric_to_local(P_p_G, lat_valdes, lon_valdes, cnfg.H_VALDES)
        fijos.append([P_local[0], P_local[1]])

    return np.array(fijos)


def generar_medidas_radar(pos_reales):
    """
    Simula medidas de radar a partir de posiciones reales.

    Convierte las posiciones Cartesianas a coordenadas polares (rho, theta)
    y añade ruido gaussiano para simular errores de medición del radar.

    Args:
        pos_reales: Array de shape (n, 2) con posiciones [x, y] reales en metros.

    Returns:
        tuple: (medidas_radar, medidas_reales) donde:
            - medidas_radar: Matriz de tamaño n×2 con [rho, theta] medidos (con ruido).
            - medidas_reales: Matriz de tamaño n×2 con [rho, theta] reales sin ruido.
    """
    x_real = pos_reales[:, 0]
    y_real = pos_reales[:, 1]
    # Convertir a coordenadas polares
    rho_real, theta_real = trns.cartesian_to_radar(x_real, y_real)
    medidas_reales = np.column_stack((rho_real, theta_real))

    # Generar ruido gaussiano independiente para rho y theta
    ruido_rho = np.random.normal(0, cnfg.SIGMA_RHO, len(rho_real))
    ruido_theta = np.random.normal(0, cnfg.SIGMA_THETA, len(theta_real))

    # Aplicar ruido a las medidas
    rho_medido = rho_real + ruido_rho
    theta_medido = theta_real + ruido_theta
    medidas_radar = np.column_stack((rho_medido, theta_medido))

    return medidas_radar, medidas_reales