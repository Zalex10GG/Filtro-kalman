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
    # Inicialización: primer punto es RATAS, velocidad inicial V1
    trayectoria = [fijos[0]]
    velocidades = [cnfg.V1_MS]

    # Tramo 1: RATAS -> NUBLO a velocidad constante
    dist_total_1 = np.linalg.norm(fijos[1] - fijos[0])  # Distancia total del tramo
    dir_1 = (fijos[1] - fijos[0]) / dist_total_1  # Vector director normalizado
    dist_recorrida = 0.0

    while dist_recorrida < dist_total_1:
        # Avanzar DT segundos a velocidad V1
        dist_recorrida += cnfg.V1_MS * cnfg.DT
        if dist_recorrida >= dist_total_1:
            # Llegamos al waypoint NUBLO
            trayectoria.append(fijos[1])
            velocidades.append(cnfg.V1_MS)
            break
        # Calcular posición actual a lo largo del tramo
        trayectoria.append(fijos[0] + dist_recorrida * dir_1)
        velocidades.append(cnfg.V1_MS)

    # Tramo 2: NUBLO -> ROVAK con aceleración
    dist_total_2 = np.linalg.norm(fijos[2] - fijos[1])
    dir_2 = (fijos[2] - fijos[1]) / dist_total_2
    dist_recorrida_f2 = 0.0
    vel_actual = cnfg.V1_MS  # Velocidad al inicio del tramo

    while dist_recorrida_f2 < dist_total_2:
        # Determinar cuánto avanzamos en este paso DT
        if vel_actual < cnfg.V2_MS:
            # Todavía estamos acelerando
            t_to_v2 = (cnfg.V2_MS - vel_actual) / cnfg.ACCEL
            if t_to_v2 < cnfg.DT:
                # Alcanzamos V2 antes de terminar el paso DT
                # Distancia = distancia hasta V2 + distancia a V2 constante
                paso_dist = vel_actual * t_to_v2 + 0.5 * cnfg.ACCEL * (t_to_v2**2) + cnfg.V2_MS * (cnfg.DT - t_to_v2)
                vel_actual = cnfg.V2_MS
            else:
                # Continuamos acelerando durante todo el paso DT
                paso_dist = vel_actual * cnfg.DT + 0.5 * cnfg.ACCEL * (cnfg.DT**2)
                vel_actual += cnfg.ACCEL * cnfg.DT
        else:
            # Ya alcanzamos la velocidad máxima, avanzamos a V2 constante
            paso_dist = cnfg.V2_MS * cnfg.DT

        dist_recorrida_f2 += paso_dist

        if dist_recorrida_f2 >= dist_total_2:
            # Llegamos al waypoint ROVAK
            trayectoria.append(fijos[2])
            velocidades.append(vel_actual)
            break

        # Calcular posición actual a lo largo del tramo
        trayectoria.append(fijos[1] + dist_recorrida_f2 * dir_2)
        velocidades.append(vel_actual)

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