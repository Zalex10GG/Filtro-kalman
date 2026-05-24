"""
Módulo de objetivos: orquestación del procesamiento completo.

Coordina la generación de datos, inicialización de filtros y ejecución
de las estimaciones de Kalman para los modelos CV (Constant Velocity)
y CA (Constant Acceleration).
"""

import numpy as np
import src.config as cnfg
import src.generar_datos as gdat
import src.medidas as mds
import src.kalman as klmn


def preparar_datos():
    """
    Genera todos los datos necesarios para la simulación.

    Returns:
        tuple: (pos_reales, vel_reales, medidas_radar) con:
            - pos_reales: Matriz de tamaño n×2 con posiciones reales [x, y].
            - vel_reales: Vector de tamaño n con velocidades reales en m/s.
            - medidas_radar: Matriz de tamaño n×2 con medidas [rho, theta].
    """
    fijos2D = gdat.calcular_puntos_fijos()
    pos_reales, vel_reales = gdat.generar_trayectoria(fijos2D)
    medidas_radar, _ = gdat.generar_medidas_radar(pos_reales)
    return pos_reales, vel_reales, medidas_radar


def inicializar_estado_cv(pos_ratas, pos_nublo, v_cruzero):
    """
    Inicializa el estado y covarianza para el modelo CV (Constant Velocity).

    El vector de estado para CV es [x, y, vx, vy].
    La posición inicial se toma del waypoint RATAS.
    La velocidad inicial se calcula en la dirección RATAS -> NUBLO.

    Args:
        pos_ratas: Vector [x, y] del waypoint RATAS en metros.
        pos_nublo: Vector [x, y] del waypoint NUBLO en metros.
        v_cruzero: Velocidad de crucero en m/s.

    Returns:
        tuple: (x0, P0) donde:
            - x0: Vector de estado inicial [x, y, vx, vy].
            - P0: Matriz de covarianza inicial de tamaño 4×4.
    """
    # Dirección RATAS -> NUBLO
    dir_nublo = pos_nublo - pos_ratas
    dist = np.linalg.norm(dir_nublo)
    if dist > 0:
        dir_nublo = dir_nublo / dist

    # Descomponer velocidad en componentes x, y
    v_este = v_cruzero * dir_nublo[0]
    v_norte = v_cruzero * dir_nublo[1]

    # Estado inicial: posición en RATAS, velocidad en dirección NUBLO
    x0 = np.array([pos_ratas[0], pos_ratas[1], v_este, v_norte])

    # Covarianza inicial: alta incertidumbre en posición, menor en velocidad
    P0 = np.diag([cnfg.P0_DIAG, cnfg.P0_DIAG, 100.0, 100.0])
    return x0, P0


def inicializar_estado_ca(pos_ratas, pos_nublo, v_cruzero):
    """
    Inicializa el estado y covarianza para el modelo CA (Constant Acceleration).

    El vector de estado para CA es [x, y, vx, vy, ax, ay].
    Añade aceleraciones iniciales cero al modelo CV.

    Args:
        pos_ratas: Vector [x, y] del waypoint RATAS en metros.
        pos_nublo: Vector [x, y] del waypoint NUBLO en metros.
        v_cruzero: Velocidad de crucero en m/s.

    Returns:
        tuple: (x0, P0) donde:
            - x0: Vector de estado inicial [x, y, vx, vy, ax, ay].
            - P0: Matriz de covarianza inicial de tamaño 6×6.
    """
    x0_cv, P0_cv = inicializar_estado_cv(pos_ratas, pos_nublo, v_cruzero)
    # Añadir aceleraciones iniciales cero
    x0 = np.append(x0_cv, [0.0, 0.0])
    # Covarianza con mayor incertidumbre en aceleraciones
    P0 = np.diag([cnfg.P0_DIAG, cnfg.P0_DIAG, 100.0, 100.0, 10.0, 10.0])
    return x0, P0


def construir_matrices_cv():
    """
    Construye las matrices del modelo CV (Constant Velocity).

    El modelo CV, suponer velocidad constante con ruido de proceso en la aceleración.
    Estado: [x, y, vx, vy]
    Modelo: x_k = A @ x_{k-1} + ruido

    Returns:
        tuple: (A, H, Q) donde:
            - A: Matriz de transición de tamaño 4×4.
            - H: Matriz de observación de tamaño 2×4.
            - Q: Matriz de covarianza del ruido de proceso de tamaño 4×4.
    """
    dt = cnfg.DT
    # Matriz de transición de estado
    A = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # Matriz de observación (solo observamos posición, no velocidad)
    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])
    # Matriz de entrada para el ruido de proceso
    G = np.array([
        [0.5 * dt**2, 0.0],
        [0.0, 0.5 * dt**2],
        [dt, 0.0],
        [0.0, dt]
    ])
    # Covarianza del ruido de aceleración (proceso)
    Q_sigma = np.array([
        [cnfg.Q_ACCEL_SIGMA**2, 0.0],
        [0.0, cnfg.Q_ACCEL_SIGMA**2]
    ])
    # Q = G @ Q_sigma @ G.T
    Q = G @ Q_sigma @ G.T
    return A, H, Q


def construir_matrices_ca():
    """
    Construye las matrices del modelo CA (Constant Acceleration).

    El modelo CA, suponer aceleración constante con ruido de proceso en el jerk.
    Estado: [x, y, vx, vy, ax, ay]

    Returns:
        tuple: (A, H, Q) donde:
            - A: Matriz de transición de tamaño 6×6.
            - H: Matriz de observación de tamaño 2×6.
            - Q: Matriz de covarianza del ruido de proceso de tamaño 6×6.
    """
    dt = cnfg.DT
    # Matriz de transición de estado con término de aceleración
    A = np.array([
        [1.0, 0.0, dt, 0.0, 0.5*dt**2, 0.0],
        [0.0, 1.0, 0.0, dt, 0.0, 0.5*dt**2],
        [0.0, 0.0, 1.0, 0.0, dt, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ])
    # Matriz de observación
    H = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ])
    # Matriz de entrada para el ruido de jerk
    G = np.array([
        [dt**3/6, 0.0],
        [0.0, dt**3/6],
        [0.5*dt**2, 0.0],
        [0.0, 0.5*dt**2],
        [dt, 0.0],
        [0.0, dt]
    ])
    # Covarianza del ruido de jerk (proceso)
    Q_sigma = np.array([
        [cnfg.Q_JERK_SIGMA**2, 0.0],
        [0.0, cnfg.Q_JERK_SIGMA**2]
    ])
    # Q = G @ Q_sigma @ G.T
    Q = G @ Q_sigma @ G.T
    return A, H, Q


def ejecutar_cv(z_k_seq, R_k_seq, x0, P0, A, H, Q):
    """
    Ejecuta el filtro de Kalman con modelo CV.

    Args:
        z_k_seq: Secuencia de medidas en Cartesianas shape (n, 2).
        R_k_seq: Secuencia de matrices de covarianza shape (n, 2, 2).
        x0: Estado inicial 4x1.
        P0: Covarianza inicial 4x4.
        A: Matriz de transición 4x4.
        H: Matriz de observación 2x4.
        Q: Matriz de covarianza del ruido 4x4.

    Returns:
        tuple: (estados, trazas_p, gains) donde:
            - estados: Matriz de tamaño n×4 con los estados estimados [x, y, vx, vy].
            - trazas_p: Vector de tamaño n con la traza de la covarianza P en cada paso.
            - gains: Matriz de tamaño n×4×2 con las ganancias de Kalman en cada paso.
    """
    kf = klmn.KalmanFilter(A, H, Q, x0, P0)
    estados = []
    trazas_p = [np.trace(P0[:2, :2])]
    gains = []
    for z, R in zip(z_k_seq, R_k_seq):
        kf.predict()
        kf.update(z, R)
        estados.append(kf.x.copy())
        trazas_p.append(np.trace(kf.P[:2, :2]))
        gains.append(kf.K.copy())
    return np.array(estados), np.array(trazas_p), np.array(gains)


def ejecutar_ca(z_k_seq, R_k_seq, x0, P0, A, H, Q):
    """
    Ejecuta el filtro de Kalman con modelo CA.

    Args:
        z_k_seq: Secuencia de medidas en Cartesianas shape (n, 2).
        R_k_seq: Secuencia de matrices de covarianza shape (n, 2, 2).
        x0: Estado inicial 6x1.
        P0: Covarianza inicial 6x6.
        A: Matriz de transición 6x6.
        H: Matriz de observación 2x6.
        Q: Matriz de covarianza del ruido 6x6.

    Returns:
        tuple: (estados, trazas_p, gains) donde:
            - estados: Matriz de tamaño n×6 con los estados estimados [x, y, vx, vy, ax, ay].
            - trazas_p: Vector de tamaño n + 1 con la traza de la covarianza P en cada paso (incluyendo P0).
            - gains: Matriz de tamaño n×6×2 con las ganancias de Kalman en cada paso.
    """
    kf = klmn.KalmanFilter(A, H, Q, x0, P0)
    estados = []
    trazas_p = [np.trace(P0[:2, :2])]
    gains = []
    for z, R in zip(z_k_seq, R_k_seq):
        kf.predict()
        kf.update(z, R)
        estados.append(kf.x.copy())
        trazas_p.append(np.trace(kf.P[:2, :2]))
        gains.append(kf.K.copy())
    return np.array(estados), np.array(trazas_p), np.array(gains)


def ejecutar():
    """
    Ejecuta el proceso completo: genera datos y ejecuta ambos filtros Kalman.

    Returns:
        tuple: (pos_reales, vel_reales, medidas_radar, estados_cv, estados_ca, trazas_cv, trazas_ca, gains_cv, gains_ca) con:
            - pos_reales: Matriz de tamaño n×2 con posiciones reales.
            - vel_reales: Vector de tamaño n con velocidades reales.
            - medidas_radar: Matriz de tamaño n×2 con medidas del radar.
            - estados_cv: Matriz de tamaño n×4 con estimaciones del modelo CV.
            - estados_ca: Matriz de tamaño n×6 con estimaciones del modelo CA.
            - trazas_cv: Vector de tamaño n con traza de P del modelo CV.
            - trazas_ca: Vector de tamaño n con traza de P del modelo CA.
            - gains_cv: Matriz de ganancias del modelo CV (n×4×2).
            - gains_ca: Matriz de ganancias del modelo CA (n×6×2).
    """
    # Generar datos
    pos_reales, vel_reales, medidas_radar = preparar_datos()
    z_k_seq, R_k_seq = mds.calcular_z_y_r_batch(medidas_radar)

    # Obtener posiciones de los waypoints
    fijos_raw = gdat.calcular_puntos_fijos()
    pos_ratas = fijos_raw[0]
    pos_nublo = fijos_raw[1]

    # Ejecutar modelo CV
    x0_cv, P0_cv = inicializar_estado_cv(pos_ratas, pos_nublo, cnfg.V1_MS)
    A_cv, H_cv, Q_cv = construir_matrices_cv()
    estados_cv, trazas_cv, gains_cv = ejecutar_cv(z_k_seq, R_k_seq, x0_cv, P0_cv, A_cv, H_cv, Q_cv)

    # Ejecutar modelo CA
    x0_ca, P0_ca = inicializar_estado_ca(pos_ratas, pos_nublo, cnfg.V1_MS)
    A_ca, H_ca, Q_ca = construir_matrices_ca()
    estados_ca, trazas_ca, gains_ca = ejecutar_ca(z_k_seq, R_k_seq, x0_ca, P0_ca, A_ca, H_ca, Q_ca)

    return pos_reales, vel_reales, medidas_radar, estados_cv, estados_ca, trazas_cv, trazas_ca, gains_cv, gains_ca

