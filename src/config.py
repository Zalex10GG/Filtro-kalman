"""
Configuración del sistema de filtro de Kalman para control de tráfico aéreo.

Este módulo contiene todos los parámetros de simulación del sistema:
- Coordenadas del radar de referencia y de los waypoints
- Parámetros de vuelo (velocidades, aceleración)
- Parámetros del radar (sigmas de medida)
- Parámetros del filtro de Kalman (covarianza inicial, ruido de proceso)
"""

import numpy as np

#: Latitud del radar VALDES en formato DMS (grados, minutos, segundos con hemisferio).
LAT_VALDES = "420852.94N"

#: Longitud del radar VALDES en formato DMS. Valor negativo por ser hemisferio W.
LON_VALDES = "042532.44W"

#: Altura del radar VALDES en metros. 3182 pies convertido a metros (1 pie = 0.3048 m).
H_VALDES = 3182 * 0.3048

#: Coordenadas DMS del waypoint RATAS (punto inicial de la trayectoria).
PTO_RATAS = "423428N 0040151W"

#: Coordenadas DMS del waypoint NUBLO (punto intermedio, lugar del giro).
PTO_NUBLO = "423958N 0045920W"

#: Coordenadas DMS del waypoint ROVAK (punto final de la trayectoria).
PTO_ROVAK = "424431N 0055123W"

#: Altitud de vuelo de las aeronaves en metros (FL100 = 10000 ft).
H_VUELO = 10000.0

#: Paso de tiempo de simulación en segundos. Cada 4s se procesa una medida del radar.
DT = 4.0

#: Velocidad inicial de vuelo en m/s. 410 nudos convertidos a m/s (1 nudo = 0.51444 m/s).
V1_MS = 410 * 0.51444

#: Velocidad máxima de vuelo en m/s. 510 nudos convertidos a m/s (fase de crucero tras aceleración).
V2_MS = 510 * 0.51444

#: Aceleración del avión en m/s² durante la fase de aceleración entre V1 y V2.
ACCEL = 10.0

#: Desviación estándar del error de distancia (rho) del radar en metros.
SIGMA_RHO = 30.0

#: Desviación estándar del error angular del radar en grados sexagesimales.
SIGMA_THETA_DEG = 0.068

#: Desviación estándar del error angular del radar en radianes. Se calcula en init().
SIGMA_THETA = None

#: Valor diagonal de la covarianza inicial del estado (P0) para posición.
#: Indica alta incertidumbre inicial sobre la posición exacta.
P0_DIAG = 1e6

#: Desviación estándar del ruido de aceleración (proceso) en m/s².
#: Cuanto mayor, más confianza en las correcciones de posición del filtro.
Q_ACCEL_SIGMA = 1.0

#: Desviación estándar del ruido de jerk (derivada de aceleración) en m/s³.
#: Se calcula como ACCEL/DT para mantener consistencia física.
Q_JERK_SIGMA = Q_ACCEL_SIGMA / DT


def init():
    """
    Inicializa las variables que requieren numpy.

    Se llama automáticamente al importar el módulo para convertir
    SIGMA_THETA de grados a radianes.
    """
    global SIGMA_THETA
    import numpy as np
    SIGMA_THETA = np.radians(SIGMA_THETA_DEG)


init()