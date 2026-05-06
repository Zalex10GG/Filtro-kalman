"""
Módulo para procesar medidas de radar y calcular matrices de observación.

Proporciona funciones para:
- Convertir medidas radar (rho, theta) a coordenadas Cartesianas (x, y)
- Calcular la matriz de covarianza de la medida R_k usando el método de
  Lerro & Bar-Shalom para compensar sesgos estadísticos
- Procesar un lote de medidas
"""

import numpy as np
import src.config as cnfg


def calcular_z_y_r(rho_m, theta_m):
    """
    Convierte una medida radar a espacio Cartesiano y calcula la covarianza.

    Aplica el método de Lerro & Bar-Shalom para desesgo y covarianza consistente:
    - Calcula el sesgo μ_a según la Ecuación 12
    - Calcula la covarianza R_a según las Ecuaciones 13a-13c
    - Resta el sesgo a la medida convertida: z^C = [x_m, y_m] - μ_a

    Args:
        rho_m: Distancia medida (r_m) en metros.
        theta_m: Ángulo medido (θ_m) en radianes.

    Returns:
        tuple: (z_k, R_k) donde:
            - z_k: Vector [x, y] en metros (medida desesgada en Cartesianas).
            - R_k: Matriz de covarianza de tamaño 2×2 de la medida convertida.
    """
    # 1. Convertir medidas polares a Cartesianas
    x_m = rho_m * np.cos(theta_m)
    y_m = rho_m * np.sin(theta_m)

    # Parámetros de ruido del radar
    drho = cnfg.SIGMA_RHO  # σ_r
    dtheta = cnfg.SIGMA_THETA  # σ_θ
    stheta2 = dtheta ** 2  # σ_θ²

    # 2. Calcular sesgo μ_a según Ecuación 12 de Lerro & Bar-Shalom
    # μ_a = [r_m cos(θ_m)(e^{-σ²} - e^{-σ²/2}), r_m sin(θ_m)(e^{-σ²} - e^{-σ²/2})]
    exp_neg = np.exp(-stheta2)
    exp_neg_half = np.exp(-stheta2 / 2)
    bias_factor = exp_neg - exp_neg_half
    mu_a = np.array([
        rho_m * np.cos(theta_m) * bias_factor,
        rho_m * np.sin(theta_m) * bias_factor
    ])

    # 3. Calcular covarianza R_a según Ecuaciones 13a-13c de Lerro & Bar-Shalom
    c = np.cos(theta_m)
    s = np.sin(theta_m)
    c2 = c ** 2
    s2 = s ** 2

    st = np.sinh(stheta2)
    ch = np.cosh(stheta2)
    st2 = np.sinh(2 * stheta2)
    ch2 = np.cosh(2 * stheta2)

    exp_neg2 = np.exp(-2 * stheta2)
    exp_neg4 = np.exp(-4 * stheta2)
    exp_pos = np.exp(stheta2)

    # Elemento R^11_a (Ecuación 13a)
    R11 = exp_neg2 * (
        rho_m ** 2 * (c2 * (ch2 - ch) + s2 * (st2 - st))
        + drho ** 2 * (c2 * (2 * ch2 - ch) + s2 * (2 * st2 - st))
    )

    # Elemento R^22_a (Ecuación 13b)
    R22 = exp_neg2 * (
        rho_m ** 2 * (s2 * (ch2 - ch) + c2 * (st2 - st))
        + drho ** 2 * (s2 * (2 * ch2 - ch) + c2 * (2 * st2 - st))
    )

    # Elemento R^12_a (Ecuación 13c)
    R12 = s * c * exp_neg4 * (drho ** 2 + (rho_m ** 2 + drho ** 2) * (1 - exp_pos))

    R_k = np.array([[R11, R12], [R12, R22]])

    # 4. Medida Cartesian final desesgada: z^C = [x_m, y_m]^T - μ_a
    z_k = np.array([x_m - mu_a[0], y_m - mu_a[1]])

    return z_k, R_k


def calcular_z_y_r_batch(medidas_radar):
    """
    Procesa un lote de medidas de radar usando debiasing de Lerro & Bar-Shalom.

    Args:
        medidas_radar: Matriz de tamaño n×2 con medidas [rho, theta] en metros y radianes.

    Returns:
        tuple: (z_k_seq, R_k_seq) donde:
            - z_k_seq: Matriz de tamaño n×2 con medidas desesgadas en Cartesianas.
            - R_k_seq: Tensor de tamaño n×2×2 con matrices de covarianza.
    """
    n = len(medidas_radar)
    z_list = []
    R_list = []

    for i in range(n):
        rho_m = medidas_radar[i, 0]
        theta_m = medidas_radar[i, 1]
        z_k, R_k = calcular_z_y_r(rho_m, theta_m)
        z_list.append(z_k)
        R_list.append(R_k)

    return np.array(z_list), np.array(R_list)