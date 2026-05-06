"""
Módulo de transformaciones de coordenadas geodésicas.

Proporciona funciones para convertir entre diferentes sistemas de coordenadas:
- DMS (grados, minutos, segundos) a radianes
- Geodésicas (lat, lon, h) a geocéntricas (X, Y, Z)
- Geocéntricas a sistema de referencia local con origen en el radar
- Cartesianas a polares (radar)
"""

import numpy as np

#: Semieje mayor del elipsoide WGS84 en metros.
A = 6378137.0

#: Excentricidad al cuadrado del elipsoide WGS84 (e² = 2f - f²).
E2 = 0.00669437999013


def dms_to_rad(dms_str):
    """
    Convierte coordenadas en formato DMS ( grados, minutos, segundos) a radianes.

    Args:
        dms_str: String con formato DDMMSS.ssN o DDMMSS.ssW (ej: "423428N").

    Returns:
        float: Coordenada en radianes. Positivo para N/E, negativo para S/W.

    Raises:
        ValueError: Si el formato DMS no es reconocido.
    """
    import re
    dms_str = dms_str.strip()
    # Formato: 2-3 dígitos para grados, 2 para minutos, 2+ decimales para segundos, letra de hemisferio
    match = re.match(r'^(\d{2,3})(\d{2})(\d{2}(?:\.\d+)?)([NSWE])$', dms_str)
    if not match:
        raise ValueError(f"Formato DMS no reconocido: {dms_str}")
    deg = float(match.group(1))
    minu = float(match.group(2))
    sec = float(match.group(3))
    decimal = deg + minu/60 + sec/3600
    # Aplicar signo según hemisferio
    if match.group(4) in 'SW':
        decimal = -decimal
    return np.radians(decimal)


def geodetic_to_geocentric(lat, lon, h):
    """
    Convierte coordenadas geodésicas (lat, lon, h) a geocéntricas (X, Y, Z).

    Utiliza el modelo de elipsoide WGS84 para la transformación.

    Args:
        lat: Latitud en radianes.
        lon: Longitud en radianes.
        h: Altitud sobre el elipsoide en metros.

    Returns:
        numpy.ndarray: Vector [XG, YG, ZG] en metros en el sistema geocéntrico.
    """
    # Radio de curvatura en el primer vertical
    nu = A / np.sqrt(1 - E2 * np.sin(lat)**2)
    # Conversión a geocéntricas
    XG = (nu + h) * np.cos(lat) * np.cos(lon)
    YG = (nu + h) * np.cos(lat) * np.sin(lon)
    ZG = (nu * (1 - E2) + h) * np.sin(lat)
    return np.array([XG, YG, ZG])


def geocentric_to_local(P_geo, lat_ref, lon_ref, h_ref):
    """
    Transforma coordenadas geocéntricas a sistema local con origen en el radar.

    El sistema local tiene:
    - X: positivo hacia el Este
    - Y: positivo hacia el Norte
    - Z: positivo hacia arriba (altura)

    Args:
        P_geo: Vector [X, Y, Z] geocéntrico del punto a convertir.
        lat_ref: Latitud del radar de referencia en radianes.
        lon_ref: Longitud del radar de referencia en radianes.
        h_ref: Altura del radar de referencia en metros.

    Returns:
        numpy.ndarray: Vector [x, y, z] en metros en coordenadas locales.
    """
    # Centro geocéntrico del radar
    T = geodetic_to_geocentric(lat_ref, lon_ref, h_ref)
    # Matriz de rotación del sistema geocéntrico al local
    S = np.array([
        [-np.sin(lon_ref),  np.cos(lon_ref), 0.0],
        [-np.sin(lat_ref)*np.cos(lon_ref), -np.sin(lat_ref)*np.sin(lon_ref), np.cos(lat_ref)],
        [np.cos(lat_ref)*np.cos(lon_ref),  np.cos(lat_ref)*np.sin(lon_ref), np.sin(lat_ref)]
    ])
    return S @ (P_geo - T)


def cartesian_to_radar(x, y):
    """
    Convierte coordenadas cartesianas a polares (sistema radar).

    Args:
        x: Coordenada X en metros.
        y: Coordenada Y en metros.

    Returns:
        tuple: (rho, theta) donde:
            - rho: Distancia radial en metros.
            - theta: Ángulo en radianes [-π, π].
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return rho, theta