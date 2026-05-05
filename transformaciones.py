import numpy as np

# --- CONSTANTES WGS84 ---
a = 6378137.0
e2 = 0.00669437999013

def dms_to_rad(dms_str):
    """
    Convierte formato AIP (DDMMSSN/W) a radianes
    """
    # Ejemplo: 420042N -> 42.0116 grados
    deg = float(dms_str[:2])
    minu = float(dms_str[2:4])
    sec = float(dms_str[4:6])
    decimal = deg + minu/60 + sec/3600
    if 'S' in dms_str or 'W' in dms_str:
        decimal = -decimal
    return np.radians(decimal)

def geodetic_to_geocentric(lat, lon, h):
    """
    Transforma Geodésicas (L, G, H) a Geocéntricas (XG, YG, ZG)"""
    nu = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    XG = (nu + h) * np.cos(lat) * np.cos(lon)
    YG = (nu + h) * np.cos(lat) * np.sin(lon)
    ZG = (nu * (1 - e2) + h) * np.sin(lat)
    return np.array([XG, YG, ZG])

def geocentric_to_local(P_geo, lat_ref, lon_ref, h_ref):
    """
    Transforma coordenadas Geocéntricas a Cartesianas Locales (Xl, Yl, Zl)
    respecto a un radar de referencia.
    """

    nu = a / np.sqrt(1 - e2 * np.sin(lat_ref)**2)

    # Vector de traslación del origen
    T = np.array([
        (nu + h_ref) * np.cos(lat_ref) * np.cos(lon_ref),
        (nu + h_ref) * np.cos(lat_ref) * np.sin(lon_ref),
        (nu * (1 - e2) + h_ref) * np.sin(lat_ref)
    ])
    
    # Matriz de rotación al plano local (North, East, Up -> Xl, Yl, Zl)
    # Basado en el documento de EUROCONTROL (Apéndice A)
    S = np.array([
        [-np.sin(lon_ref), np.cos(lon_ref), 0],
        [-np.sin(lat_ref)*np.cos(lon_ref), -np.sin(lat_ref)*np.sin(lon_ref), np.cos(lat_ref)],
        [np.cos(lat_ref)*np.cos(lon_ref), np.cos(lat_ref)*np.sin(lon_ref), np.sin(lat_ref)]
    ])

    return S.dot(P_geo - T)

def cartesian_to_radar(x, y):
    """
    Convierte coordenadas cartesianas a polares (Rango, Acimut)
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y) # atan2(x, y) mide desde el eje Y (Norte)
    return rho, theta