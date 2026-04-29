import numpy as np

# --- CONSTANTES WGS84 ---
a = 6378137.0
e2 = 0.00669437999013

# --- COORDENADAS DEL RADAR ---
lat_valdes_dms = "420852.94N"
lon_valdes_dms = "042532.44W"
h_valdes = 3182 * 0.3048 # Elevación en metros

# --- COORDENADAS DE LOS PUNTOS FIJOS (RATAS, NUBLO, ROVAK) ---
puntos = {
    "RATAS": "423428N 0040151W",
    "NUBLO": "423958N 0045920W",
    "ROVAK": "424431N 0055123W"
}

h_vuelo = 10000.0 # Suponemos altitud de crucero en metros (FL330)

# --- DATOS DE LA TRAYECTORIA ---
dt = 4.0  # Paso de tiempo en segundos
v1_ms = 410 * 0.51444  # m/s (Velocidad inicial)
v2_ms = 510 * 0.51444  # m/s (Velocidad final)
accel = 1  # m/s^2 (Aceleración constante)

# --- CONFIGURACIÓN DEL RADAR ---
sigma_rho = 30.0               # metros (1-sigma)
sigma_theta_deg = 0.068        # grados (1-sigma)
sigma_theta = np.radians(sigma_theta_deg) # conversión a radianes

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

def generar_trayectoria(fijos):
    """
    Genera una trayectoria 2D (Xl, Yl) entre los puntos fijos con las fases de velocidad y aceleración.
    """
    trayectoria = [fijos[0]]
    
    # --- FASE 1: RATAS -> NUBLO (Velocidad Constante v1) ---
    dist_total_1 = np.linalg.norm(fijos[1] - fijos[0])
    dir_1 = (fijos[1] - fijos[0]) / dist_total_1
    dist_recorrida = 0.0
    
    while dist_recorrida < dist_total_1:
        dist_recorrida += v1_ms * dt
        if dist_recorrida >= dist_total_1:
            trayectoria.append(fijos[1]) # Forzamos llegada exacta
            break
        trayectoria.append(fijos[0] + dist_recorrida * dir_1)

    # --- FASE 2: NUBLO -> ROVAK (Aceleración y Crucero v2) ---
    dist_total_2 = np.linalg.norm(fijos[2] - fijos[1])
    dir_2 = (fijos[2] - fijos[1]) / dist_total_2
    dist_recorrida_f2 = 0.0
    vel_actual = v1_ms
    
    # Distancia teórica para alcanzar v2: v2^2 = v1^2 + 2*a*d
    d_necesaria_acel = (v2_ms**2 - v1_ms**2) / (2 * accel)
    
    # Iniciamos el bucle de la Fase 2
    while dist_recorrida_f2 < dist_total_2:
        # ¿Estamos acelerando o ya estamos en v2?
        if vel_actual < v2_ms:
            # Calculamos avance con aceleración: s = v*t + 0.5*a*t^2
            paso_dist = vel_actual * dt + 0.5 * accel * (dt**2)
            vel_actual += accel * dt
            # Si nos pasamos de v2 en este segundo, la limitamos
            if vel_actual > v2_ms: vel_actual = v2_ms
        else:
            # Velocidad constante v2
            paso_dist = v2_ms * dt
            
        dist_recorrida_f2 += paso_dist
        
        if dist_recorrida_f2 >= dist_total_2:
            trayectoria.append(fijos[2]) # Forzamos llegada exacta
            break
        
        trayectoria.append(fijos[1] + dist_recorrida_f2 * dir_2)
            
    return np.array(trayectoria)

def cartesian_to_radar(x, y):
    """
    Convierte coordenadas cartesianas a polares (Rango, Acimut)
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y) # atan2(x, y) mide desde el eje Y (Norte)
    return rho, theta

lat_valdes = dms_to_rad(lat_valdes_dms)
lon_valdes = dms_to_rad(lon_valdes_dms)

fijos = []

for nombre, coords in puntos.items():
    c = coords.split()
    lat_p = dms_to_rad(c[0])
    lon_p = dms_to_rad(c[1])
    
    # Transformar a Geocéntricas
    P_p_G = geodetic_to_geocentric(lat_p, lon_p, h_vuelo)
    
    # Transformar a Locales respecto al Radar
    P_local = geocentric_to_local(P_p_G, lat_valdes, lon_valdes, h_valdes)
    
    fijos.append([nombre, P_local[0], P_local[1], P_local[2]])

fijos2D = np.array([[f[1], f[2]] for f in fijos]) # Solo Xl, Yl para la trayectoria

pos_reales = generar_trayectoria(fijos2D)

# Suponemos que 'posiciones_reales' es el array (N, 2) del paso anterior
x_real = pos_reales[:, 0]
y_real = pos_reales[:, 1]

# Generar Medidas Reales (sin ruido) en formato Radar
rho_real, theta_real = cartesian_to_radar(x_real, y_real)
medidas_reales = np.column_stack((rho_real, theta_real))

# Generar Ruido Aleatorio (Gaussiano)
# np.random.seed(42) # Para reproducibilidad. Puedes comentar esta línea para obtener diferentes resultados cada vez.
ruido_rho = np.random.normal(0, sigma_rho, len(rho_real))
ruido_theta = np.random.normal(0, sigma_theta, len(theta_real))

# Crear Medidas Radar
rho_medido = rho_real + ruido_rho
theta_medido = theta_real + ruido_theta
medidas_radar = np.column_stack((rho_medido, theta_medido))