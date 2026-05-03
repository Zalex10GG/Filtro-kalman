from transformaciones import dms_to_rad, geodetic_to_geocentric, geocentric_to_local, cartesian_to_radar
from generar_datos import generar_trayectoria, generar_radar
import numpy as np
import matplotlib.pyplot as plt

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

def main():
    # --- TRANSFORMACIÓN DE COORDENADAS ---
    lat_valdes = dms_to_rad(lat_valdes_dms)
    lon_valdes = dms_to_rad(lon_valdes_dms)

    fijos_geo = []
    for key in ["RATAS", "NUBLO", "ROVAK"]:
        lat_str, lon_str = puntos[key].split()
        lat_rad = dms_to_rad(lat_str)
        lon_rad = dms_to_rad(lon_str)
        fijos_geo.append(geodetic_to_geocentric(lat_rad, lon_rad, h_vuelo))
    
    fijos_local = [geocentric_to_local(p, lat_valdes, lon_valdes, h_valdes) for p in fijos_geo]
    fijos2D = np.array([[f[0], f[1]] for f in fijos_local]) # Solo Xl, Yl para la trayectoria

    # --- GENERACIÓN DE LA TRAYECTORIA ---
    trayectoria_local = generar_trayectoria(fijos2D, dt, v1_ms, v2_ms, accel)

    # --- CONVERSIÓN DE LA TRAYECTORIA A MEDICIONES RADAR ---
    rho_real, theta_real = cartesian_to_radar(trayectoria_local[:, 0], trayectoria_local[:, 1])
    medidas_reales = np.column_stack((rho_real, theta_real))

    # --- GENERAR MEDIDAS RADAR CON RUIDO ---
    medidas_radar = generar_radar(medidas_reales, sigma_rho, sigma_theta)

if __name__ == "__main__":
    main()