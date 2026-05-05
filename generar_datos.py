from transformaciones import cartesian_to_radar
import numpy as np

def generar_trayectoria(fijos, dt, v1_ms, v2_ms, accel):
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

def generar_radar(medidas_reales, sigma_rho, sigma_theta):
    """
    Genera medidas de radar con ruido gaussiano a partir de las medidas reales.
    """
    np.random.seed(42) # Para reproducibilidad. Puedes comentar esta línea para obtener diferentes resultados cada vez.
    ruido_rho = np.random.normal(0, sigma_rho, len(medidas_reales))
    ruido_theta = np.random.normal(0, sigma_theta, len(medidas_reales))

    rho_medido = medidas_reales[:, 0] + ruido_rho
    theta_medido = medidas_reales[:, 1] + ruido_theta
    return np.column_stack((rho_medido, theta_medido))