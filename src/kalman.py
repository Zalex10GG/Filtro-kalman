"""
Módulo del Filtro de Kalman lineal.

Implementa un filtro de Kalman discreto para sistemas lineales:
- Predict: Propaga el estado y la covarianza usando el modelo dinámico
- Update: Corrige el estado usando una medida observada
"""

import numpy as np


class KalmanFilter:
    """
    Filtro de Kalman discreto para sistemas lineales.

    Attributes:
        A: Matriz de transición de estado (estado_k = A @ estado_{k-1}).
        H: Matriz de observación (medida = H @ estado + ruido).
        Q: Matriz de covarianza del ruido de proceso.
        x: Vector de estado actual.
        P: Matriz de covarianza del estado actual.
    """

    def __init__(self, A, H, Q, x0, P0):
        """
        Inicializa el filtro de Kalman.

        Args:
            A: Matriz de transición de estado de tamaño n×n.
            H: Matriz de observación de tamaño m×n.
            Q: Matriz de covarianza del ruido de proceso de tamaño n×n.
            x0: Vector de estado inicial de tamaño n.
            P0: Matriz de covarianza inicial de tamaño n×n.
        """
        self.A = A
        self.H = H
        self.Q = Q
        self.x = x0.copy()
        self.P = P0.copy()

    def predict(self):
        """
        Paso de predicción: propaga el estado y covarianza al siguiente instante.

        usa el modelo dinámico x_k = A @ x_{k-1} y añade el ruido de proceso Q.
        """
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z, R):
        """
        Paso de corrección: incorpora una nueva medida para actualizar el estado.

        Args:
            z: Vector de medida observada de tamaño m.
            R: Matriz de covarianza de la medida de tamaño m×m.
        """
        # Covarianza de la innovación
        S = self.H @ self.P @ self.H.T + R
        # Ganancia de Kalman
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        # Innovación (error de medición)
        y = z - self.H @ self.x
        # Actualización del estado
        self.x = self.x + K @ y
        # Actualización de la covarianza (forma de Joseph real)
        I_KH = np.eye(len(self.x)) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T