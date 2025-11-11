import numpy as np
import control as ct
from constants import M, mp, L, g

# --- Classe Base (Interface) ---
class BaseController:
    """Classe base para nossos controladores."""
    def __init__(self):
        pass

    def compute_action(self, state):
        """Calcula a força de controle u."""
        # state é [x, x_dot, theta, theta_dot]
        raise NotImplementedError("Implemente este método na subclasse")

# --- Controlador LQR ---
class LQRController(BaseController):
    def __init__(self):
        super().__init__()
        # Monta o modelo linearizado (A, B)
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, (mp * g) / M, 0],
            [0, 0, 0, 1],
            [0, 0, (M + mp) * g / (M * L), 0]
        ])
        B = np.array([[0], [1 / M], [0], [-1 / (M * L)]])
        
        # Define custos (Q para estados, R para controle)
        Q = np.diag([1, 1, 10, 1])  # Penaliza mais o ângulo
        R = np.array([[0.1]])      # Penaliza o esforço
        
        # Calcula os ganhos K do LQR
        self.K, S, E = ct.lqr(A, B, Q, R)
        print(f"Ganhos LQR K: {self.K}")

    def compute_action(self, state):
        # u = -Kx
        # state precisa ser (4,1) para a multiplicação
        state_col = state.reshape(4, 1)
        force_array = -np.dot(self.K, state_col)
        force = force_array[0, 0]

        return force / 100.0  # Converte para sinal de controle adequado

# --- Controlador por Linearização Exata (FL) ---
class FLController(BaseController):
    def __init__(self, kp_th, kd_th, kp_x, kd_x):
        super().__init__()
        self.kp_th = kp_th
        self.kd_th = kd_th
        self.kp_x = kp_x
        self.kd_x = kd_x
        print(f"Controlador FL inicializado com Kp_th={kp_th}, Kd_th={kd_th}, Kp_x={kp_x}, Kd_x={kd_x}")

    def compute_action(self, state):
        # Desempacota o estado
        x, x_dot, theta, theta_dot = state
        
        # --- CUIDADO: Ponto de singularidade ---
        if np.abs(np.cos(theta)) < 0.01:
            # Se estiver perto de 90 graus, a fórmula explode.
            # Retorna uma ação máxima ou zero para evitar NaN.
            return 0.0 
        
        # 1. Calcula o controle linear 'v'
        v_theta = -self.kp_th * theta - self.kd_th * theta_dot
        v_x = -self.kp_x * x - self.kd_x * x_dot

        v = v_theta + v_x
        
        # 2. Calcula a lei de controle não linear 'u'
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        tan_th = np.tan(theta)
        theta_dot_sq = theta_dot**2
        
        # Termos da equação que derivamos
        denominador_comum = L * (M + mp * sin_th**2)
        
        # u = (1/g(x)) * (v - f(x))
        f_x = (-mp * L * sin_th * cos_th * theta_dot_sq + (M + mp) * g * sin_th) / denominador_comum
        g_x = cos_th / denominador_comum
        
        u = (v - f_x) / g_x
        
        return -u / 100.0  # Converte para sinal de controle adequado