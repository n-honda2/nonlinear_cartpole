import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from controllers import LQRController, FLController
import mujoco
import custom_envs

custom_envs.register_custom_env()


def run_simulation(controller, start_angle, render=True):
    """Roda uma simulação com o ambiente InvertedPendulum-v4."""
    
    # --- USA O AMBIENTE MUJOCO CORRETO ---
    env = gym.make("MyInvertedPendulum-v0", render_mode="human" if render else None)
    
    # O estado do InvertedPendulum é [x, x_dot, cos(theta), sin(theta), theta_dot]
    # Precisamos ser mais espertos para definir o estado inicial.
    obs, info = env.reset(seed=42)
    
    # Define o estado manualmente (x, x_dot, theta, theta_dot)
    # qpos = [x, theta], qvel = [x_dot, theta_dot]
    qpos = np.array([0.0, start_angle]) # [pos_carrinho, angulo_pendulo]
    qvel = np.array([0.0, 0.0])       # [vel_carrinho, vel_pendulo]
    env.unwrapped.set_state(qpos, qvel)
    obs, _, _, _, _ = env.step(np.array([0.0])) # Pega a observação inicial

    data = {'time': [], 'theta': [], 'x':[], 'force_signal': []}
    
    for t in range(500):
        # O estado do LQR/FL é [x, x_dot, theta, theta_dot]
        # A observação 'obs' do InvertedPendulum é [x, x_dot, cos, sin, theta_dot]
        # Vamos construir o estado que nossos controladores entendem
        x = obs[0]
        x_dot = obs[2]
        theta = obs[1]
        theta_dot = obs[3]        
        
        # Monta o vetor de estado que nossos controladores esperam
        current_state = np.array([x, x_dot, theta, theta_dot])

        
        # 1. Calcula o sinal de controle (u_ctrl)
        u_ctrl = controller.compute_action(current_state)
        
        # 2. Satura o sinal de controle entre -3 e 3
        # O 'ctrlrange' do XML!
        action = np.clip(u_ctrl, -3.0, 3.0)
        
        # 3. Aplica a ação (precisa ser um array)
        obs, reward, terminated, truncated, info = env.step(np.array([action]))
        
        data['time'].append(t * env.unwrapped.dt) # env.dt é 0.02
        data['theta'].append(theta)
        data['x'].append(x)
        data['force_signal'].append(action)
        
        if terminated or truncated:
            break
            
    env.close()
    return data

def plot_results(results_map, title):
    """Plota os resultados da comparação."""
    plt.figure(figsize=(12, 6))
    plt.title(title)
    
    for label, data in results_map.items():
        # Converte o ângulo de radianos para graus
        theta_degrees = np.rad2deg(data['theta'])
        plt.plot(data['time'], theta_degrees, label=label)
        
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ângulo do Pêndulo (graus)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.title(title)

    for label, data in results_map.items():
        plt.plot(data['time'], data['x'], label=label)

    plt.xlabel("Tempo (s)")
    plt.ylabel("Posição do Carrinho (m)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.title(title)

    for label, data in results_map.items():
        plt.plot(data['time'], data['force_signal'], label=label)

    plt.xlabel("Tempo (s)")
    plt.ylabel("Força Aplicada")
    plt.legend()
    plt.grid(True)
    plt.show()

    

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    
    # 1. Inicializa os controladores
    lqr_ctrl = LQRController()
    fl_ctrl = FLController(kp_th=150.0, kd_th=20.0, kp_x=10.0, kd_x=15.0)

    # --- Experimento 1: Ângulo Pequeno ---
    print("Rodando Experimento 1: Ângulo Pequeno (3 graus)")
    angle_small = 0.05 # ~3 graus
    
    results_lqr_small = run_simulation(lqr_ctrl, start_angle=angle_small)
    results_fl_small = run_simulation(fl_ctrl, start_angle=angle_small)
    
    plot_results({
        "LQR": results_lqr_small,
        "Linearização Exata (FL)": results_fl_small
    }, "Comparação com Ângulo Pequeno (3 graus)")

    # --- Experimento 2: Ângulo Grande ---
    deg_ang = 15
    print(f"Rodando Experimento 2: Ângulo Grande ({deg_ang} graus)")
    angle_large = np.deg2rad(deg_ang)

    results_lqr_large = run_simulation(lqr_ctrl, start_angle=angle_large)
    results_fl_large = run_simulation(fl_ctrl, start_angle=angle_large)
        
    plot_results({
        "LQR": results_lqr_large,
        "Linearização Exata (FL)": results_fl_large
    }, f"Comparação com Ângulo Grande ({deg_ang} graus)")

    # --- Experimento 3: Limite RQL ---
    deg_ang = 27.2 # 19.2
    print(f"Rodando Experimento 3: Limite RQL ({deg_ang} graus)")
    angle_large = np.deg2rad(deg_ang)

    results_lqr_large = run_simulation(lqr_ctrl, start_angle=angle_large)
    
    plot_results({
        "LQR": results_lqr_large,
    }, f"Comparação com Limite RQL ({deg_ang} graus)")

    # --- Experimento 4: Limite Linearizado ---
    deg_ang = 28
    print(f"Rodando Experimento 4: Limite Linearizado ({deg_ang} graus)")
    angle_large = np.deg2rad(deg_ang)

    results_fl_large = run_simulation(fl_ctrl, start_angle=angle_large)
    
    plot_results({
        "Linearização Exata (FL)": results_fl_large
    }, f"Comparação com Limite Linearizado ({deg_ang} graus)")