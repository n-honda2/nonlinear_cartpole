import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from controllers import LQRController, FLController
import custom_envs 

custom_envs.register_env()

def run_simulation(controller, start_angle, render=True):
    """Roda uma simulação e retorna os dados."""
    env = gym.make("ContinuousCartPole-v0", render_mode="human" if render else None)
    
    # Define o ângulo inicial
    # obs, info = env.reset(seed=42)
    # env.unwrapped.state = [0, 0, start_angle, 0] # Define o estado manualmente
    # obs = np.array(env.unwrapped.state)
    
    # A API nova do Gymnasium recomenda passar opções no reset
    obs, info = env.reset(seed=42)
    obs = np.array([0, 0, start_angle, 0])
    env.unwrapped.state = obs

    data = {'time': [], 'theta': [], 'force': []}
    
    for t in range(5000): # Simula por 500 passos (10 segundos)
        # 1. Calcula a força contínua
        force = controller.compute_action(obs)
        
        # 2. Prepara a ação para o ambiente
        # Ação agora é um array numpy, ex: [2.5]
        # Também vamos limitar (saturar) a força para o máximo do atuador
        max_force = env.action_space.high[0]
        action = np.clip(force, -max_force, max_force)

        # 3. Aplica a ação no ambiente
        obs, reward, terminated, truncated, info = env.step(np.array([action]))
        
        # Salva os dados
        data['time'].append(t * 0.02) # O timestep do CartPole é 0.02s
        data['theta'].append(obs[2])  # obs[2] é o ângulo theta
        data['force'].append(force)
        
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

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    
    # 1. Inicializa os controladores
    lqr_ctrl = LQRController()
    fl_ctrl = FLController(kp=100.0, kd=20.0) # Ajuste Kp e Kd conforme necessário

    # --- Experimento 1: Ângulo Pequeno ---
    print("Rodando Experimento 1: Ângulo Pequeno (3 graus)")
    angle_small = 0.05 # ~3 graus
    
    results_lqr_small = run_simulation(lqr_ctrl, start_angle=angle_small)
    results_fl_small = run_simulation(fl_ctrl, start_angle=angle_small)
    
    plot_results({
        "LQR": results_lqr_small,
        "Linearização Exata (FL)": results_fl_small
    }, "Comparação com Ângulo Pequeno (3 graus)")

    # # --- Experimento 2: Ângulo Grande ---
    # print("Rodando Experimento 2: Ângulo Grande (17 graus)")
    # angle_large = 0.3 # ~17 graus
    
    # results_lqr_large = run_simulation(lqr_ctrl, start_angle=angle_large)
    # results_fl_large = run_simulation(fl_ctrl, start_angle=angle_large)
    
    # # Rode a simulação FL com renderização para ver
    # # print("Rodando FL com renderização...")
    # # run_simulation(fl_ctrl, start_angle=angle_large, render=True)
    
    # plot_results({
    #     "LQR": results_lqr_large,
    #     "Linearização Exata (FL)": results_fl_large
    # }, "Comparação com Ângulo Grande (17 graus)")