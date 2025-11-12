import numpy as np
import gymnasium as gym

# --- Etapa 1: "Descobrir" a classe base dinamicamente ---
# Isso evita o ModuleNotFoundError
try:
    # Usamos v5, como o warning sugeriu, mas v4 também funciona
    temp_env = gym.make("InvertedPendulum-v5")
    # Pega a classe real (ex: ...inverted_pendulum.InvertedPendulumEnv)
    InvertedPendulumEnvBaseClass = temp_env.unwrapped.__class__
    temp_env.close()
    
    print(f"Classe base do ambiente encontrada: {InvertedPendulumEnvBaseClass}")
    
except Exception as e:
    print("Erro crítico ao tentar encontrar a classe base do InvertedPendulum.")
    print("Verifique se 'gymnasium[mujoco]' ou 'gymnasium-robotics' está instalado.")
    print(f"Erro original: {e}")
    raise e

# Esta classe herda do ambiente padrão do MuJoCo
class CustomInvertedPendulumEnv(InvertedPendulumEnvBaseClass):
    """
    Versão customizada do InvertedPendulum que SÓ muda o limite
    do ângulo de falha de 0.2 rad (11.5°) para um valor maior.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Vamos definir nosso novo limite aqui (em radianos)
        # 1.57 radianos é aprox. 90 graus.
        self.theta_failure_threshold = 1.57 

    # Nós sobrescrevemos a função step SÓ para mudar a condição de término
    def step(self, action):
        
        # A física original do step() é executada
        # (Isso atualiza self.data)
        self.do_simulation(action, self.frame_skip)

        # Pega a observação
        ob = self._get_obs()
        
        # Pega as posições (qpos) para checar os limites
        qpos = self.data.qpos
        
        # --- A ÚNICA MUDANÇA É AQUI ---
        # Código original: terminated = bool(not np.isfinite(qpos).all() or (np.abs(qpos[1]) > 0.2))
        # Nosso código:
        terminated = bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > self.theta_failure_threshold))
        # -------------------------------

        reward = 1.0 # Recompensa padrão (não importa para nós)

        if self.render_mode == "human":
            self.render()
            
        # truncation=False pois o TimeLimit wrapper cuida disso
        return ob, reward, terminated, False, {}

def register_custom_env():
    """Registra nosso ambiente customizado no Gymnasium."""
    gym.register(
        id='MyInvertedPendulum-v0',
        entry_point='custom_envs:CustomInvertedPendulumEnv',
        max_episode_steps=1000 # 1000 passos * 0.02s/passo = 20s (como no seu gráfico)
    )