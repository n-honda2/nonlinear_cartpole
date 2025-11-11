import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.spaces import Box
import numpy as np

class ContinuousCartPoleEnv(CartPoleEnv):
    """
    Uma versão do CartPole-v1 que aceita uma força contínua como ação.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # O espaço de ação agora é contínuo, representando a força
        # Vamos usar a força máxima padrão do CartPole (10N)
        self.force_mag = 10.0
        self.action_space = Box(low=-self.force_mag, 
                                high=self.force_mag, 
                                shape=(1,), 
                                dtype=np.float32)

    def step(self, action):
        # A ação recebida 'action' é um array, ex: [2.5]
        # Pegamos o valor escalar da força
        force = action[0] 
        
        # Garante que a força não exceda os limites (embora o Box já deva fazer isso)
        force = np.clip(force, -self.force_mag, self.force_mag)
        
        # A física original do CartPole usa 'force' baseado em 'action' (0 ou 1)
        # Aqui, nós passamos nossa própria 'force' contínua
        
        # ---- Início da física do step() original ----
        # (copiado/adaptado da fonte do gymnasium)
        err_msg = f"{action!r} ({type(action)}) inválido"
        assert self.state is not None, "Chame reset() antes de step()"
        
        x, x_dot, theta, theta_dot = self.state
        
        # A única mudança é aqui:
        # force = self.force_mag if action == 1 else -self.force_mag
        # A linha acima é substituída pela 'force' que recebemos
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        # ---- Fim da física do step() original ----

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                gym.logger.warn(
                    "Você está chamando step() em um ambiente que já terminou."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
            
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

def register_env():
    """Registra o ambiente customizado no Gymnasium."""
    gym.register(
        id='ContinuousCartPole-v0',
        entry_point='custom_envs:ContinuousCartPoleEnv',
        max_episode_steps=500, # Você pode aumentar isso se quiser
    )