import numpy as np

class Policy:

    def __init__(self, init_mean_value=0):
        # estimación inicial (optimista/pesimista) común a todos los brazos
        self.init_mean_value = init_mean_value
        self._n_arms = None
        self._qhat = None    # estimaciones q̂(a) en línea
        self._pulls = None   # contadores por brazo N_t(a)
        self.t = 0           # tiradas ya completadas

    def setup(self, num_arms):
        self._n_arms = int(num_arms)
        # vector de medias iniciales y contadores en cero
        self._qhat = np.full(self._n_arms, float(self.init_mean_value), dtype=float)
        self._pulls = np.zeros(self._n_arms, dtype=int)
        self.t = 0
        
    def choose(self) -> int:
        raise NotImplementedError

    def tell_reward(self, arm: int, reward: float) -> None:
        # actualizar contador y media incremental del brazo
        self._pulls[arm] += 1
        n = self._pulls[arm]
        # q_n = q_{n-1} + (r - q_{n-1}) / n
        self._qhat[arm] = self._qhat[arm] + (reward - self._qhat[arm]) / n
        # avanzar el reloj de jugadas completadas
        self.t += 1

    @property
    def mean_estimates(self):
        # devolver copia defensiva
        return self._qhat.copy()
    
class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon, init_mean_value=0):
        super().__init__(init_mean_value=init_mean_value)
        self.epsilon = epsilon  # float o callable

    def setup(self, num_arms):
        super().setup(num_arms)

    def _eps_now(self):
        # el usuario recibe t empezando en 1 (paso de decisión actual)
        t_for_user = self.t + 1
        if callable(self.epsilon):
            return float(self.epsilon(t_for_user))
        return float(self.epsilon)

    def choose(self) -> int:
        # ---- arranque: seleccionar cualquier brazo jamás probado (en orden) ----
        untried = np.where(self._pulls == 0)[0]
        if untried.size > 0:
            return int(untried[0])

        eps = self._eps_now()
        # Explorar con probabilidad ε_t
        if np.random.random() < eps:
            return int(np.random.randint(0, self._n_arms))
        # Explotar: elegir el de mayor media estimada
        return int(np.argmax(self._qhat))
    
class UCB(Policy):
    def __init__(self, c, init_mean_value=0):
        super().__init__(init_mean_value=init_mean_value)
        self.c = float(c)

    def setup(self, num_arms):
        super().setup(num_arms)

    @property
    def exploration_terms(self):
        t_for_log = max(1, int(self.t))  # sin +1
        out = np.zeros(self._n_arms, dtype=float)
        mask = self._pulls > 0
        out[mask] = np.sqrt(np.log(float(t_for_log)) / self._pulls[mask])
        return out

    def choose(self):
        # warm-start determinista: probar cada brazo una vez en orden 0,1,2,...
        untried = np.where(self._pulls == 0)[0]
        if untried.size > 0:
            return int(untried[0])

        # score = q̂ + c * exploration
        scores = self._qhat + self.c * self.exploration_terms
        return int(np.argmax(scores))  # empates → índice menor