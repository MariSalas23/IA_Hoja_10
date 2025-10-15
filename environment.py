import numpy as np
import pandas as pd

class BanditProblem:

    def __init__(self, distributions):
        # Guardar una versión normalizada de las distribuciones de entrada
        self._arms_cfg = []
        for vals, probs in distributions:
            vals_arr = np.asarray(vals, dtype=float)
            prob_arr = np.asarray(probs, dtype=float)
            assert vals_arr.shape == prob_arr.shape and vals_arr.ndim == 1, \
                "Cada brazo debe definirse como (valores, probabilidades) con la misma longitud."
            # Normalizar por si hay pequeñas desviaciones numéricas
            prob_arr = prob_arr / prob_arr.sum()
            self._arms_cfg.append((vals_arr, prob_arr))

        self.K = len(self._arms_cfg)

        # Medias verdaderas por brazo para métricas de referencia
        self._true_means = np.array(
            [np.dot(v, p) for (v, p) in self._arms_cfg],
            dtype=float
        )
        self._best_mean = float(self._true_means.max())
        # Conjunto de brazos óptimos (permite empates)
        self._optimal_arms = set(
            np.where(np.isclose(self._true_means, self._best_mean))[0].tolist()
        )

    def _draw_reward(self, arm_idx: int) -> float:
        vals, probs = self._arms_cfg[arm_idx]
        # Muestreo de una r.v. discreta dada por (vals, probs)
        return float(np.random.choice(vals, p=probs))

    def simulate_policy(self, policy, max_t):
        # Avisar a la política cuántos brazos hay
        policy.setup(self.K)

        log = {
            'arm': [],
            'reward': [],
            'avg_reward': [],
            'optimal_action_rate': [],
            'cumulative_regret': [],
        }

        acc_reward = 0.0
        acc_regret = 0.0
        opt_hits = 0

        for round_idx in range(1, max_t + 1):
            a = policy.choose()
            r = self._draw_reward(a)
            policy.tell_reward(a, r)

            # Actualización de métricas acumuladas
            acc_reward += r
            if a in self._optimal_arms:
                opt_hits += 1
            acc_regret += (self._best_mean - self._true_means[a])

            # Registro por ronda
            log['arm'].append(int(a))
            log['reward'].append(float(r))
            log['avg_reward'].append(acc_reward / round_idx)
            log['optimal_action_rate'].append(opt_hits / round_idx)
            log['cumulative_regret'].append(acc_regret)

        return pd.DataFrame(log)