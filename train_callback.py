import json
import numpy as np
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback


class RecurrentTestEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq,
        n_eval_episodes,
        model_path,
        deterministic=True,
        verbose=False,
    ):
        super().__init__(verbose)

        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.model_path = Path(model_path)
        self.deterministic = deterministic

        self.best_mean_reward = -np.inf

        best_info_file = self.model_path / "best_info.json"
        if best_info_file.is_file():
            with open(best_info_file, "r") as f:
                best_info = json.load(f)

            self.best_mean_reward = float(best_info["best_mean_reward"])

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        episode_rewards = []
        episode_lengths = []

        obs = self.eval_env.reset()
        num_envs = self.eval_env.num_envs

        lstm_states = None
        episode_starts = np.ones((num_envs,), dtype=bool)

        current_rewards = np.zeros(num_envs, dtype=np.float32)
        current_lengths = np.zeros(num_envs, dtype=np.int32)

        while len(episode_rewards) < self.n_eval_episodes:
            actions, lstm_states = self.model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=self.deterministic,
            )

            obs, rewards, dones, infos = self.eval_env.step(actions)

            current_rewards += rewards
            current_lengths += 1
            episode_starts = dones

            for i, done in enumerate(dones):
                if done:
                    episode_rewards.append(float(current_rewards[i]))
                    episode_lengths.append(int(current_lengths[i]))

                    current_rewards[i] = 0.0
                    current_lengths[i] = 0

                    if len(episode_rewards) >= self.n_eval_episodes:
                        break

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        mean_ep_length = float(np.mean(episode_lengths))

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

            self.model.save(self.model_path / "best_model")

            with open(self.model_path / "best_info.json", "w") as f:
                json.dump(
                    {
                        "best_mean_reward": self.best_mean_reward,
                        "num_timesteps": self.num_timesteps,
                    },
                    f,
                    indent=4,
                )

            if self.verbose:
                print(
                    f"[test eval] new best model at {self.num_timesteps} steps: "
                    f"mean_reward={mean_reward:.4f}"
                )
        elif self.verbose:
            print(
                f"[test eval] steps={self.num_timesteps} "
                f"mean_reward={mean_reward:.4f} "
                f"best={self.best_mean_reward:.4f}"
            )

        self.logger.record("test/mean_reward", mean_reward)
        #self.logger.record("test/std_reward", std_reward)
        self.logger.record("test/mean_ep_length", mean_ep_length)
        self.logger.record("test/best_mean_reward", self.best_mean_reward)
        self.logger.dump(self.num_timesteps)

        return True