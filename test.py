import time
import numpy as np
import yaml
from pathlib import Path

from sb3_contrib import RecurrentPPO

from imitation_env import MouseArmImitationEnv

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

model_name = config["general"]["name"]
path = Path(f"./agents/{model_name}")
if not path.is_dir():
    raise ValueError("model doesnt exist??")
print("loading model...")
#get the most recent iteration
numbers = []
for item in path.iterdir():
    if item.is_file() and item.suffix == ".zip" and item.stem.isdigit():
        numbers.append(int(item.stem))
iteration = max(numbers) if numbers else None
if iteration is None:
    raise ValueError("no iterations in this folder??")
print(f"iteration: {iteration}")

with open(f"./agents/{model_name}/config.yml", "r") as file:
    config_loaded = yaml.safe_load(file)

# Create env (rendering so you can watch it)
env = MouseArmImitationEnv(
    render_mode="human", 
    model=config_loaded["environment"]["model"],
    kinematics=config_loaded["environment"]["kinematics"],
    w_bone_diff=config_loaded["environment"]["w_bone_diff"],
    w_elbow=config_loaded["environment"]["w_elbow"],
    w_paw=config_loaded["environment"]["w_paw"],
    w_effort=config_loaded["environment"]["w_effort"],
    w_jitter=config_loaded["environment"]["w_jitter"],
    w_action=config_loaded["environment"]["w_action"],
    control_dt=config_loaded["environment"]["control_dt"],
    n_substeps=config_loaded["environment"]["n_substeps"],
)
model = RecurrentPPO.load(f"./agents/{model_name}/{iteration}", env=env)
    

obs, info = env.reset()
env.render()
# LSTM state handling
lstm_states = None
episode_start = np.array([True], dtype=bool)

episode_reward = 0.0
slowmo = config["testing"]["slowmo"]  # seconds per step (increase for slower playback)

terminated = False
truncated = False
input("Starting, must do ctrl+c in terminal to quit, otherwise will run forever; enter to continue...")
while True:
    # Predict action with LSTM state
    action, lstm_states = model.predict(
        obs,
        state=lstm_states,
        episode_start=episode_start,
        deterministic=True,
    )

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    episode_reward += float(reward)
    # If the episode ended, tell the LSTM to reset next step
    episode_start[:] = terminated or truncated
    if terminated or truncated:
        lstm_states = None
        obs, info = env.reset()
        env.render()
        input("Episode ended, resetting; enter to continue...")

    time.sleep(slowmo)

