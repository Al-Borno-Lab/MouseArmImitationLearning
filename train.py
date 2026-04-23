from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import torch
import yaml
from pathlib import Path
import shutil

from imitation_env import MouseArmImitationEnv

def make_env(rank, config):
    def _init():
        env = MouseArmImitationEnv(
            render_mode=None,
            model=config["environment"]["model"],
            kinematics=config["environment"]["kinematics"], 
            w_bone_diff=config["environment"]["w_bone_diff"],
            w_elbow=config["environment"]["w_elbow"],
            w_paw=config["environment"]["w_paw"],
            w_effort=config["environment"]["w_effort"],
            w_jitter=config["environment"]["w_jitter"],
            w_action=config["environment"]["w_action"],
            control_dt=config["environment"]["control_dt"],
            n_substeps=config["environment"]["n_substeps"],
        )
        return Monitor(env)
    return _init

if __name__ == "__main__":

    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    model_name = config["general"]["name"]
    path = Path(f"./agents/{model_name}")
    iteration = 0

    if path.is_dir():
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

        env = SubprocVecEnv([make_env(i, config_loaded) for i in range(config["training"]["num_envs"])])

        model = RecurrentPPO.load(f"./agents/{model_name}/{iteration}", env=env)
        iteration += 1 
    else:
        print("creating new model...")
        policy_kwargs = dict(
            share_features_extractor=True,
            shared_lstm=True,
            enable_critic_lstm=False,
            ortho_init=True,
            activation_fn=torch.nn.Tanh,

            lstm_hidden_size=config["policy"]["lstm_hidden_size"],
            n_lstm_layers=config["policy"]["n_lstm_layers"],
            net_arch=dict(pi=config["policy"]["net_arch_pi"], vf=config["policy"]["net_arch_vf"]),
        )

        env = SubprocVecEnv([make_env(i, config) for i in range(config["training"]["num_envs"])])

        model = RecurrentPPO(
            policy=MlpLstmPolicy,
            policy_kwargs=policy_kwargs,
            
            env=env,
            verbose=1,
            tensorboard_log=f"./agents/{model_name}/ppo_logs/",

            learning_rate=config["algorithm"]["learning_rate"],
            
            n_steps=config["algorithm"]["n_steps"],       # rollout length
            batch_size=config["algorithm"]["batch_size"],
            n_epochs=config["algorithm"]["n_epochs"],

            gamma=config["algorithm"]["gamma"],
            gae_lambda=config["algorithm"]["gae_lambda"],
            
            clip_range=config["algorithm"]["clip_range"],
            clip_range_vf = config["algorithm"]["clip_range_vf"],

            ent_coef=config["algorithm"]["ent_coef"],
            vf_coef=config["algorithm"]["vf_coef"],

            max_grad_norm=config["algorithm"]["max_grad_norm"],
        )

        #save config data just to have it
        config_path = Path("./config.yml")
        target_folder = Path(f"./agents/{model_name}")
        target_folder.mkdir(parents=True, exist_ok=True)
        destination = target_folder / config_path.name
        shutil.copy2(config_path, destination)


    print("start learning...")
    model.learn(
        total_timesteps=config["training"]["timesteps"],
        tb_log_name=model_name,
        reset_num_timesteps=False
    )
    model.save(f"./agents/{model_name}/{iteration}")