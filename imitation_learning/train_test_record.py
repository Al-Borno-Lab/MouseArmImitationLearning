import multiprocessing as mp
import sys
import yaml
import termios
import tty
import json
import copy
import csv
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from tqdm import tqdm
import argparse

import gymnasium as gym

from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO_CFG
from skrl.agents.torch.ppo import PPO_RNN as PPO
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import StepTrainer
from skrl.utils import set_seed

from imitation_env import make_MouseArmImitationEnv
from data_helper import setup_files
from models import SharedModel, ReccurentLayerType

import torch
import random
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disable-progressbar",
        action="store_true",
        help="Disable the manual tqdm training progress bar.",
    )
    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    config_file = SCRIPT_DIR / "config.yml"

    with config_file.open("r") as file:
        config = yaml.safe_load(file)

    #CHECK FOR PREVIOUS AGENT
    is_new_agent: bool
    general_config = config["general"]
    mode = general_config["mode"]
    if mode not in ("train", "test", "record"):
        raise ValueError("general.mode must be one of: train, test, record")

    general_seed = config["general"]["seed"]
    set_seed(general_seed)

    random.seed(general_seed)
    np.random.seed(general_seed)
    torch.manual_seed(general_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(general_seed)
        torch.cuda.manual_seed_all(general_seed)

    model_name = general_config["name"]
    model_folder = general_config["folder"]
    model_path = Path(model_folder)
    full_model_path = model_path / model_name
    saved_config_path = full_model_path / config_file.name
    if full_model_path.is_dir():
        print("found existing agent...")
        is_new_agent = False
        with open(saved_config_path, "r") as file:
            saved_config = yaml.safe_load(file)
        saved_config["general"] = config["general"]
        saved_config["training"] = config["training"]
        saved_config["testing"] = config["testing"]
        config = saved_config
    else:
        print("creating new agent...")
        is_new_agent = True
        full_model_path.mkdir(parents=True, exist_ok=True)
        with open(saved_config_path, "w") as file:
            yaml.safe_dump(config, file, sort_keys=False)

    metadata_path = full_model_path / "metadata.yml"
    start_timestep = 0
    if metadata_path.is_file(): 
        with open(metadata_path, "r") as file:
            metadata = yaml.safe_load(file) or {}
        start_timestep = int(metadata.get("timesteps", 0))

    #DATA SETUP
    train_files, test_files = setup_files(
        path=config["environment"]["kinematics"],
        train_ratio=config["environment"]["train_ratio"],
        seed=config["environment"]["seed"],
    )

    all_files = train_files + test_files if len(train_files) != 1 else train_files

    # Normalized paths make membership checks reliable even if some paths are
    # relative and others contain "." or ".."
    train_file_set = {
        Path(path).expanduser().resolve()
        for path in train_files
    }

    test_file_set = {
        Path(path).expanduser().resolve()
        for path in test_files
    }

    if mode == "record":
        config["environment"]["kinematic_files"] = all_files
    else:
        config["environment"]["kinematic_files"] = train_files

    #ENVIRONMENT SETUP
    ctx = mp.get_context("fork")
    shared_var = ctx.Value("i", 0)

    training_config = config["training"]
    test_environment_config = copy.deepcopy(config["environment"]) # THIS JUST NEEDS TO BE UP HERE EVEN THO ITS PART OF: TEST ENVIRONMENT SETUP
    environment_config = config["environment"]
    environment_config["shared_var"]=shared_var
    
    if mode == "train":
        num_envs = training_config["num_envs"]
        environment_config["render_mode"] = None
        environment_config["step_delay"] = 0
        environment_config["early_stop_enabled"] = True
    elif mode == "test":
        num_envs = 1
        environment_config["render_mode"] = "human"
        environment_config["step_delay"] = config["testing"]["step_delay"]
        environment_config["early_stop_enabled"] = False
    elif mode == "record":
        num_envs = 1
        environment_config["render_mode"] = None
        environment_config["step_delay"] = 0
        environment_config["early_stop_enabled"] = False
    
    if num_envs <= 1:
        env = make_MouseArmImitationEnv(0, environment_config)()
    else:
        env = gym.vector.AsyncVectorEnv(
            [
                make_MouseArmImitationEnv(i, environment_config)
                for i in range(num_envs)
            ],
            context="fork",
        )

    env = wrap_env(env)
    device = env.device
    print(f"using device: {device}")

    memory_size=training_config["rollout_length"] #NOTE THIS MUST BE EXACTLY TRUE: memory_size = mini_batches * sequence_length * sequences_per_mini_batch (i think...)
    
    #PPO CFG SETUP
    algorithm_config = config["algorithm"]
    cfg = PPO_CFG()
    cfg.rollouts = memory_size  # memory_size
    cfg.learning_epochs = algorithm_config["n_epochs"]
    cfg.mini_batches = algorithm_config["n_mini_batches"] 
    cfg.discount_factor = algorithm_config["discount_factor"] 
    cfg.gae_lambda = algorithm_config["gae_lambda"]
    cfg.learning_rate = algorithm_config["learning_rate"]
    cfg.learning_rate_scheduler = None#KLAdaptiveLR
    cfg.learning_rate_scheduler_kwargs = {"kl_threshold": None#algorithm_config["lr_scheduler_kl_threshold"]
                                          }
    cfg.grad_norm_clip = algorithm_config["max_grad_norm"]
    cfg.ratio_clip = algorithm_config["clip_range"]
    cfg.value_clip = algorithm_config["clip_range_vf"]
    cfg.entropy_loss_scale = algorithm_config["ent_coef"]
    cfg.value_loss_scale = algorithm_config["vf_coef"]
    cfg.kl_threshold = None#algorithm_config["kl_threshold"]
    cfg.observation_preprocessor = RunningStandardScaler
    cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
    cfg.value_preprocessor = RunningStandardScaler
    cfg.value_preprocessor_kwargs = {"size": 1, "device": device}

    # TensorBoard logging. Percentage checkpoints are saved manually in the
    # training loop, so SKRL's timestep-based checkpointing is disabled.
    if mode == "train":
        cfg.experiment.write_interval = memory_size #training_config["write_interval"] #i think we switch to rollout_length otherwise the write_interval bugs out when writing things
        cfg.experiment.checkpoint_interval = 0
        cfg.experiment.directory = str(model_path)
        cfg.experiment.experiment_name = model_name
    else:
        cfg.experiment.write_interval = 0
        cfg.experiment.checkpoint_interval = 0
        cfg.experiment.directory = ""
        cfg.experiment.experiment_name = ""

    #PPO SETUP
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)
    models = {}
    model_config = config["model"]
    rnn_type = model_config["rnn_type"]
    try:
        layer_type = ReccurentLayerType(rnn_type.lower())
    except ValueError:
        raise ValueError(
            f"Invalid recurrent layer type: {rnn_type}. "
            f"Valid options: {[e.value for e in ReccurentLayerType]}"
        )

    models['policy'] = SharedModel(
        observation_space=env.observation_space, 
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        num_envs=env.num_envs,

        sequence_length=model_config["sequence_length"],
        
        rnn_type=layer_type,
        rnn_hidden_size=model_config["rnn_hidden_size"],
        rnn_layers=model_config["rnn_layers"],
        policy_hidden_size=model_config["policy_hidden_size"],
        policy_layers=model_config["policy_layers"],
        value_hidden_size=model_config["value_hidden_size"],
        value_layers=model_config["value_layers"]
    )
    
    models['value'] = models['policy']

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
    )
    
    # CHECKPOINT SETUP
    checkpoint_dir = full_model_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    newest_checkpoint_path = checkpoint_dir / "newest_agent.pt"
    best_checkpoint_path = checkpoint_dir / "best_agent.pt"

    if mode in ("test", "record"):
        checkpoint_path = best_checkpoint_path
    else:
        checkpoint_path = newest_checkpoint_path

    if not is_new_agent:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"loading checkpoint: {checkpoint_path}")
        agent.load(str(checkpoint_path))

    best_checkpoint_path = checkpoint_dir / "best_agent.pt"
    best_test_reward = -float("inf")

    if metadata_path.is_file():
        with open(metadata_path, "r") as file:
            metadata = yaml.safe_load(file) or {}

        start_timestep = int(metadata.get("timesteps", 0))
        best_test_reward = float(metadata.get("best_test_reward", -float("inf")))


    # HELPERS - TEST
    def wait_for_play_key() -> None:
        """Pause until the user starts the next episode.

        Accepts Enter, Space, or P. Some keyboards do not pass the dedicated
        media Play/Pause key through to the terminal, so Space/P are reliable
        fallbacks.
        """
        print("Episode finished. Press Enter, Space, or P to play the next episode.", flush=True)

        if not sys.stdin.isatty():
            input()
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                key = sys.stdin.read(1).lower()
                if key in ("\r", "\n", " ", "p"):
                    print()
                    return
                if key == "q":
                    print("\nQuit requested.")
                    raise KeyboardInterrupt
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def unwrap_skrl_env(env):
        current = env

        for attr in ("_env", "env", "venv"):
            if hasattr(current, attr):
                current = getattr(current, attr)

        return current


    def pop_reward_stats_from_env(env):
        raw_env = unwrap_skrl_env(env)

        # Vector env case
        if hasattr(raw_env, "call"):
            stats_list = raw_env.call("pop_reward_stats")

            # In your evaluator, this should only be one env, but this keeps it safe.
            if len(stats_list) == 0:
                return {"_count": 0}

            return stats_list[0]

        # Single env case
        if hasattr(raw_env, "pop_reward_stats"):
            return raw_env.pop_reward_stats()

        return {"_count": 0}

    # HELPERS - TRAIN
    def build_checkpoint_percentages(interval_percent: int) -> list[int]:
        """Return checkpoint milestones including both 0% and 100%."""
        if isinstance(interval_percent, bool) or not isinstance(interval_percent, int):
            raise TypeError("training.checkpoint_interval must be a whole-number percentage")

        if not 1 <= interval_percent <= 99:
            raise ValueError("training.checkpoint_interval must be between 1 and 99")

        percentages = list(range(0, 100, interval_percent))
        if percentages[-1] != 100:
            percentages.append(100)

        return percentages

    def save_percentage_checkpoint(agent, checkpoint_dir: Path, percentage: int) -> Path:
        checkpoint_path = checkpoint_dir / f"percent_{percentage:03d}_agent.pt"
        print(f"saving {percentage}% checkpoint: {checkpoint_path}")
        agent.save(str(checkpoint_path))
        return checkpoint_path

    def run_generalization_check(eval_trainer, eval_agent, num_episodes, global_timestep):
        completed_episode_rewards = []

        #eval_env.reset()
        eval_trainer.reset()
        reset_ppo_rnn_states(eval_agent)

        current_episode_rewards = None
        local_timestep = 0

        while len(completed_episode_rewards) < num_episodes:
            _, reward, terminated, truncated, _ = eval_trainer.eval(
                timestep=global_timestep + local_timestep,
                timesteps=sys.maxsize,
            )

            reward = reward.reshape(-1).detach()
            done = terminated.reshape(-1) | truncated.reshape(-1)

            if current_episode_rewards is None:
                current_episode_rewards = reward.clone()
            else:
                current_episode_rewards += reward

            if bool(done.any()):
                completed_episode_rewards.extend(
                    current_episode_rewards.detach().cpu().tolist()
                )
                current_episode_rewards = None

            local_timestep += 1

        return sum(completed_episode_rewards) / len(completed_episode_rewards)

    def copy_agent_checkpoint_modules(dst_agent, src_agent):
        for name, src_module in src_agent.checkpoint_modules.items():
            dst_module = dst_agent.checkpoint_modules.get(name, None)

            if dst_module is None:
                print(f"Skipping {name}: missing on destination agent")
                continue

            if hasattr(src_module, "state_dict") and hasattr(dst_module, "load_state_dict"):
                dst_module.load_state_dict(src_module.state_dict())
                #print(f"Copied {name}")
            else:
                print(f"Skipping {name}: not state_dict/load_state_dict compatible")
    
    def reset_ppo_rnn_states(agent):
        for role in ("policy", "value"):
            for i in range(len(agent._rnn_initial_states[role])):
                agent._rnn_initial_states[role][i].zero_()

            for i in range(len(agent._rnn_final_states[role])):
                agent._rnn_final_states[role][i].zero_()

    def make_eval_env(num_envs, environment_config):
        if num_envs <= 1:
            eval_env = make_MouseArmImitationEnv(0, environment_config)()
        else:
            eval_env = gym.vector.AsyncVectorEnv(
                [
                    make_MouseArmImitationEnv(i, environment_config)
                    for i in range(num_envs)
                ],
                context="fork",
            )

        return wrap_env(eval_env)

    # HELPERS - record
    def get_raw_env(env):
        raw_env = env

        for attr in ("_env", "env"):
            if hasattr(raw_env, attr):
                raw_env = getattr(raw_env, attr)

        return raw_env

    # TRAINER
    if mode == "train":
        print(f'\nnum train files: {len(train_files)}\nnum test files: {len(test_files)}\n\n')
        # TEST ENVIRONMENT SETUP
        shared_var_test = ctx.Value("i", 0)
        shared_var_train_eval = ctx.Value("i", 0)

        eval_interval = training_config.get("eval_interval", 10000)
        eval_num_envs = training_config.get("eval_num_envs", num_envs)

        test_environment_config["kinematic_files"] = test_files
        test_environment_config["render_mode"] = None
        test_environment_config["step_delay"] = 0
        test_environment_config["shared_var"] = shared_var_test
        test_environment_config["early_stop_enabled"] = False

        test_env = make_eval_env(eval_num_envs, test_environment_config)

        cfg_test_trainer = {
            "timesteps": sys.maxsize,
            "headless": True,
            "disable_progressbar": True,
        }
        test_memory = RandomMemory(
            memory_size=memory_size,
            num_envs=test_env.num_envs,
            device=device,
        )

        test_models = {}

        test_models["policy"] = SharedModel(
            observation_space=test_env.observation_space,
            state_space=test_env.state_space,
            action_space=test_env.action_space,
            device=device,
            num_envs=test_env.num_envs,

            sequence_length=model_config["sequence_length"],

            rnn_type=layer_type,
            rnn_hidden_size=model_config["rnn_hidden_size"],
            rnn_layers=model_config["rnn_layers"],
            policy_hidden_size=model_config["policy_hidden_size"],
            policy_layers=model_config["policy_layers"],
            value_hidden_size=model_config["value_hidden_size"],
            value_layers=model_config["value_layers"],
        )

        test_models["value"] = test_models["policy"]

        test_agent = PPO(
            models=test_models,
            memory=test_memory,
            cfg=cfg,
            observation_space=test_env.observation_space,
            state_space=test_env.state_space,
            action_space=test_env.action_space,
            device=device,
        )

        test_trainer = StepTrainer(cfg=cfg_test_trainer, env=test_env, agents=test_agent)

        # TRAIN EVAL ENVIRONMENT SETUP
        train_eval_environment_config = test_environment_config.copy()
        train_eval_environment_config.pop("shared_var", None)
        train_eval_environment_config = copy.deepcopy(train_eval_environment_config)
        train_eval_environment_config["kinematic_files"] = train_files
        train_eval_environment_config["shared_var"] = shared_var_train_eval

        train_eval_env = make_eval_env(eval_num_envs, train_eval_environment_config)

        train_eval_memory = RandomMemory(
            memory_size=memory_size,
            num_envs=train_eval_env.num_envs,
            device=device,
        )

        train_eval_models = {}

        train_eval_models["policy"] = SharedModel(
            observation_space=train_eval_env.observation_space,
            state_space=train_eval_env.state_space,
            action_space=train_eval_env.action_space,
            device=device,
            num_envs=train_eval_env.num_envs,

            sequence_length=model_config["sequence_length"],

            rnn_type=layer_type,
            rnn_hidden_size=model_config["rnn_hidden_size"],
            rnn_layers=model_config["rnn_layers"],
            policy_hidden_size=model_config["policy_hidden_size"],
            policy_layers=model_config["policy_layers"],
            value_hidden_size=model_config["value_hidden_size"],
            value_layers=model_config["value_layers"],
        )

        train_eval_models["value"] = train_eval_models["policy"]

        train_eval_agent = PPO(
            models=train_eval_models,
            memory=train_eval_memory,
            cfg=cfg,
            observation_space=train_eval_env.observation_space,
            state_space=train_eval_env.state_space,
            action_space=train_eval_env.action_space,
            device=device,
        )

        train_eval_trainer = StepTrainer(cfg=cfg_test_trainer, env=train_eval_env, agents=train_eval_agent)

        tb_writer = SummaryWriter(log_dir=str(full_model_path))

        # TRAINING
        run_timesteps = training_config["timesteps"]
        if run_timesteps <= 0:
            raise ValueError("training.timesteps must be greater than 0")

        checkpoint_interval_percent = training_config["checkpoint_interval"]
        checkpoint_percentages = build_checkpoint_percentages(
            checkpoint_interval_percent
        )
        checkpoint_targets = {
            percentage: round(run_timesteps * percentage / 100)
            for percentage in checkpoint_percentages
        }
        saved_checkpoint_percentages = set()

        end_timestep = start_timestep + run_timesteps

        cfg_trainer = {
            "timesteps": end_timestep,
            "headless": True,
            "disable_progressbar": True,
        }

        trainer = StepTrainer(cfg=cfg_trainer, env=env, agents=agent)

        progress_bar = tqdm(
            enumerate(range(start_timestep, end_timestep), start=1),
            total=run_timesteps,
            desc="Training",
            disable=args.disable_progressbar,
        )

        # Save the model state before any training steps in this run.
        save_percentage_checkpoint(agent, checkpoint_dir, 0)
        saved_checkpoint_percentages.add(0)
        
        for i, timestep in progress_bar:
            trainer.train(timestep=timestep, timesteps=end_timestep)

            global_timestep = timestep + 1
            completed_run_timesteps = global_timestep - start_timestep

            # A rounded target can coincide with another target for very short
            # runs, so save every newly reached percentage milestone once.
            for percentage in checkpoint_percentages:
                if percentage in saved_checkpoint_percentages:
                    continue
                if completed_run_timesteps >= checkpoint_targets[percentage]:
                    save_percentage_checkpoint(agent, checkpoint_dir, percentage)
                    saved_checkpoint_percentages.add(percentage)

            if eval_interval > 0 and global_timestep % eval_interval == 0:
                copy_agent_checkpoint_modules(test_agent, agent)
                copy_agent_checkpoint_modules(train_eval_agent, agent)

                mean_test_reward = run_generalization_check(
                    eval_trainer=test_trainer,
                    eval_agent=test_agent,
                    num_episodes=len(test_files),
                    global_timestep=global_timestep,
                )
                
                mean_train_reward = run_generalization_check(
                    eval_trainer=train_eval_trainer,
                    eval_agent=train_eval_agent,
                    num_episodes=len(train_files),
                    global_timestep=global_timestep,
                )
                
                if mean_test_reward > best_test_reward:  
                    best_test_reward = mean_test_reward

                    # print(
                    #     f"new best test reward: {best_test_reward:.6f} "
                    #     f"at timestep {global_timestep}"
                    # )

                    agent.save(str(best_checkpoint_path))

                    tb_writer.add_scalar(
                        "eval/best_test_mean_episode_reward",
                        best_test_reward,
                        global_timestep,
                    )

                tb_writer.add_scalar(
                    "eval/train_mean_episode_reward",
                    mean_train_reward,
                    global_timestep,
                )

                tb_writer.add_scalar(
                    "eval/test_mean_episode_reward",
                    mean_test_reward,
                    global_timestep,
                )

                tb_writer.add_scalar(
                    "eval/generalization_gap",
                    mean_train_reward - mean_test_reward,
                    global_timestep,
                )

                tb_writer.flush()

                progress_bar.set_postfix({
                    "global": f"{global_timestep}/{end_timestep}",
                    "train_eval": f"{mean_train_reward:.3f}",
                    "test_eval": f"{mean_test_reward:.3f}",
                })
            else:
                progress_bar.set_postfix({
                    "global": f"{global_timestep}/{end_timestep}"
                })

        tb_writer.close()
        test_env.close()
        train_eval_env.close()

        # The loop normally saves 100%, but this guarantees it even if the
        # training loop implementation changes later.
        if 100 not in saved_checkpoint_percentages:
            save_percentage_checkpoint(agent, checkpoint_dir, 100)
            saved_checkpoint_percentages.add(100)

        print(f"saving checkpoint: {newest_checkpoint_path}")
        agent.save(str(newest_checkpoint_path))
        with open(metadata_path, "w") as file:
            yaml.safe_dump(
                {
                    "timesteps": end_timestep, 
                    "best_test_reward": best_test_reward
                }, 
                file, 
                sort_keys=False)
        # Percentage checkpoints are intentionally retained after training.
            
    # TEST
    elif mode == "test":
        eval_timesteps = sys.maxsize
        cfg_trainer = {
            "timesteps": eval_timesteps,
            "headless": False,
            "render_interval": 1,
            "disable_progressbar": True,
        }

        trainer = StepTrainer(cfg=cfg_trainer, env=env, agents=agent)

        print("Test mode. Running indefinitely; close the window or press Ctrl+C to quit.")
        print("Each episode pauses on the reset frame before the first step.")

        try:
            timestep = 0
            episode = 1

            while timestep < eval_timesteps:
                env.reset()
                env.render()

                print(f"Episode {episode} ready. Press Enter, Space, or P to play.")
                wait_for_play_key()

                while timestep < eval_timesteps:
                    _, _, terminated, truncated, _ = trainer.eval(
                        timestep=timestep,
                        timesteps=eval_timesteps,
                    )
                    timestep += 1

                    if terminated.any() or truncated.any():
                        episode += 1
                        break

        except KeyboardInterrupt:
            print("Test stopped.")

    # RECORD
    elif mode == "record":
        eval_timesteps = sys.maxsize
        cfg_trainer = {
            "timesteps": eval_timesteps,
            "headless": True,
            "disable_progressbar": True,
        }

        trainer = StepTrainer(cfg=cfg_trainer, env=env, agents=agent)

        recordings_dir = full_model_path / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)

        recording_files_path = recordings_dir / "files.json"

        with open(recording_files_path, "w") as f:
            json.dump(
                {
                    "train_files": [str(path) for path in train_files],
                    "test_files": [str(path) for path in test_files],
                },
                f,
                indent=4,
            )

        print(f"Saved recording file split to {recording_files_path}")
        print("Record mode. Running without visualization or slowmo; press Ctrl+C to quit.")

        try:
            num_record_episodes = len(all_files)

            for _ in range(num_record_episodes):
                raw_env = get_raw_env(env)

                current_file = Path(
                    raw_env.kinematic_files[raw_env.kinematics_index]
                ).expanduser().resolve()

                current_file_stem = current_file.stem

                if current_file in train_file_set:
                    recording_split = "train"
                elif current_file in test_file_set:
                    recording_split = "test"
                else:
                    raise ValueError(
                        f"Current recording file is not in the train or test split: "
                        f"{current_file}"
                    )

                # Clear stale stats before this episode starts
                pop_reward_stats_from_env(env)

                models["policy"].reccurent_layers.start_recording()

                episode_reward_stats_rows = []
                episode_step = 0
                timestep = 0
                while episode_step < eval_timesteps:
                    _, _, terminated, truncated, _ = trainer.eval(
                        timestep=timestep,
                        timesteps=eval_timesteps,
                    )

                    stats = pop_reward_stats_from_env(env)

                    is_terminated = bool(terminated.any())
                    is_truncated = bool(truncated.any())
                    done = is_terminated or is_truncated

                    if int(stats.get("_count", 0)) > 0:
                        row = {
                            "episode_step": episode_step,
                        }

                        for key, value in stats.items():
                            if key == "_count":
                                continue
                            row[key] = float(value)

                        episode_reward_stats_rows.append(row)

                    timestep += 1
                    episode_step += 1

                    if done:
                        break

                hidden_states = models["policy"].reccurent_layers.get_recorded_hidden_states()
                # Expected shape: [T, E, N]
                # T = timesteps
                # E = environments
                # N = neurons

                if hidden_states.ndim != 3:
                    raise ValueError(
                        f"Expected hidden_states shape [T, E, N], got {hidden_states.shape}"
                    )

                if hidden_states.shape[1] != 1:
                    raise ValueError(
                        f"Expected env dim E == 1, got hidden_states shape {hidden_states.shape}"
                    )

                # [T, E, N] -> [T, N]
                hidden_states = hidden_states.squeeze(dim=1)

                # [T, N] -> [N, T]
                hidden_states = hidden_states.transpose(0, 1)

                # Save hidden states as JSON list of lists: [N][T]
                hidden_states_path = (
                    full_model_path
                    / "recordings"
                    / "hidden_states"
                    / recording_split
                    / f"{current_file_stem}.json"
                )
                hidden_states_path.parent.mkdir(parents=True, exist_ok=True)

                with open(hidden_states_path, "w") as f:
                    json.dump(hidden_states.detach().cpu().tolist(), f)

                print(f"Saved hidden states to {hidden_states_path}")

                # Save reward details as CSV, one row per eval timestep
                reward_details_path = (
                    full_model_path
                    / "recordings"
                    / "reward_details"
                    / recording_split
                    / f"{current_file_stem}.csv"
                )
                reward_details_path.parent.mkdir(parents=True, exist_ok=True)

                if len(episode_reward_stats_rows) > 0:
                    fieldnames = list(episode_reward_stats_rows[0].keys())

                    with open(reward_details_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(episode_reward_stats_rows)

                    #print(f"Saved reward details to {reward_details_path}")
                else:
                    print(f"No reward details recorded for episode {current_file_stem}")

                models["policy"].reccurent_layers.stop_recording()

        except KeyboardInterrupt:
            print("Record stopped.")