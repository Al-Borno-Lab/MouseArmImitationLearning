import multiprocessing as mp
import sys
import yaml
import termios
import tty

from pathlib import Path
from tqdm import tqdm

import gymnasium as gym

from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO_CFG
from skrl.agents.torch.ppo import PPO_RNN as PPO
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import StepTrainer

from imitation_env import make_MouseArmImitationEnv
from data_helper import setup_files
from models import SharedModel, ReccurentLayerType


if __name__ == "__main__":
    config_file = "config.yml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    #CHECK FOR PREVIOUS AGENT
    is_new_agent: bool
    general_config = config["general"]
    model_name = general_config["name"]
    model_folder = general_config["folder"]
    model_path = Path(model_folder)
    full_model_path = model_path / model_name
    saved_config_path = full_model_path / config_file
    if full_model_path.is_dir():
        print("found existing agent...")
        is_new_agent = False
        with open(saved_config_path, "r") as file:
            saved_config = yaml.safe_load(file)
        saved_config["training"]=config["training"]
        saved_config["testing"]=config["testing"]
        config=saved_config
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
    train_files, test_files = setup_files(path=config["environment"]["kinematics"],
                                            train_ratio=config["environment"]["train_ratio"],
                                            seed=config["environment"]["seed"])
    config['environment']['kinematic_files'] = train_files

    #ENVIRONMENT SETUP
    ctx = mp.get_context("forkserver")
    shared_var = ctx.Value("i", 0)

    training_config = config["training"]
    environment_config = config["environment"]
    environment_config["shared_var"]=shared_var
    
    if general_config["training"]:
        num_envs = training_config["num_envs"]
        environment_config["render_mode"] = None
        environment_config["step_delay"] = 0
    else:
        num_envs = 1
        environment_config["render_mode"] = "human"
        environment_config["step_delay"] = config["testing"]["step_delay"]
    
    if num_envs <= 1:
        env = make_MouseArmImitationEnv(0, environment_config)()
    else:
        env = gym.vector.AsyncVectorEnv(
            [
                make_MouseArmImitationEnv(i, environment_config)
                for i in range(num_envs)
            ],
            context="forkserver",
        )

    env = wrap_env(env)
    device = env.device

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
    cfg.learning_rate_scheduler = KLAdaptiveLR
    cfg.learning_rate_scheduler_kwargs = {"kl_threshold": algorithm_config["lr_scheduler_kl_threshold"]}
    cfg.grad_norm_clip = algorithm_config["max_grad_norm"]
    cfg.ratio_clip = algorithm_config["clip_range"]
    cfg.value_clip = algorithm_config["clip_range_vf"]
    cfg.entropy_loss_scale = algorithm_config["ent_coef"]
    cfg.value_loss_scale = algorithm_config["vf_coef"]
    cfg.kl_threshold = algorithm_config["kl_threshold"]
    cfg.observation_preprocessor = RunningStandardScaler
    cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
    cfg.value_preprocessor = RunningStandardScaler
    cfg.value_preprocessor_kwargs = {"size": 1, "device": device}

        # logging to TensorBoard and write checkpoints (in timesteps)
    if general_config["training"]:
        cfg.experiment.write_interval = training_config["write_interval"]
        cfg.experiment.checkpoint_interval = training_config["checkpoint_interval"]
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

    checkpoint_path = checkpoint_dir / "newest_agent.pt"

    if not is_new_agent:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"loading checkpoint: {checkpoint_path}")
        agent.load(str(checkpoint_path))


    # HELPER
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

    # TRAINER / EVALUATOR
    if general_config['training']:
        run_timesteps = training_config["timesteps"]
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
        )
        
        for i, timestep in progress_bar:
            trainer.train(timestep=timestep, timesteps=end_timestep)
            progress_bar.set_postfix({
                "global": f"{timestep + 1}/{end_timestep}"
            })
        
        print(f"saving checkpoint: {checkpoint_path}")
        agent.save(str(checkpoint_path))
        with open(metadata_path, "w") as file:
            yaml.safe_dump({"timesteps": end_timestep}, file, sort_keys=False)
        for checkpoint_file in checkpoint_dir.glob("agent_*.pt"):
            checkpoint_file.unlink()
    else:
        eval_timesteps = sys.maxsize
        cfg_trainer = {
            "timesteps": eval_timesteps,
            "headless": False,
            "render_interval": 1,
            "disable_progressbar": True,
        }

        trainer = StepTrainer(cfg=cfg_trainer, env=env, agents=agent)

        print("Evaluation mode. Running indefinitely; close the window or press Ctrl+C to quit.")
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
            print("Evaluation stopped.")