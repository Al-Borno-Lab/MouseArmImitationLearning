# About
### Development
- **Author:** Dylan Zelkin 
- **Employer:** University of Colorado, Denver
- **Supervisor:** Mazen Al Borno 
- **Lab:** http://cse.ucdenver.edu/~alborno/

### Description
This is an imitation learning project which uses reinforcmenent learning to train deep neural networks to control biomechanical and torque driven models by minimizing the difference between a desired kinematic motion and the actual motion enacted by a network. In this implementation of imitation learning, the network takes, as input, the joint angles and velocities, and outputs the muscle activations or torque activations respectively where each network learns a single unique motion. 

In addition, there is a variation of the environment that can be setup to train a generalized kinematic model which can perform the motion of any desired kinematics on the fly by training on a large number of unique kinematics. It does this by adding a vector to the observation space with values that are the difference between the current position and a set number of future kineamtic positions (see path_steps under the config parameters to enable this functionality).

The mouse forelimb physics models have been adapted from the biomechanical mouse forelimb model from Gilmer et al. [1]; originally an OpenSim model, the torque and muscle models available here are implemented and simulated in MuJoCo [2]. The DRL libray used here is SKRL [3] which offers a varity of reliable learning algorithms; however, the main algorithm in use here is PPO [4]. The model architecture is a shared recurrent backbone split off into dense layered reward and action heads; the recurrent backbones that can be selected are: RNN [5], GRU [6], and LSTM [7]. 

This project was created and tested on linux (specifically ubuntu), and while it might work on other systems, is not guarenteed. 

### Examples
<table>
  <tr>
    <td align="center" width="50%">
      <img src="./readme/torque_agent_sample.gif" width="100%">
      <br>
      <b>Torque Driven Solution</b>
    </td>
    <td align="center" width="50%">
      <img src="./readme/muscle_agent_sample.gif" width="100%">
      <br>
      <b>Muscle Driven Solution</b>
    </td>
  </tr>
</table>

---

# Setup
### Miniconda Installation (if not done so already)
1. Download Miniconda
    ~~~
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ~~~

2. Run the Installer
    ~~~
    bash Miniconda3-latest-Linux-x86_64.sh
    ~~~

### Repo Setup 
1. Install Git (if not done so already)
    ~~~
    sudo apt update && sudo apt install -y git
    ~~~

2. Clone Repo from Github and Open
    ~~~
    git clone https://github.com/Al-Borno-Lab/MouseArmImitationLearning.git
    cd MouseArmImitationLearning
    ~~~

3. Create Python Environment and Activate
    ~~~
    conda env create -f environment.yml
    conda activate MouseArmImitationLearningEnv
    ~~~

4. (Optional) Install Tensorboard for Numerical Results Visualization
    ~~~
    pip install tensorboard
    ~~~

### Huggingface Installations
1. Install Huggingface Hub (if not done so already)
    ~~~
    pip install -U huggingface_hub
    ~~~

2. Download Mujoco Model
    ~~~
    hf download AlBornoLab/MouseArmModel --repo-type dataset --local-dir ./models
    ~~~

3. (OPTIONAL) Download Dataset: MouseArmKinematics 
    ~~~
    hf download AlBornoLab/MouseArmKinematics --repo-type dataset --local-dir ./MouseArmKinematics
    ~~~

4. (OPTIONAL, REQUIRES AUTHORIZATION) Download Dataset: Welle 
    ~~~
    hf download AlBornoLab/Welle --repo-type dataset --local-dir ./Welle
    ~~~

---

# How to Use
### Configuration Parameters
This section details which parameters can be tuned from the imitation learning environment, policy, algorithm, and training and testing scripts.

1. General
    - **name**: Name of the model 
    - **folder**: Folder where the model exists, or will be created at
    - **training**: Whether the mains script is running training or a test visualizer
&nbsp;
2. Environment
    - **model**: Mujoco model file to use
    - **kinematics**: Kinematic data to use (can be a file for single kinematics, or a folder containing files for generalized kinematics)
    - **train_ratio**: The trainig ratio used in splitting the kineamtic data (only matters for generalized kineamtics)
    - **seed**: Random seed used when shuffling and splitting the kinmatic data (only matters for generalized kinematics)
    - **path_steps**: The number of future timesteps to sample kinematics from and include in the observation (0 for single kinematics, >1 for generalized kinematics)
    - **w_bone_diff**: A weight on the average difference between tracked bone locations in the reward function
    - **w_elbow**: A weight on the elbow in the bone average difference
    - **w_paw**: A weight on the paw in the bone average difference
    - **w_effort**: A weight on the effort used by all actuaturos in the reward function
    - **w_qvel**: A weight on the difference between qvel on the joints in the reward function
    - **w_qpos**: A weight on the difference between qpos on the joints in the reward function
    - **w_action:** A weight on the difference between action outputs in the reward function
    - **control_dt**: Total simulation time step size per environment step
    - **n_substeps**: Simulation substeps per environment step (increasing improves simulation stability)
&nbsp;
3. Model
    - **rnn_type**: Recurrent Backbone Type, choices are: lstm, gru, rnn
    - **rnn_hidden_size**: Hidden layer size in rnn
    - **rnn_layer**: Hidden layers in rnn
    - **policy_hidden_size**: Hidden layer size in policy
    - **policy_layers**: Hidden layers in policy
    - **value_hidden_size**: Hidden layer size in value
    - **value_layers**: Hidden layers in value
    - **sequence_length**: The sequence size used during training
&nbsp;
4. Algorithm (There are more advanced terms in the config that are unlisted here, see the SKRL PPO API for more info)
    - **learning_rate**: Learning rate for training
    - **n_steps**: Total number of steps per environment per iteration
    - **batch_size**: Total number of steps per batch
    - **n_epochs**: Training epochs per iteration
&nbsp;
5. Training
    - **timesteps**: Total timesteps across all training  
    - **num_envs**: Number of environments running in parallel
    - **write_interval**: How often to write training stats to tensorboard in timesteps
    - **checkpoint_interval**: How often to save a model checkpoint (incase of failure during training) in timesteps
    - **rollout_length**: Timesteps per training iteration
&nbsp;
6. Testing
    - **slowmo**: Sleep time between frames (visual only), increase for greater slowmo effect

### Running the Programs
1. Resize Mujoco Model to Kinematics (replace model.xml, kinematics_file.csv, and new_model.xml; recommended to save new_model to same folder as model, because it will need to use the same geometry folder)
    ~~~
    python scale_model.py \
        model.xml \
        kinematics_file.csv \
        new_model.xml
    ~~~

2. Train a Model (make sure in config, under general, that training is set to True)
    ~~~
    python main.py
    ~~~

3. Visualize Training Results with Tensorboard
    ~~~
    PORT=$(shuf -i 6006-9000 -n 1); tensorboard --logdir ./logs --port $PORT & sleep 2 && xdg-open http://localhost:$PORT
    ~~~

4. Test a Model's Performance in a Live Viewer (make sure in config, under general, that training is set to False)
    ~~~
    python main.py
    ~~~

---

# References

> [1] Gilmer, Jesse I., Susan K. Coltman, Geraldine Cuenu, John R. Hutchinson, Daniel Huber, Abigail L. Person, and Mazen Al Borno. "A novel biomechanical model of the proximal mouse forelimb predicts muscle activity in optimal control simulations of reaching movements." Journal of neurophysiology 133, no. 4 (2025): 1266-1278.

> [2] Todorov, Emanuel, Tom Erez, and Yuval Tassa. "MuJoCo: A physics engine for model-based control." *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems* (2012): 5026-5033.

> [3] Serrano-Muñoz, A., Chrysostomou, D., Bøgh, S., & Arana-Arexolaleiba, N. (2022). skrl: Modular and Flexible Library for Reinforcement Learning. Journal of Machine Learning Research, 24(254), 1-9.

> [4] Schulman, John, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347* (2017).

> [5] Elman, J. L. (1990). Finding structure in time. Cognitive Science, 14(2), 179–211.

> [6] Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.

> [7] Hochreiter, Sepp, and Jürgen Schmidhuber. "Long Short-Term Memory." *Neural Computation* 9, no. 8 (1997): 1735-1780.
---
