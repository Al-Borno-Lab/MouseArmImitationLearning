import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from scipy.optimize import least_squares

# -----------------------------
# Helpers
# -----------------------------
def zxy_to_xyz(v):
    """
    Convert a 3-vector from kinematic [z, x, y] ordering into MuJoCo/world [x, y, z] ordering.
    """
    # v is shape (3,) in [z, x, y]
    return np.array([v[2], v[0], v[1]], dtype=np.float64)

def add_sphere_marker(viewer, pos, rgba, radius=0.02):
    """
    Add a sphere marker to viewer.user_scn at world position pos.
    """
    scn = viewer.user_scn

    # Prevent overflow if you add too many
    if scn.ngeom >= scn.maxgeom:
        return

    g = scn.geoms[scn.ngeom]
    scn.ngeom += 1

    # Identity rotation matrix (flattened 3x3)
    mat = np.eye(3).reshape(9)

    # size: for sphere uses (radius, 0, 0)
    size = np.array([radius, 0.0, 0.0], dtype=np.float64)

    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size,
        np.array(pos, dtype=np.float64),
        mat,
        np.array(rgba, dtype=np.float32),
    )

def add_line_marker(viewer, p1, p2, rgba=(1, 1, 1, 1), width=0.002):
    """
    Draw a line segment between world points p1 and p2.
    """
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return

    g = scn.geoms[scn.ngeom]
    scn.ngeom += 1

    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    p2 = np.asarray(p2, dtype=np.float64).reshape(3)

    # Create a "connector" geom from p1 to p2 (LINE / ARROW / etc.)
    mujoco.mjv_connector(
        g,
        mujoco.mjtGeom.mjGEOM_LINE,
        float(width),
        p1,
        p2,
    )

    # Set color/alpha
    g.rgba[:] = np.asarray(rgba, dtype=np.float32)

def make_start_qpos_from_named_hinges(model, current_data, named_angles_rad: dict[str, float]):
    """
    named_angles_rad: {"sh_elv": -0.111..., "sh_extension": 0.174..., ...}
    Returns a full start_qpos vector of shape (model.nq,).
    """
    qpos = current_data.qpos.copy()  # start from current (or model.qpos0.copy())

    for jname, angle in named_angles_rad.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid == -1:
            raise ValueError(f"Joint '{jname}' not found in model.")
        qadr = int(model.jnt_qposadr[jid])
        qpos[qadr] = float(angle)

    return qpos

def mujoco_unstable(data):
    """
    Check whether the current MuJoCo simulation state appears numerically unstable.
    Returns:
        True if qpos, qvel, or ctrl contain non-finite values, or if any velocity
        magnitude is extremely large; otherwise False.
    """
    return (
        not np.isfinite(data.qpos).all()
        or not np.isfinite(data.qvel).all()
        or not np.isfinite(data.ctrl).all()
        or np.abs(data.qvel).max() > 1e6
    )

# -----------------------------
# Environment
# -----------------------------
class MouseArmImitationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, kinematics, model,
                 render_mode: str | None = None,
                 w_bone_diff = 1.0, w_effort = 1.0, w_elbow = 1.0, w_paw = 1.0,
                 w_jitter = 1.0, w_action = 1.0,
                 muscle_color_theme = "grey_red",
                 control_dt = 0.006666666, n_substeps = 16):
        """
        Initialize the mouse arm imitation environment.
        Args:
            kinematics: Path to the CSV file containing tracked kinematic points.
            model: Path to the MuJoCo XML model file.
            render_mode: Rendering mode, typically None or "human".
            w_bone_diff: Weight for target bone-position matching error.
            w_effort: Weight for actuator effort penalty.
            w_elbow: Relative weight for elbow tracking error.
            w_paw: Relative weight for paw/hand tracking error.
            jitter_percent_multiplier: multiplies the error by: (1+(this value * num_inflections)), MUST BE >=0
            muscle_color_theme: Tendon coloring theme for visualization.
            control_dt: Simulated control interval per environment step.
            n_substeps: Number of MuJoCo substeps per environment step.
        """

        super().__init__()
        self.xml_path = model
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Action = controls (actuators)
        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0],
            high=self.model.actuator_ctrlrange[:, 1],
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        # Observation = qpos + qvel (typical)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.model.nq + self.model.nv,),
            dtype=np.float32
        )

        # Viewer handle (human mode)
        self._viewer = None
        self.step_num = 0

        self.humerus_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elv_angle")
        self.radius_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flex")
        self.geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "handm")

        self.kinematic_data = np.loadtxt(
            kinematics,
            delimiter=","
        )
        self.max_steps = self.kinematic_data.shape[0]
        self.time_col = 0
        self.paw_cols = slice(1, 4)
        self.elbow_cols = slice(10, 13)
        self.shoulder_cols = slice(7, 10)

        angles = {
            "elv_angle": 1.27,
            "shoulder_ext": -0.404,
            "shoulder_rot": 0.204,
            "elbow_flex": -0.011,
        }
        self.start_qpos = make_start_qpos_from_named_hinges(self.model, self.data, angles)#self.model.qpos0.copy()          # replace later with your own
        self.start_qvel = np.zeros(self.model.nv)          # replace later with your own

        # Optional: start controls (actuators), shape (model.nu,)
        self.start_ctrl = np.zeros(self.model.nu) if self.model.nu > 0 else None

        # Solve frame 0 IK and replace the hard-coded start pose
        self.kinematic_scale = None
        self.solve_kinematic_scale()
        self.solve_first_frame_start_qpos()

        self.control_dt = control_dt  # T seconds per env.step()
        self.n_substeps = n_substeps
        self.model.opt.timestep=self.control_dt/self.n_substeps
        self.n_substeps = int(np.round(self.control_dt / self.model.opt.timestep))
        self.n_substeps = max(1, self.n_substeps)

        self.weight_bone_diff = w_bone_diff
        self.weight_effort = w_effort
        self.weight_paw = w_paw
        self.weight_elbow = w_elbow

        self.w_jitter = w_jitter
        self.w_action = w_action

        self.muscle_color_theme = muscle_color_theme
        # Build actuator -> tendon mapping once for visualization
        self._tendon_actuator_ids = []
        self._has_tendon_coloring = False

        tendon_trn_type = mujoco.mjtTrn.mjTRN_TENDON

        for act_id in range(self.model.nu):
            if self.model.actuator_trntype[act_id] == tendon_trn_type:
                ten_id = int(self.model.actuator_trnid[act_id, 0])
                if ten_id >= 0:
                    self._tendon_actuator_ids.append((act_id, ten_id))

        self._has_tendon_coloring = len(self._tendon_actuator_ids) > 0


    def reset(self, seed=None, options=None):
        """
        Reset the environment to the solved start pose and return the initial observation.
        """
        super().reset(seed=seed)

        # Apply custom start state
        self.data.qpos[:] = self.start_qpos
        self.data.qvel[:] = self.start_qvel
    
        # Optional: initialize controls
        if self.model.nu > 0:
            self.data.ctrl[:] = self.start_ctrl

        mujoco.mj_forward(self.model, self.data)

        self.step_num = 0

        self.post_phys_step()

        return self._get_obs(), {}

    def post_phys_step(self):
        """
        Update cached world-space model points and aligned kinematic target points
        for the current frame after physics has advanced.

        This computes the current model shoulder/elbow/hand positions and transforms
        the tracked kinematic points into the environment's working coordinate frame.
        """
        self.humerus_pos = self.data.xanchor[self.humerus_id]
        self.radius_pos = self.data.xanchor[self.radius_id]
        self.wrist_marker_world_pos = self.data.site_xpos[self.geom_id]

        self.shoulder_pos = zxy_to_xyz(self.kinematic_data[self.step_num, self.shoulder_cols])
        self.elbow_pos = (zxy_to_xyz(self.kinematic_data[self.step_num, self.elbow_cols]) - self.shoulder_pos)/self.kinematic_scale
        self.paw_pos = (zxy_to_xyz(self.kinematic_data[self.step_num, self.paw_cols]) - self.shoulder_pos)/self.kinematic_scale
        self.shoulder_pos = self.shoulder_pos - self.shoulder_pos

        shoulder_offset = self.shoulder_pos - self.humerus_pos
        self.shoulder_pos-=shoulder_offset
        self.paw_pos-=shoulder_offset
        self.elbow_pos-=shoulder_offset

    def step(self, action):
        """
        Apply an action, advance physics, update targets, and compute reward and termination.
        """
        action = np.asarray(action, dtype=np.float32)
        self.previous_action = self.data.ctrl[:].copy()
        self.data.ctrl[:] = action

        # Step physics 
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_num+=1
        self.post_phys_step()

        self.previous_qvel = self.new_qvel.copy()
        obs = self._get_obs()

        # ----- Reward -----
        shoulder_diff = np.sum((self.humerus_pos - self.shoulder_pos) ** 2)
        elbow_diff = np.sum((self.radius_pos - self.elbow_pos) ** 2)
        paw_diff = np.sum((self.wrist_marker_world_pos - self.paw_pos) ** 2)
        average_diff = (self.weight_elbow*elbow_diff + self.weight_paw*paw_diff)/(self.weight_elbow+self.weight_paw)

        action_error = np.sum((action - self.previous_action) ** 2)

        effort = np.sum(np.abs(self.data.actuator_force))
        
        jitter_error = np.sum((self.new_qvel - self.previous_qvel) ** 2)

        reward = -(self.weight_bone_diff*average_diff + self.weight_effort*effort + self.w_jitter*jitter_error + self.w_action*action_error)

        terminated = (self.step_num >= self.max_steps-1)
        truncated = False

        if self.render_mode == "human":
            self.render()

        # Detect MuJoCo instability
        if self.data.warning.number.any():
            # Clear warnings so it doesn't trigger forever
            for i in range(len(self.data.warning)):
                self.data.warning[i].number = 0
            terminated = True
            reward*=1.2 #this scales the last reward nicely instead of random +c on unknown dt
            print("caught bad state, ending...")

        if mujoco_unstable(self.data):
            terminated=True
            reward*=1.2 #this scales the last reward nicely instead of random +c on unknown dt
            print("caught bad state, ending...")

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """
        Construct the current observation vector.
        """
        self.new_qvel = self.data.qvel.copy()
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def render(self):
        """
        Render the environment in human mode and overlay target/model markers and lines.
        """
        if self.render_mode != "human":
            return

        # Create viewer once
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.opt.sitegroup[:] = 0

        # Make sure world poses are up to date before reading xpos/geom_xpos
        mujoco.mj_forward(self.model, self.data)

        with self._viewer.lock():
            # Set camera once (values copied from test file)
            if not hasattr(self._viewer, "_cam_initialized"):
                self._viewer.cam.lookat[:] = [-0.016869488782851483, 0.011378697078641621, -0.003377446879289613]
                self._viewer.cam.distance = 0.03941385172193466
                self._viewer.cam.azimuth = -124.5
                self._viewer.cam.elevation = -5.25

                self._viewer._cam_initialized = True

            # ---- draw markers ----
            self._viewer.user_scn.ngeom = 0  # clear previous frame markers

            r = 0.001
            add_sphere_marker(self._viewer, self.humerus_pos, rgba=[0, 0, 1, 1], radius=r)
            add_sphere_marker(self._viewer, self.radius_pos,  rgba=[0, 0, 1, 1], radius=r)
            add_sphere_marker(self._viewer, self.wrist_marker_world_pos, rgba=[0, 0, 1, 1], radius=r)

            add_sphere_marker(self._viewer, self.shoulder_pos, rgba=[1, 0, 0, 1], radius=r)
            add_sphere_marker(self._viewer, self.elbow_pos, rgba=[1, 0, 0, 1], radius=r)
            add_sphere_marker(self._viewer, self.paw_pos, rgba=[1, 0, 0, 1], radius=r)

            # Lines 
            w = 1.0
            add_line_marker(self._viewer, self.shoulder_pos, self.humerus_pos, rgba=[1, 1, 1, 1], width=w)
            add_line_marker(self._viewer, self.elbow_pos, self.radius_pos, rgba=[1, 1, 1, 1], width=w)
            add_line_marker(self._viewer, self.paw_pos, self.wrist_marker_world_pos, rgba=[1, 1, 1, 1], width=w)
                
        self._update_tendon_colors_from_ctrl()
        self._viewer.sync()

    def close(self):
        """
        Close the MuJoCo viewer if it has been created and release the handle.
        """
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
    
    def print_muscles(self):
        """
        Print each actuator's name, control value, and resulting actuator force.
        Useful for debugging muscle/actuator activity during simulation.
        """
        print("------------")
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            ctrl = self.data.ctrl[i]
            force = self.data.actuator_force[i]
            print(f"{name:20s} ctrl={ctrl: .4f}  force={force: .4f}")
        print("------")

    def _get_targets_for_frame(self, frame_idx: int, humerus_world_pos: np.ndarray):
        """
        Rebuild the same target points you currently use in post_phys_step(),
        but without mutating env state.
        Returns:
            shoulder_pos, elbow_pos, paw_pos   each shape (3,)
        """
        shoulder_pos = zxy_to_xyz(self.kinematic_data[frame_idx, self.shoulder_cols])
        elbow_pos = (zxy_to_xyz(self.kinematic_data[frame_idx, self.elbow_cols]) - shoulder_pos) / self.kinematic_scale
        paw_pos = (zxy_to_xyz(self.kinematic_data[frame_idx, self.paw_cols]) - shoulder_pos) / self.kinematic_scale
        shoulder_pos = shoulder_pos - shoulder_pos   # zero it, same as your current code

        shoulder_offset = shoulder_pos - humerus_world_pos
        shoulder_pos = shoulder_pos - shoulder_offset
        elbow_pos = elbow_pos - shoulder_offset
        paw_pos = paw_pos - shoulder_offset

        return shoulder_pos, elbow_pos, paw_pos

    def solve_first_frame_start_qpos(self, q0=None, max_nfev=100):
        """
        Solve only the first frame and store the result into self.start_qpos.
        This uses elbow + hand targets, since those are the points your reward
        is already comparing against.
        """
        joint_names = ["elv_angle", "shoulder_ext", "shoulder_rot", "elbow_flex"]

        joint_ids = []
        qpos_ids = []
        lower_bounds = []
        upper_bounds = []

        for jname in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid == -1:
                raise ValueError(f"IK joint '{jname}' not found in model.")

            joint_ids.append(jid)
            qpos_ids.append(int(self.model.jnt_qposadr[jid]))

            if self.model.jnt_limited[jid]:
                lower_bounds.append(float(self.model.jnt_range[jid, 0]))
                upper_bounds.append(float(self.model.jnt_range[jid, 1]))
            else:
                lower_bounds.append(-np.inf)
                upper_bounds.append(np.inf)

        qpos_ids = np.asarray(qpos_ids, dtype=np.int32)
        lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        upper_bounds = np.asarray(upper_bounds, dtype=np.float64)

        if q0 is None:
            q0_full = self.start_qpos.copy()
            q0 = q0_full[qpos_ids].copy()
        else:
            q0 = np.asarray(q0, dtype=np.float64).copy()

        # start from current start pose
        work_qpos = self.start_qpos.copy()
        work_qvel = np.zeros_like(self.data.qvel)

        # small regularization to keep the solution near the initial guess
        q_ref = q0.copy()
        reg_weight = 1e-3

        def residual(q):
            self.data.qpos[:] = work_qpos
            self.data.qpos[qpos_ids] = q
            self.data.qvel[:] = work_qvel
            if self.model.nu > 0:
                self.data.ctrl[:] = 0.0

            mujoco.mj_forward(self.model, self.data)

            humerus_pos = self.data.xanchor[self.humerus_id].copy()
            elbow_pos_model = self.data.xanchor[self.radius_id].copy()
            paw_pos_model = self.data.site_xpos[self.geom_id].copy()

            _, elbow_pos_target, paw_pos_target = self._get_targets_for_frame(
                frame_idx=0,
                humerus_world_pos=humerus_pos,
            )

            elbow_res = elbow_pos_model - elbow_pos_target
            paw_res = paw_pos_model - paw_pos_target
            reg_res = reg_weight * (q - q_ref)

            return np.concatenate([elbow_res, paw_res, reg_res], axis=0)

        result = least_squares(
            residual,
            x0=q0,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=max_nfev,
            verbose=0,
        )

        self.start_qpos[qpos_ids] = result.x
        self.start_qvel[:] = 0.0
        if self.start_ctrl is not None:
            self.start_ctrl[:] = 0.0

        # make data reflect the solved start state immediately
        self.data.qpos[:] = self.start_qpos
        self.data.qvel[:] = self.start_qvel
        if self.model.nu > 0:
            self.data.ctrl[:] = self.start_ctrl
        mujoco.mj_forward(self.model, self.data)
        '''
        print("IK first-frame solve done")
        print("     success:", result.success)
        print("     status:", result.status)
        print("     cost:", result.cost)
        print("     nfev:", result.nfev)
        print("     solved q:", result.x)
        '''
        return result
    
    def _update_tendon_colors_from_ctrl(self):
        """
        Color each tendon directly from its actuator control.
        Assumes one actuator per tendon.
        """
        if not self._has_tendon_coloring:
            return

        for act_id, ten_id in self._tendon_actuator_ids:
            ctrl = float(self.data.ctrl[act_id])

            umin = float(self.model.actuator_ctrlrange[act_id, 0])
            umax = float(self.model.actuator_ctrlrange[act_id, 1])

            if umax <= umin:
                u = 0.0
            else:
                u = np.clip((ctrl - umin) / (umax - umin), 0.0, 1.0)

            if self.muscle_color_theme=="blue_red":
                # blue -> red
                rgba = np.array([u, 0.0, 1.0 - u, 1.0], dtype=np.float32)
            elif self.muscle_color_theme=="grey_red":
                # grey -> red
                rgba = np.array(
                    [
                        0.2 + 0.8 * u,
                        0.2 * (1.0 - u),
                        0.2 * (1.0 - u),
                        1.0
                    ], 
                    dtype=np.float32
                )
            else:
                raise ValueError("muscle_color_theme set wrong")

            try:
                self.model.tendon_rgba[ten_id] = rgba
            except Exception:
                self.model.tendon_rgba[4 * ten_id : 4 * ten_id + 4] = rgba
    
    def _get_model_points_for_qpos(self, qpos: np.ndarray):
        """
        Forward the model at qpos and return the model shoulder/elbow/hand points.
        """
        old_qpos = self.data.qpos.copy()
        old_qvel = self.data.qvel.copy()
        old_ctrl = self.data.ctrl.copy() if self.model.nu > 0 else None

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        humerus_pos = self.data.xanchor[self.humerus_id].copy()
        radius_pos = self.data.xanchor[self.radius_id].copy()
        hand_pos = self.data.site_xpos[self.geom_id].copy()

        self.data.qpos[:] = old_qpos
        self.data.qvel[:] = old_qvel
        if self.model.nu > 0:
            self.data.ctrl[:] = old_ctrl
        mujoco.mj_forward(self.model, self.data)

        return humerus_pos, radius_pos, hand_pos
    
    def solve_kinematic_scale(self, frame_idx: int = 0):
        """
        Solve one scalar kinematic scale so that the two actual arm segments match:
            upper: shoulder -> elbow
            lower: elbow -> paw

        This is separate from IK.
        """
        humerus_pos, radius_pos, hand_pos = self._get_model_points_for_qpos(self.start_qpos)

        shoulder_raw = zxy_to_xyz(self.kinematic_data[frame_idx, self.shoulder_cols])
        elbow_raw = zxy_to_xyz(self.kinematic_data[frame_idx, self.elbow_cols])
        paw_raw = zxy_to_xyz(self.kinematic_data[frame_idx, self.paw_cols])

        # Kinematic segment lengths
        kin_upper_len = np.linalg.norm(elbow_raw - shoulder_raw)
        kin_lower_len = np.linalg.norm(paw_raw - elbow_raw)

        # Model segment lengths
        model_upper_len = np.linalg.norm(radius_pos - humerus_pos)
        model_lower_len = np.linalg.norm(hand_pos - radius_pos)

        eps = 1e-8
        if kin_upper_len <= eps or kin_lower_len <= eps:
            raise ValueError("Kinematic segment lengths were near zero.")
        if model_upper_len <= eps or model_lower_len <= eps:
            raise ValueError("Model segment lengths were near zero.")

        def residual(scale_array):
            scale = float(scale_array[0])

            if scale <= eps:
                return np.array([1e6, 1e6], dtype=np.float64)

            scaled_upper_len = kin_upper_len / scale
            scaled_lower_len = kin_lower_len / scale

            upper_res = scaled_upper_len - model_upper_len
            lower_res = scaled_lower_len - model_lower_len

            return np.array([upper_res, lower_res], dtype=np.float64)

        # starting guess from direct ratios
        s0 = 0.5 * (
            kin_upper_len / model_upper_len +
            kin_lower_len / model_lower_len
        )

        result = least_squares(
            residual,
            x0=np.array([s0], dtype=np.float64),
            bounds=(np.array([eps], dtype=np.float64), np.array([np.inf], dtype=np.float64)),
            max_nfev=100,
            verbose=0,
        )

        self.kinematic_scale = float(result.x[0])

        scaled_upper_len = kin_upper_len / self.kinematic_scale
        scaled_lower_len = kin_lower_len / self.kinematic_scale
        '''
        print("Kinematic scale solve done")
        print("     success:", result.success)
        print("     status:", result.status)
        print("     cost:", result.cost)
        print("     nfev:", result.nfev)
        print("     frame_idx:", frame_idx)
        print("     model_upper_len:", model_upper_len)
        print("     model_lower_len:", model_lower_len)
        print("     scaled_upper_len:", scaled_upper_len)
        print("     scaled_lower_len:", scaled_lower_len)
        print("     kinematic_scale:", self.kinematic_scale)
        '''
        return self.kinematic_scale