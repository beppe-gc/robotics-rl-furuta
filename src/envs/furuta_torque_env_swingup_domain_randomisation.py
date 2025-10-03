import gymnasium as gym
from gymnasium import spaces
import math
import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
from pinocchio.visualize import MeshcatVisualizer as PMV
import time

class FurutaPendulumSimulatorRandom:
    def __init__(self, urdf_model_path, parameters_model, render=False):
        self.urdf_model_path = urdf_model_path
        self.render = render
        self.set_model(parameters_model)

    def set_model(self, parameters_model):
        # Load and build model
        model, collision_model, visual_model = pin.buildModelsFromUrdf(self.urdf_model_path)

        # Retrieve joint IDs
        joint1_id = model.getJointId("joint0")
        joint2_id = model.getJointId("joint1")

        if joint1_id == 0 or joint2_id == 0:
            raise ValueError("Joint names 'joint0' or 'joint1' not found in the URDF.")

        # Modify inertias
        body1_id, body2_id = joint1_id, joint2_id

        inertia_1 = model.inertias[body1_id]
        inertia_1.mass = parameters_model["m1"]
        inertia_1.inertia = np.diag([0.0, 0.0, parameters_model["J1"]])
        inertia_1.lever = np.array([0.0, 0.0, parameters_model["l1"]])
        model.inertias[body1_id] = inertia_1

        inertia_2 = model.inertias[body2_id]
        inertia_2.mass = parameters_model["m2"]
        inertia_2.inertia = np.diag([0.0, 0.0, parameters_model["J2"]])
        inertia_2.lever = np.array([0.0, 0.0, parameters_model["l2"]])
        model.inertias[body2_id] = inertia_2

        model.damping[model.joints[joint1_id].idx_v] = parameters_model["b1"]
        model.damping[model.joints[joint2_id].idx_v] = parameters_model["b2"]
      

        self.model = model
        cmodel = cpin.Model(model)
        self.f_state_transition = self.create_f_state_transition_from_model(cmodel)
        self.x = np.zeros(4)
        self.x_hist = [self.x.copy()]

    def create_f_state_transition_from_model(self, cmodel):
        q = ca.SX.sym("q", cmodel.nq)
        dq = ca.SX.sym("dq", cmodel.nv)
        x = ca.vertcat(q, dq)
        u = ca.SX.sym("u", 1)
        tau = ca.vertcat(u, ca.SX.zeros(cmodel.nv - 1))

        cdata = cmodel.createData()
        M = cpin.crba(cmodel, cdata, q)
        C = cpin.computeCoriolisMatrix(cmodel, cdata, q, dq)
        g = cpin.computeGeneralizedGravity(cmodel, cdata, q)
        qdd = ca.solve(M, tau - C @ dq - g)
        dx = ca.vertcat(dq, qdd)

        dt = ca.SX.sym("dt")
        x_next = self.integrate_RK4(x, u, dx, dt)
        return ca.Function("f_state_transition", [x, u, dt], [x_next])

    def integrate_RK4(self, x_expr, u_expr, xdot_expr, dt, N_steps=1):
        h = dt / N_steps
        x_end = x_expr
        xdot_fun = ca.Function('xdot', [x_expr, u_expr], [xdot_expr])
        for _ in range(N_steps):
            k1 = xdot_fun(x_end, u_expr)
            k2 = xdot_fun(x_end + 0.5 * h * k1, u_expr)
            k3 = xdot_fun(x_end + 0.5 * h * k2, u_expr)
            k4 = xdot_fun(x_end + h * k3, u_expr)
            x_end += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_end

    def step(self, u, dt):
        self.x = self.f_state_transition(self.x, u, dt).full().flatten()
        self.x_hist.append(self.x.copy())

class FurutaPendulumSwingupRandomEnv(gym.Env):
    def __init__(self, urdf_model_path, parameters_model, render=False, swingup=True):
        self.urdf_model_path = urdf_model_path
        self.base_parameters_model = parameters_model
        self.render = render
        self.swingup = swingup

        self._randomization_ranges = {
            "m1": (0.8, 1.2),
            "m2": (0.8, 1.2),
            "l1": (0.9, 1.1),
            "l2": (0.9, 1.1),
            "L1": (0.8, 1.2),
            "L2": (0.8, 1.2),
            "J1": (0.8, 1.2),
            "J2": (0.8, 1.2),
            "b1": (0.7, 1.3),
            "b2": (0.7, 1.3),
        }

        self.dt = 0.01
        self.time_limit = 5000
        self.time_step = 0

        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)

        self.reset()

    def _sample_random_parameters(self):
        randomized = self.base_parameters_model.copy()
        for k, (low, high) in self._randomization_ranges.items():
            if k in randomized:
                randomized[k] *= np.random.uniform(low, high)

        # Ensure essential limit parameters exist and are valid
        required_keys = [
            "max_velocity_joint0", "max_velocity_joint1",
            "max_angle_joint0", "max_angle_joint1",
            "max_torque_joint0"
        ]
        for key in required_keys:
            if key not in randomized or randomized[key] < 1e-6 or not np.isfinite(randomized[key]):
                raise ValueError(f"Invalid or missing value for parameter '{key}': {randomized.get(key)}")

        return randomized

    def reset(self, seed=None, options=None):
        randomized_parameters = self._sample_random_parameters()
        self.pendulum_sim = FurutaPendulumSimulatorRandom(
            urdf_model_path=self.urdf_model_path,
            parameters_model=randomized_parameters,
            render=self.render
        )

        qpos = np.array([0.0, 0.0]) if self.swingup else np.array([0.0, np.pi])
        qvel = np.zeros(2)
        self.pendulum_sim.x = np.concatenate([qpos, qvel])
        self.qpos, self.qvel = qpos, qvel

        self._max_velocity_joint0 = randomized_parameters["max_velocity_joint0"]
        self._max_velocity_joint1 = randomized_parameters["max_velocity_joint1"]
        self._max_angle_joint0 = randomized_parameters["max_angle_joint0"]
        self._max_angle_joint1 = randomized_parameters["max_angle_joint1"]
        self._max_torque_joint0 = randomized_parameters["max_torque_joint0"]

        self.time_step = 0
        return self._get_obs(), {}

    def step(self, action):
        u = action[0] * self._max_torque_joint0
        self.pendulum_sim.step(u, self.dt)
        self.qpos = self.pendulum_sim.x[:2]
        self.qvel = self.pendulum_sim.x[2:]
        obs = self._get_obs()
        reward = self.calculate_reward(obs, action)
        if (np.abs(self.qvel[0]) > self._max_velocity_joint0 or 
            np.abs(self.qvel[1]) > self._max_velocity_joint1 or 
            np.abs(self.qpos[0]) > self._max_angle_joint0):
            terminated = True
            reward = -100
        else:
            terminated = False
        truncated = self.time_step >= self.time_limit
        self.time_step += 1
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        obs = np.array([
            np.sin(self.qpos[0]), np.cos(self.qpos[0]),
            np.sin(self.qpos[1]), np.cos(self.qpos[1]),
            self.qvel[0] / self._max_velocity_joint0,
            self.qvel[1] / self._max_velocity_joint1,
            self.qpos[0] / self._max_angle_joint0,
        ])

        if not np.all(np.isfinite(obs)):
            print("NaN or Inf in observation:", obs)
            print("qpos:", self.qpos, "qvel:", self.qvel)
            raise ValueError("Invalid observation due to parameter values")

        return obs

    def calculate_reward(self, obs: np.array, a: np.array):

        if not self.swingup:
            # Reward function parameters
            theta1_weight = 0.0
            theta2_weight = 0.0
            dtheta1_weight = 5.0
            dtheta2_weight = 5.0

            self._desired_obs_values = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0]
            self._obs_weights = [
                theta1_weight,
                theta1_weight,
                theta2_weight, 
                theta2_weight,
                dtheta1_weight,
                dtheta2_weight,
                theta1_weight,
            ]
            observation_reward = np.sum(
                [
                    -weight * np.power((desired_value - observation_value), 2)
                    for (observation_value, desired_value, weight) in zip(
                        obs, self._desired_obs_values, self._obs_weights
                    )
                ]
            )

            self._action_weight = 1.0
            action_reward = -self._action_weight * np.power(a[0], 2)

            reward = observation_reward + action_reward

            reward_normalized = reward / 4000

            return reward_normalized
        
        else:
            # Reward weights
            theta1_weight = 0.05    # mild penalty for base rotation
            theta2_weight = 20.0    # Increase emphasis on being upright
            dtheta1_weight = 2.0    # Penalize rotating base if unnecessary
            dtheta2_weight = 10.0   # Strongly penalize pendulum motion near upright   
            torque_penalty_weight = 0.05  # Slightly discourage over-control

            self._desired_obs_values = [
                0.0,     # sin(theta1)
                0.0,     # cos(theta1)
                0.0,     # sin(theta2)
                -1.0,    # cos(theta2)
                0.0,     # dtheta1
                0.0,     # dtheta2
                0.0,     # normalized theta1
            ]

            self._obs_weights = [
                theta1_weight,
                theta1_weight,
                theta2_weight, 
                theta2_weight,
                dtheta1_weight,
                dtheta2_weight,
                theta1_weight,
            ]

            observation_reward = np.sum([
                -weight * np.power((desired_value - observation_value), 2)
                for (observation_value, desired_value, weight) in zip(
                    obs, self._desired_obs_values, self._obs_weights
                )
            ])

            action_reward = -torque_penalty_weight * np.power(a[0], 2)

            reward = observation_reward + action_reward
            reward_normalized = reward / 4000  # Normalize to keep in useful range

            # Bonus for staying very close to upright & still
            if (
                np.abs(obs[2]) < 0.1 and              # sin(θ₂) ≈ 0
                np.abs(obs[3] + 1.0) < 0.1 and        # cos(θ₂) ≈ -1
                np.abs(obs[5]) < 0.1 and              # dθ₂ small
                np.abs(obs[4]) < 0.1                  # dθ₁ small
            ):
                reward_normalized += 0.5  # Bonus


            return reward_normalized

