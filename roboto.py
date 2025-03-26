import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import imageio

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO

import multiprocessing as mp


class MultiChestKukaEnv(gym.Env):
    """
    A Gymnasium environment for a KUKA robot reaching one of several 'chests' (boxes).
    One chest is chosen as the 'target' (command). The agent sees which chest
    to reach and is rewarded for moving the end-effector near that chosen chest.
    Termination occurs if the agent keeps the end-effector close (< 0.07) to the target
    box for 5 consecutive steps, or runs out of steps.
    A penalty is applied if the robot pushes the box (i.e., if the box moves).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, num_chests=3, use_gui=False):
        super().__init__()

        self.num_chests = num_chests
        self.use_gui = use_gui

        if self.use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=[0.5, 0, -0.65],
            baseOrientation=[0, 0, 0.7071, 0.7071],
        )
        self.kuka_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        self.chest_ids = []
        for _ in range(num_chests):
            cid = p.loadURDF("cube.urdf", globalScaling=0.05)
            self.chest_ids.append(cid)

        self.num_joints = p.getNumJoints(self.kuka_id)
        self.end_effector_index = 6
        
        # Action: move end-effector in x,y,z
        self.action_space = spaces.Box(low=-0.3, high=0.3, shape=(3,), dtype=np.float32)

        # Observation: (ee_x, ee_y, ee_z, chest_x, chest_y, chest_z, cmd_id)
        high = np.array([3]*7, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.max_steps = 100
        self.step_count = 0

        # For camera-based rendering
        self.cam_target_pos = [0.95, -0.2, 0.2]
        self.cam_distance = 2.05
        self.cam_yaw = -50
        self.cam_pitch = -40
        self.cam_width = 480
        self.cam_height = 360

        # Track consecutive steps within 0.07
        self.consecutive_close_steps = 0

        # We'll store the target chest's previous position to detect movement
        self.prev_chest_pos = None

        self.reset()

    def _reset_robot_arm(self):
        """Reset KUKA joints to a neutral pose."""
        init_joint_positions = [0, 0, 0, -1.57, 0, 1.57, 0]
        for j in range(self.num_joints):
            p.resetJointState(
                self.kuka_id, j, init_joint_positions[j % len(init_joint_positions)]
            )

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)

        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane_id = p.loadURDF("plane.urdf")

        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=[1.0, -0.2, 0.0],
            baseOrientation=[0, 0, 0.7071, 0.7071]
        )

        self.kuka_id = p.loadURDF(
            "kuka_iiwa/model_vr_limits.urdf",
            1.4, -0.2, 0.6,
            0, 0, 0, 1
        )
        self.num_joints = p.getNumJoints(self.kuka_id)
        init_joint_positions = [0, 0, 0, 1.5708, 0, -1.0367, 0]
        for j in range(self.num_joints):
            p.resetJointState(self.kuka_id, j, init_joint_positions[j])
            p.setJointMotorControl2(
                self.kuka_id, j, p.POSITION_CONTROL,
                targetPosition=init_joint_positions[j],
                force=500
            )

        table_top_z = 0.65
        table_x_min = 0.70
        table_x_max = 1.0
        table_y_min = -0.6
        table_y_max = 0.2

        self.chest_ids = []
        for _ in range(self.num_chests):
            cid = p.loadURDF("cube.urdf", globalScaling=0.05)
            self.chest_ids.append(cid)

            x = np.random.uniform(table_x_min, table_x_max)
            y = np.random.uniform(table_y_min, table_y_max)
            p.resetBasePositionAndOrientation(cid, [x, y, table_top_z + 0.01], [0, 0, 0, 1])

        # Choose a random target chest
        self.target_idx = np.random.randint(0, self.num_chests)
        target_chest_id = self.chest_ids[self.target_idx]
        p.changeVisualShape(target_chest_id, -1, rgbaColor=[1, 0, 0, 1])

        # Step the simulation a few times
        for _ in range(10):
            p.stepSimulation()

        self.step_count = 0
        self.consecutive_close_steps = 0  # reset close-step counter

        # Store the target chest's initial position so we can track movement
        chest_pos, _ = p.getBasePositionAndOrientation(target_chest_id)
        self.prev_chest_pos = np.array(chest_pos, dtype=np.float32)

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        # End-effector position
        ee_state = p.getLinkState(self.kuka_id, self.end_effector_index)
        ee_pos = np.array(ee_state[0], dtype=np.float32)

        # Target chest's position
        target_chest_id = self.chest_ids[self.target_idx]
        chest_pos, _ = p.getBasePositionAndOrientation(target_chest_id)
        chest_pos = np.array(chest_pos, dtype=np.float32)

        # Command ID
        cmd_id = float(self.target_idx)

        obs = np.concatenate([ee_pos, chest_pos, [cmd_id]]).astype(np.float32)
        return obs

    def step(self, action):
        self.step_count += 1

        # Current end-effector position
        ee_state = p.getLinkState(self.kuka_id, self.end_effector_index)
        ee_pos = np.array(ee_state[0])

        # Desired new EE position
        new_ee_pos = ee_pos + action

        # IK to move the end-effector
        ee_orn = p.getQuaternionFromEuler([0, -np.pi, 0])
        joint_poses = p.calculateInverseKinematics(
            self.kuka_id, self.end_effector_index, new_ee_pos, ee_orn
        )
        for j in range(self.num_joints):
            p.setJointMotorControl2(
                self.kuka_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[j],
                force=500
            )

        p.stepSimulation()

        obs = self._get_obs()
        ee_pos = obs[:3]
        chest_pos = obs[3:6]

        # Distance to target
        dist = np.linalg.norm(ee_pos - chest_pos)

        # ----------------
        # REWARD LOGIC
        # ----------------
        # Base negative reward = -distance
        reward = -dist

        # Penalize movement of the target chest
        chest_move_dist = np.linalg.norm(chest_pos - self.prev_chest_pos)
        if chest_move_dist > 1e-6:  # if it moved at all
            # E.g., penalize proportionally
            reward -= chest_move_dist * 10

        # Update previous chest position
        self.prev_chest_pos = chest_pos.copy()

        # ----------------
        # TERMINATION LOGIC
        # ----------------
        terminated = False
        truncated = False

        # If we stayed within 0.07 for 5 consecutive steps => success
        if dist < 0.06:
            self.consecutive_close_steps += 1
        else:
            self.consecutive_close_steps = 0

        if self.consecutive_close_steps >= 5:
            terminated = True
            # Add a "success" bonus
            reward += 20
            print(f"Success! End-effector close for 5+ consecutive steps. Distance={dist:.3f}")

        # If we reached max steps, we truncate
        if self.step_count >= self.max_steps:
            truncated = True

        info = {"is_success": terminated}  # or track success differently if you prefer

        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.cam_target_pos,
                distance=self.cam_distance,
                yaw=self.cam_yaw,
                pitch=self.cam_pitch,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(self.cam_width)/self.cam_height,
                nearVal=0.1,
                farVal=100.0
            )
            img = p.getCameraImage(
                width=self.cam_width,
                height=self.cam_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix
            )
            rgb_array = np.reshape(img[2], (self.cam_height, self.cam_width, 4))[:, :, :3]
            return rgb_array
        else:
            return None

    def close(self):
        p.disconnect()


# ------------------------------------------------------------------------------
# Below is the rest of your code (callbacks, env-creation, main, etc.) unchanged
# but you can simply replace the MultiChestKukaEnv definition with the one above.
# ------------------------------------------------------------------------------
class KukaEvalCallback(BaseCallback):
    """
    A callback that runs evaluation episodes every `eval_freq` steps
    and saves a video in the local directory.
    Also logs mean reward and success rate to TensorBoard.
    Uses the new Gymnasium 5-item step signature.
    """
    def __init__(self, eval_env, n_eval_episodes=5, eval_freq=20000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.best_success_rate = -np.inf

    def _init_callback(self):
        if self.logger is None:
            raise ValueError("No logger set for KukaEvalCallback.")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            episode_rewards = []
            episode_successes = []
            frames = []

            for _ in range(self.n_eval_episodes):
                obs, _info = self.eval_env.reset()
                terminated = False
                truncated = False
                ep_reward = 0.0
                success = 0

                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_reward += reward

                    frame = self.eval_env.render(mode="rgb_array")
                    if frame is not None:
                        frames.append(frame)

                    if info.get("is_success", False):
                        success = 1

                episode_rewards.append(ep_reward)
                episode_successes.append(success)

            mean_reward = np.mean(episode_rewards)
            success_rate = np.mean(episode_successes)

            if self.verbose > 0:
                print(f"\n[EvalCallback] Step={self.n_calls} | "
                      f"MeanReward={mean_reward:.3f} | SuccessRate={success_rate*100:.1f}%")

            self.logger.record("eval/mean_reward", mean_reward, self.n_calls)
            self.logger.record("eval/success_rate", success_rate, self.n_calls)
            self.logger.dump(self.n_calls)

            video_path = f"evaluation_{self.n_calls}.mp4"
            with imageio.get_writer(video_path, fps=30) as writer:
                for frame in frames:
                    writer.append_data(frame)
            if self.verbose > 0:
                print(f"[EvalCallback] Saved evaluation video: {video_path}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save("best_model_by_reward")
                if self.verbose > 0:
                    print("[EvalCallback] New best mean reward => Saved model as best_model_by_reward")

            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                self.model.save("best_model_by_success")
                if self.verbose > 0:
                    print("[EvalCallback] New best success rate => Saved model as best_model_by_success")

        return True


from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env_fn(rank, num_chests=3, use_gui=False):
    """
    Factory to create a single environment instance, wrapped with a Monitor.
    `rank` is just an ID for logging or debugging.
    """
    def _init():
        env = MultiChestKukaEnv(num_chests=num_chests, use_gui=use_gui)
        env = Monitor(env)
        return env
    return _init

def create_parallel_envs(n_envs=4, num_chests=3, use_gui=False):
    """
    Creates a SubprocVecEnv with n_envs parallel MultiChestKukaEnv environments.
    """
    env_fns = [make_env_fn(i, num_chests, use_gui) for i in range(n_envs)]
    return SubprocVecEnv(env_fns)


def main():
    n_envs = 16
    num_chests = 3
    train_env = create_parallel_envs(n_envs=n_envs, num_chests=num_chests, use_gui=False)

    eval_env = MultiChestKukaEnv(num_chests=num_chests, use_gui=False)
    eval_callback = KukaEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=5,
        eval_freq=20000,
        verbose=1
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./runs",
        n_steps=int(2048/n_envs),
        batch_size=64,
    )

    model.learn(
        total_timesteps=10000000,
        callback=eval_callback
    )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
