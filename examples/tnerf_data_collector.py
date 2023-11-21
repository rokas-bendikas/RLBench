import os

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, JointTorque, JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import ReachTarget, CloseDoor, TakeCupOutFromCabinet, PutAllGroceriesInCupboard, PutGroceriesInCupboard, PutItemInDrawer
from tqdm import trange
import json
from json import JSONEncoder
from PIL import Image
import numpy as np

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    


DATASET_DIR = 'collected_data'
TASK_NAME = 'PutItemInDrawer'
NUM_DEMOS = 500

# Check if directory for task exists and if not create it
if not os.path.exists(os.path.join(DATASET_DIR, TASK_NAME)):
    os.makedirs(os.path.join(DATASET_DIR, TASK_NAME))

cam_config = CameraConfig(
    rgb=True,
    depth=False,
    point_cloud=False,
    mask=False,
    image_size=(224, 224),
    masks_as_one_channel=True,
    depth_in_meters=False
)

obs_config = ObservationConfig(
    left_shoulder_camera = cam_config,
    right_shoulder_camera = cam_config,
    overhead_camera = cam_config,
    wrist_camera = cam_config,
    front_camera = cam_config,
    joint_velocities=False,
    joint_positions=True,
    joint_forces=False,
    gripper_open=False,
    gripper_pose=False,
    gripper_matrix=False,
    gripper_joint_positions=False,
    gripper_touch_forces=False,
    wrist_camera_matrix=False,
    record_gripper_closing=False,
    task_low_dim_state=False,
)

action_mode = MoveArmThenGripper(
    arm_action_mode=JointPosition(), gripper_action_mode=Discrete())
env = Environment(
    action_mode, '', obs_config, headless=True, attach_grasped_objects=False)
env.launch()

task = env.get_task(PutItemInDrawer)

for i in trange(NUM_DEMOS):
    demo = task.get_demos(1, live_demos=True)[0]
    print()

    if not os.path.exists(os.path.join(DATASET_DIR, TASK_NAME, f'episode_{i}')):
        os.mkdir(os.path.join(DATASET_DIR, TASK_NAME, f'episode_{i}'), )

    for idx, obs in enumerate(demo._observations):
        # Create step directory
        if not os.path.exists(os.path.join(DATASET_DIR, TASK_NAME, f'episode_{i}', f'step_{idx}')):
            os.mkdir(os.path.join(DATASET_DIR, TASK_NAME, f'episode_{i}', f'step_{idx}'))

        # Save camera parameters in json file
        with open(os.path.join(DATASET_DIR, TASK_NAME, f'episode_{i}', f'step_{idx}', 'camera_params.json'), 'w') as f:
            json.dump(obs.misc, f, cls=NumpyArrayEncoder)

        # Save the robot action in json file
        with open(os.path.join(DATASET_DIR, TASK_NAME, f'episode_{i}', f'step_{idx}', 'robot_action.json'), 'w') as f:
            json.dump(obs.joint_positions, f, cls=NumpyArrayEncoder)

        # Save the rgb of the wrist camera in png file
        wrist_rgb = Image.fromarray(obs.wrist_rgb, 'RGB')
        wrist_rgb.save(os.path.join(DATASET_DIR, TASK_NAME, f'episode_{i}', f'step_{idx}', 'rgb_wrist.png'))

        # Save the rgb of the front camera in png file
        front_rgb = Image.fromarray(obs.front_rgb, 'RGB')
        front_rgb.save(os.path.join(DATASET_DIR, TASK_NAME, f'episode_{i}', f'step_{idx}', 'rgb_front.png'))

        # Save the rgb of the left shoulder camera in png file
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb, 'RGB')
        left_shoulder_rgb.save(os.path.join(DATASET_DIR, TASK_NAME, f'episode_{i}', f'step_{idx}', 'rgb_left_shoulder.png'))

        # Save the rgb of the right shoulder camera in png file
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb, 'RGB')
        right_shoulder_rgb.save(os.path.join(DATASET_DIR, TASK_NAME, f'episode_{i}', f'step_{idx}', 'rgb_right_shoulder.png'))

        # Save the rgb of the overhead camera in png file
        overhead_rgb = Image.fromarray(obs.overhead_rgb, 'RGB')
        overhead_rgb.save(os.path.join(DATASET_DIR, TASK_NAME, f'episode_{i}', f'step_{idx}', 'rgb_overhead.png'))

print('Done')
env.shutdown()
