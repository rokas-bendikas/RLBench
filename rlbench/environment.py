import importlib
from os.path import exists, dirname, abspath, join
from typing import Type, List

from pyrep import PyRep
from pyrep.objects import VisionSensor
from pyrep.robots.arms.panda import Panda

from rlbench import utils
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.const import *
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene, AdditionalViewScene
from rlbench.backend.task import Task
from rlbench.const import SUPPORTED_ROBOTS
from rlbench.demo import Demo
from rlbench.observation_config import (
    ObservationConfig,
    AdditionalViewObservationConfig,
)
from rlbench.sim2real.domain_randomization import (
    RandomizeEvery,
    VisualRandomizationConfig,
    DynamicsRandomizationConfig,
)
from rlbench.sim2real.domain_randomization_scene import DomainRandomizationScene
from rlbench.task_environment import TaskEnvironment

DIR_PATH = dirname(abspath(__file__))


class Environment(object):
    """Each environment has a scene."""

    def __init__(
        self,
        action_mode: ActionMode,
        dataset_root: str = "",
        obs_config: ObservationConfig = ObservationConfig(),
        headless: bool = False,
        static_positions: bool = False,
        robot_setup: str = "panda",
        randomize_every: RandomizeEvery = None,
        frequency: int = 1,
        visual_randomization_config: VisualRandomizationConfig = None,
        dynamics_randomization_config: DynamicsRandomizationConfig = None,
        attach_grasped_objects: bool = True,
        shaped_rewards: bool = False,
        add_cam_names: dict = {},
        verbose: bool = False,
        paired_randomize: bool = False,
        default_texture: str = "default",
        input_texture: str = "random",
        remove_background: bool = False,
        randomize_cameras: bool = False,
    ):
        self._is_standard_observation = not isinstance(
            obs_config, AdditionalViewObservationConfig
        )
        self._dataset_root = dataset_root
        self._action_mode = action_mode
        self._obs_config = obs_config
        self._headless = headless
        self._static_positions = static_positions
        self._robot_setup = robot_setup.lower()

        self._randomize_every = randomize_every
        self._paired_randomize = paired_randomize
        self._frequency = frequency
        self._visual_randomization_config = visual_randomization_config
        self._dynamics_randomization_config = dynamics_randomization_config
        self._attach_grasped_objects = attach_grasped_objects
        self._shaped_rewards = shaped_rewards
        self._default_texture = default_texture
        self._input_texture = input_texture

        if robot_setup not in SUPPORTED_ROBOTS.keys():
            raise ValueError(
                "robot_configuration must be one of %s" % str(SUPPORTED_ROBOTS.keys())
            )

        # if (
        #     randomize_every is not None
        #     and visual_randomization_config is None
        #     and dynamics_randomization_config is None
        # ):
        #     raise ValueError(
        #         "If domain randomization is enabled, must supply either "
        #         "visual_randomization_config or dynamics_randomization_config"
        #     )

        self._check_dataset_structure()
        self._pyrep = None
        self._robot = None
        self._scene = None
        self._prev_task = None
        self._add_cam_names = add_cam_names
        self._verbose = verbose
        self._remove_background = remove_background
        self._radomize_cameras = randomize_cameras

    def _check_dataset_structure(self):
        if len(self._dataset_root) > 0 and not exists(self._dataset_root):
            raise RuntimeError("Data set root does not exists: %s" % self._dataset_root)

    def _string_to_task(self, task_name: str):
        task_name = task_name.replace(".py", "")
        try:
            class_name = "".join([w[0].upper() + w[1:] for w in task_name.split("_")])
            mod = importlib.import_module("rlbench.tasks.%s" % task_name)
        except Exception as e:
            raise RuntimeError(
                "Tried to interpret %s as a task, but failed. Only valid tasks "
                "should belong in the tasks/ folder" % task_name
            ) from e
        return getattr(mod, class_name)

    def launch(self):
        if self._pyrep is not None:
            raise RuntimeError("Already called launch!")
        self._pyrep = PyRep()
        self._pyrep.launch(join(DIR_PATH, TTT_FILE), headless=self._headless)

        arm_class, gripper_class, _ = SUPPORTED_ROBOTS[self._robot_setup]

        # We assume the panda is already loaded in the scene.
        if self._robot_setup != "panda":
            # Remove the panda from the scene
            panda_arm = Panda()
            panda_pos = panda_arm.get_position()
            panda_arm.remove()
            arm_path = join(DIR_PATH, "robot_ttms", self._robot_setup + ".ttm")
            self._pyrep.import_model(arm_path)
            arm, gripper = arm_class(), gripper_class()
            arm.set_position(panda_pos)
        else:
            arm, gripper = arm_class(), gripper_class()

        self._robot = Robot(arm, gripper)
        if self._is_standard_observation:
            scene_cls = Scene
        else:
            scene_cls = AdditionalViewScene
        if self._randomize_every is None:
            self._scene = scene_cls(
                self._pyrep,
                self._robot,
                self._obs_config,
                self._robot_setup,
                self._add_cam_names,
                self._verbose,
                remove_background=self._remove_background,
                randomize_cameras=self._radomize_cameras,
            )
        else:
            self._scene = scene_cls(
                self._pyrep,
                self._robot,
                self._obs_config,
                self._robot_setup,
                self._add_cam_names,
                self._verbose,
                default_texture=self._default_texture,
                input_texture=self._input_texture,
                randomize_every=self._randomize_every,
                remove_background=self._remove_background,
            )

        self._action_mode.arm_action_mode.set_control_mode(self._robot)

    def change_cams(self, add_cam_names):
        self._add_cam_names = add_cam_names
        if self._randomize_every is None:
            if self._is_standard_observation:
                scene_cls = Scene
            else:
                scene_cls = AdditionalViewScene

            self._scene = scene_cls(
                self._pyrep,
                self._robot,
                self._obs_config,
                self._robot_setup,
                self._add_cam_names,
                self._radius,
                self._phi,
                self._height,
            )
        else:
            raise NotImplementedError

    def shutdown(self):
        if self._pyrep is not None:
            self._pyrep.shutdown()
        self._pyrep = None

    def get_task(self, task_class: Type[Task]) -> TaskEnvironment:

        # If user hasn't called launch, implicitly call it.
        if self._pyrep is None:
            self.launch()

        self._scene.unload()
        task = task_class(self._pyrep, self._robot)
        self._prev_task = task
        return TaskEnvironment(
            self._pyrep,
            self._robot,
            self._scene,
            task,
            self._action_mode,
            self._dataset_root,
            self._obs_config,
            self._static_positions,
            self._attach_grasped_objects,
            self._shaped_rewards,
            self._default_texture,
        )

    @property
    def action_shape(self):
        return (self._action_mode.action_shape(self._scene),)

    def get_demos(
        self,
        task_name: str,
        amount: int,
        variation_number=0,
        image_paths=False,
        random_selection: bool = True,
        from_episode_number: int = 0,
    ) -> List[Demo]:
        if self._dataset_root is None or len(self._dataset_root) == 0:
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided."
            )
        demos = utils.get_stored_demos(
            amount,
            image_paths,
            self._dataset_root,
            variation_number,
            task_name,
            self._obs_config,
            random_selection,
            from_episode_number,
        )
        return demos

    def get_scene_data(self) -> dict:
        """Get the data of various scene/camera information.

        This temporarily starts the simulator in headless mode.

        :return: A dictionary containing scene data.
        """

        def _get_cam_info(cam: VisionSensor):
            if not cam.still_exists():
                return None
            intrinsics = cam.get_intrinsic_matrix()
            return dict(
                intrinsics=intrinsics,
                near_plane=cam.get_near_clipping_plane(),
                far_plane=cam.get_far_clipping_plane(),
                extrinsics=cam.get_matrix(),
            )

        headless = self._headless
        self._headless = True
        self.launch()
        d = dict(
            left_shoulder_camera=_get_cam_info(self._scene._cam_over_shoulder_left),
            right_shoulder_camera=_get_cam_info(self._scene._cam_over_shoulder_right),
            front_camera=_get_cam_info(self._scene._cam_front),
            wrist_camera=_get_cam_info(self._scene._cam_wrist),
            overhead_camera=_get_cam_info(self._scene._cam_overhead),
        )
        self.shutdown()
        self._headless = headless
        return d
