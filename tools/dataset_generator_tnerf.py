from multiprocessing import Process, Manager

from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from rlbench import ObservationConfig, CameraConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task

import os
import pickle
from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
import numpy as np
from copy import deepcopy

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "save_path", "/home/rokas/data/rlbench_data_v2", "Where to save the demos."
)
flags.DEFINE_list(
    "tasks",
    [
        "pick_and_lift",
    ],
    "The tasks to collect. If empty, all tasks are collected.",
)
flags.DEFINE_list("image_size", [224, 224], "The size of the images tp save.")
flags.DEFINE_enum(
    "renderer",
    "opengl3",
    ["opengl", "opengl3"],
    "The renderer to use. opengl does not include shadows, " "but is faster.",
)
flags.DEFINE_integer(
    "processes", 1, "The number of parallel processes during collection."
)
flags.DEFINE_integer(
    "episodes_per_task", 2, "The number of episodes to collect per task."
)
flags.DEFINE_integer(
    "variations", 2, "Number of variations to collect per task. -1 for all."
)
flags.DEFINE_integer(
    "num_additional_cameras", 5, "Number of additional cameras to add (between 0 and 90)."
)


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_demo(demo, example_path):
    cam_names = [cam_name for cam_name in demo._observations[0].obs_rgb.keys()]
    save_rgbs = all(value is not None for value in demo._observations[0].obs_rgb.values())
    save_depths = all(
        value is not None for value in demo._observations[0].obs_depth.values()
    )
    save_masks = all(
        value is not None for value in demo._observations[0].obs_mask.values()
    )
    rgb_paths = {cam_name: os.path.join(example_path, f"{cam_name}_rgb") for cam_name in cam_names}
    depth_paths = {cam_name: os.path.join(example_path, f"{cam_name}_depth") for cam_name in cam_names}
    mask_paths = {cam_name: os.path.join(example_path, f"{cam_name}_mask") for cam_name in cam_names}
    
    if save_rgbs:
        for rgb_path in rgb_paths.values():
            check_and_make(rgb_path)
    if save_depths:
        for depth_path in depth_paths.values():
            check_and_make(depth_path)
    if save_masks:
        for mask_path in mask_paths.values():
            check_and_make(mask_path)

    for i, obs in enumerate(demo):

        if save_rgbs:
            rgbs = {key: Image.fromarray(rgb) for key, rgb in obs.obs_rgb.items()}
            for cam_name, rgb_path in rgb_paths.items():
                rgbs[cam_name].save(os.path.join(rgb_path, IMAGE_FORMAT % i))

        if save_depths:
            depths = {
                key: utils.float_array_to_rgb_image(depth, scale_factor=DEPTH_SCALE)
                for key, depth in obs.obs_depth.items()
            }
            for cam_name, depth_path in depth_paths.items():
                depths[cam_name].save(os.path.join(depth_path, IMAGE_FORMAT % i))
        if save_masks:
            masks = {
                key: Image.fromarray((mask * 255).astype(np.uint8))
                for key, mask in obs.obs_mask.items()
            }
            for cam_name, mask_path in mask_paths.items():
                masks[cam_name].save(os.path.join(mask_path, IMAGE_FORMAT % i))
       
        
        # We save the images separately, so set these to None for pickling.
        obs.obs_rgb = None
        obs.obs_depth = None
        obs.obs_mask = None
        obs.point_cloud = None

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), "wb") as f:
        pickle.dump(demo, f)


def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    cam_config = CameraConfig(
        rgb=True,
        depth=False,
        point_cloud=False,
        mask=False,
        image_size=img_size,
        masks_as_one_channel=True,
        depth_in_meters=False,
    )

    cam_names = [
        "cam_front",
        "cam_over_shoulder_left",
        "cam_over_shoulder_right",
        "cam_overhead",
        "cam_wrist",
    ] + ["cam_additional_%d" % i for i in range(FLAGS.num_additional_cameras)]

    # Make a dictionary with all the camera configurations
    cam_configs = {cam_name: deepcopy(cam_config) for cam_name in cam_names}

    obs_config = ObservationConfig(
        cam_configs=cam_configs,
    )

    if FLAGS.renderer == "opengl":
        for cam in obs_config.cameras.values():
            cam.render_mode = RenderMode.OPENGL

    elif FLAGS.renderer == "opengl3":
        for cam in obs_config.cameras.values():
            cam.render_mode = RenderMode.OPENGL3

    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True,
    )
    rlbench_env.launch()

    task_env = None

    tasks_with_problems = results[i] = ""

    while True:
        # Figure out what task/variation this thread is going to do
        with lock:
            if task_index.value >= num_tasks:
                print("Process", i, "finished")
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations, var_target)
            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print("Process", i, "finished")
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        task_env.set_variation(my_variation_count)
        descriptions, _ = task_env.reset()

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(), VARIATIONS_FOLDER % my_variation_count
        )

        check_and_make(variation_path)

        with open(os.path.join(variation_path, VARIATION_DESCRIPTIONS), "wb") as f:
            pickle.dump(descriptions, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        for ex_idx in range(FLAGS.episodes_per_task):
            print(
                "Process",
                i,
                "// Task:",
                task_env.get_name(),
                "// Variation:",
                my_variation_count,
                ":",
                var_target,
                "// Demo:",
                ex_idx,
            )
            attempts = 3
            while attempts > 0:
                try:
                    # TODO: for now we do the explicit looping.
                    (demo,) = task_env.get_demos(amount=1, live_demos=True)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        "Process %d failed collecting task %s (variation: %d, "
                        "example: %d). Skipping this task/variation.\n%s\n"
                        % (i, task_env.get_name(), my_variation_count, ex_idx, str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                with file_lock:
                    save_demo(demo, episode_path)
                break
            if abort_variation:
                break

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def main(argv):
    task_files = [
        t.replace(".py", "")
        for t in os.listdir(task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError("Task %s not recognised!." % t)
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value("i", 0)
    variation_count = manager.Value("i", 0)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)

    processes = [
        Process(
            target=run,
            args=(i, lock, task_index, variation_count, result_dict, file_lock, tasks),
        )
        for i in range(FLAGS.processes)
    ]
    [t.start() for t in processes]
    [t.join() for t in processes]

    print("Data collection done!")
    for i in range(FLAGS.processes):
        print(result_dict[i])


if __name__ == "__main__":
    app.run(main)
