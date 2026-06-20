import os
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path

benchmark_dict = benchmark.get_benchmark_dict()

task_suite_name = "libero_spatial"
task_suite = benchmark_dict[task_suite_name]()

task_id = 0
task = task_suite.get_task(task_id)

task_bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)

print("suite:", task_suite_name)
print("task id:", task_id)
print("task name:", task.name)
print("language:", task.language)
print("bddl:", task_bddl_file)

env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
}

env = OffScreenRenderEnv(**env_args)
env.seed(0)

obs = env.reset()
print("reset ok")
print("obs keys:", obs.keys())

dummy_action = [0.0] * 7

for i in range(5):
    obs, reward, done, info = env.step(dummy_action)
    print(i, "reward:", reward, "done:", done)

env.close()
print("LIBERO env sanity check passed")
