import os
import argparse
import h5py
import numpy as np
import imageio

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path


def safe_check_success(env, reward=None, info=None):
    if isinstance(info, dict):
        for k in ["success", "is_success", "task_success"]:
            if k in info:
                return bool(info[k])

    for fn_name in ["check_success", "_check_success"]:
        if hasattr(env, fn_name):
            try:
                return bool(getattr(env, fn_name)())
            except Exception:
                pass

    if reward is not None:
        return bool(reward > 0.5)

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--demo-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-video", action="store_true", help="whether to save a video of the replay")
    parser.add_argument("--camera", type=str, default="agentview_image", choices=["agentview_image", "robot0_eye_in_hand_image"], help="camera view for the video")
    parser.add_argument("--output", type=str, default=None, help="path to save the replay video")
    args = parser.parse_args()

    print("Dataset:", args.dataset)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task = task_suite.get_task(args.task_id)

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    print("suite:", args.suite)
    print("task id:", args.task_id)
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
    env.reset()

    demo_key = f"demo_{args.demo_id}"

    with h5py.File(args.dataset, "r") as f:
        demo = f["data"][demo_key]

        actions = np.array(demo["actions"])
        states = np.array(demo["states"])
        rewards = np.array(demo["rewards"]) if "rewards" in demo else None
        dones = np.array(demo["dones"]) if "dones" in demo else None

        print("demo:", demo_key)
        print("actions shape:", actions.shape)
        print("states shape:", states.shape)
        if rewards is not None:
            print("dataset reward min/max/final:", rewards.min(), rewards.max(), rewards[-1])
        if dones is not None:
            print("dataset dones sum/final:", dones.sum(), dones[-1])

    # Reset simulator to the first stored simulator state.
    obs = env.set_init_state(states[0])
    print("set_init_state ok")

    T = len(actions)
    if args.max_steps is not None:
        T = min(T, args.max_steps)

    success = False
    final_reward = None
    max_reward = -1e9
    
    video_frames = []
    if args.save_video:
        # Append initial observation
        if args.camera in obs:
            video_frames.append(obs[args.camera][::-1])

    for t in range(T):
        obs, reward, done, info = env.step(actions[t])
        final_reward = reward
        max_reward = max(max_reward, reward)

        success = safe_check_success(env, reward=reward, info=info)
        
        if args.save_video:
            if args.camera in obs:
                video_frames.append(obs[args.camera][::-1])

        if t % 20 == 0 or success or t == T - 1:
            print(
                f"step {t:04d} | reward={reward:.4f} | "
                f"done={done} | success={success}"
            )

        if success:
            print("SUCCESS reached at step:", t)
            break

    env.close()

    print("\n==== Replay Summary ====")
    print("demo:", demo_key)
    print("executed steps:", t + 1)
    print("final reward:", final_reward)
    print("max reward:", max_reward)
    print("success:", success)

    if args.save_video and len(video_frames) > 0:
        if args.output is None:
            dataset_basename = os.path.splitext(os.path.basename(args.dataset))[0]
            args.output = f"replay_{dataset_basename}_{demo_key}_{args.camera}.mp4"
        writer = imageio.get_writer(args.output, fps=30)
        for frame in video_frames:
            writer.append_data(frame)
        writer.close()
        print(f"Successfully saved replay video to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()