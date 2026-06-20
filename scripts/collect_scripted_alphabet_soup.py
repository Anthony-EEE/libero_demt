import argparse
import datetime
import json
import os
import time

import h5py
import numpy as np
import robosuite as suite
from robosuite import load_controller_config

import init_path
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING


DEFAULT_BDDL = (
    "/home/endongsun/robot_learning/LIBERO/libero/libero/bddl_files/libero_object/"
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
)


def make_env(args):
    controller_config = load_controller_config(default_controller=args.controller)
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    problem_name = problem_info["problem_name"]
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=args.render,
        has_offscreen_renderer=not args.render,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    return env, config, problem_info


def body_pos(env, name):
    return np.array(env.sim.data.body_xpos[env.obj_body_id[name]], dtype=np.float64)


def eef_pos(obs):
    return np.array(obs["robot0_eef_pos"], dtype=np.float64)


def action_to_target(obs, target, gripper, pos_gain, max_pos_action, action_dim):
    delta = np.asarray(target, dtype=np.float64) - eef_pos(obs)
    pos_action = np.clip(pos_gain * delta, -max_pos_action, max_pos_action)
    action = np.zeros(action_dim, dtype=np.float64)
    action[:3] = pos_action
    # Keep orientation fixed. The default Panda gripper starts pointing down.
    action[-1] = gripper
    return action


def rollout_to_target(env, obs, target, gripper, args, states, actions, label):
    for step in range(args.steps_per_waypoint):
        states.append(env.sim.get_state().flatten().copy())
        action = action_to_target(
            obs=obs,
            target=target,
            gripper=gripper,
            pos_gain=args.pos_gain,
            max_pos_action=args.max_pos_action,
            action_dim=env.action_dim,
        )
        actions.append(action.copy())
        obs, reward, done, info = env.step(action)

        if args.render:
            env.render()

        err = np.linalg.norm(eef_pos(obs) - target)
        if err <= args.waypoint_tolerance and step >= args.min_steps_per_waypoint:
            break

    if args.verbose:
        print(f"{label}: final_err={np.linalg.norm(eef_pos(obs) - target):.4f}")
    return obs


def hold(env, obs, gripper, args, states, actions, n_steps, label):
    for _ in range(n_steps):
        states.append(env.sim.get_state().flatten().copy())
        action = np.zeros(env.action_dim, dtype=np.float64)
        action[-1] = gripper
        actions.append(action.copy())
        obs, reward, done, info = env.step(action)
        if args.render:
            env.render()
    if args.verbose:
        print(label)
    return obs


def scripted_episode(env, args):
    obs = env.reset()

    soup = body_pos(env, "alphabet_soup_1")
    basket = body_pos(env, "basket_1")

    approach_soup = np.array([soup[0], soup[1], args.approach_z])
    grasp_soup = np.array([soup[0], soup[1], args.grasp_z])
    lift_soup = np.array([soup[0], soup[1], args.lift_z])
    approach_basket = np.array([basket[0], basket[1], args.lift_z])
    release_basket = np.array([basket[0], basket[1], args.release_z])

    states = []
    actions = []

    # gripper convention follows robosuite input_utils: -1=open, +1=closed.
    obs = hold(env, obs, -1.0, args, states, actions, args.initial_hold_steps, "open gripper")
    obs = rollout_to_target(env, obs, approach_soup, -1.0, args, states, actions, "approach soup")
    obs = rollout_to_target(env, obs, grasp_soup, -1.0, args, states, actions, "descend to soup")
    obs = hold(env, obs, 1.0, args, states, actions, args.grasp_hold_steps, "close gripper")
    obs = rollout_to_target(env, obs, lift_soup, 1.0, args, states, actions, "lift soup")
    obs = rollout_to_target(env, obs, approach_basket, 1.0, args, states, actions, "move to basket")
    obs = rollout_to_target(env, obs, release_basket, 1.0, args, states, actions, "descend to basket")
    obs = hold(env, obs, -1.0, args, states, actions, args.release_hold_steps, "open gripper")
    obs = rollout_to_target(
        env,
        obs,
        np.array([basket[0], basket[1], args.lift_z]),
        -1.0,
        args,
        states,
        actions,
        "retreat",
    )
    obs = hold(env, obs, -1.0, args, states, actions, args.final_hold_steps, "final hold")

    success = bool(env._check_success())
    return success, np.array(states), np.array(actions)


def save_raw_hdf5(output_path, env, env_info, problem_info, args, episodes):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")

    with h5py.File(output_path, "w") as f:
        grp = f.create_group("data")
        env_name = None

        for i, (states, actions, model_xml) in enumerate(episodes, start=1):
            ep_grp = grp.create_group(f"demo_{i}")
            ep_grp.attrs["model_file"] = model_xml
            ep_grp.create_dataset("states", data=states)
            ep_grp.create_dataset("actions", data=actions)
            env_name = type(env).__name__

        now = datetime.datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = env_info
        grp.attrs["problem_info"] = json.dumps(problem_info)
        grp.attrs["bddl_file_name"] = args.bddl_file
        with open(args.bddl_file, "r", encoding="utf-8") as f_bddl:
            grp.attrs["bddl_file_content"] = f_bddl.read()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect scripted raw demos for LIBERO alphabet soup -> basket."
    )
    parser.add_argument("--bddl-file", type=str, default=DEFAULT_BDDL)
    parser.add_argument(
        "--output",
        type=str,
        default="/home/endongsun/robot_learning/LIBERO/demonstration_data/scripted_alphabet_soup/demo.hdf5",
    )
    parser.add_argument("--num-demonstration", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda"])
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--approach-z", type=float, default=0.26)
    parser.add_argument("--grasp-z", type=float, default=0.06)
    parser.add_argument("--lift-z", type=float, default=0.32)
    parser.add_argument("--release-z", type=float, default=0.27)

    parser.add_argument("--pos-gain", type=float, default=8.0)
    parser.add_argument("--max-pos-action", type=float, default=1.0)
    parser.add_argument("--waypoint-tolerance", type=float, default=0.015)
    parser.add_argument("--steps-per-waypoint", type=int, default=220)
    parser.add_argument("--min-steps-per-waypoint", type=int, default=8)
    parser.add_argument("--initial-hold-steps", type=int, default=10)
    parser.add_argument("--grasp-hold-steps", type=int, default=50)
    parser.add_argument("--release-hold-steps", type=int, default=60)
    parser.add_argument("--final-hold-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    env, config, problem_info = make_env(args)
    env_info = json.dumps(config)
    env.seed(args.seed)

    episodes = []
    attempts = 0
    try:
        while len(episodes) < args.num_demonstration and attempts < args.max_attempts:
            attempts += 1
            print(f"[attempt {attempts}] collecting scripted episode")
            model_xml = None
            success, states, actions = scripted_episode(env, args)
            try:
                model_xml = env.sim.model.get_xml()
            except Exception:
                model_xml = env.model.get_xml()

            print(
                f"[attempt {attempts}] success={success} "
                f"steps={len(actions)} collected={len(episodes)}/{args.num_demonstration}"
            )
            if success:
                episodes.append((states, actions, model_xml))

        if len(episodes) < args.num_demonstration:
            raise RuntimeError(
                f"Only collected {len(episodes)} successful demos after {attempts} attempts. "
                "Try tuning --grasp-z, --release-z, --max-pos-action, or use --render --verbose."
            )

        save_raw_hdf5(args.output, env, env_info, problem_info, args, episodes)
        print(f"[done] saved {len(episodes)} scripted demos to {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
