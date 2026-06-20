import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot end-effector trajectories from a LIBERO dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the HDF5 dataset file")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output trajectory plot")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset path '{args.dataset}' does not exist.")
        return

    # Create figure
    fig = plt.figure(figsize=(16, 7))
    
    # 3D Subplot
    ax_3d = fig.add_subplot(121, projection='3d')
    # 2D Subplot (Top-down view, X-Y plane)
    ax_2d = fig.add_subplot(122)

    all_ee_pos = []
    all_grippers = []
    all_actions = []
    
    with h5py.File(args.dataset, "r") as f:
        demos = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[1]) if "_" in x else x)
        print(f"Found {len(demos)} demonstrations in the dataset.")
        
        # We will plot up to 50 trajectories
        num_to_plot = min(50, len(demos))
        print(f"Plotting {num_to_plot} trajectories...")
        
        # Colormap for trajectories to distinguish them
        colors = plt.cm.viridis(np.linspace(0, 0.9, num_to_plot))
        
        for idx in range(num_to_plot):
            demo_key = demos[idx]
            demo = f["data"][demo_key]
            
            ee_pos = np.array(demo["obs/ee_pos"])
            gripper_states = np.array(demo["obs/gripper_states"])
            actions = np.array(demo["actions"])
            
            all_ee_pos.append(ee_pos)
            all_grippers.append(gripper_states)
            all_actions.append(actions)
            
            # Extract x, y, z coordinates
            x = ee_pos[:, 0]
            y = ee_pos[:, 1]
            z = ee_pos[:, 2]
            
            # Plot in 3D
            ax_3d.plot(x, y, z, color=colors[idx], alpha=0.5, linewidth=1.5)
            # Mark start point and end point
            if idx == 0:
                ax_3d.scatter(x[0], y[0], z[0], color='green', marker='o', s=50, label='Start (t=0)')
                ax_3d.scatter(x[-1], y[-1], z[-1], color='red', marker='*', s=70, label='End (t=T)')
            else:
                ax_3d.scatter(x[0], y[0], z[0], color='green', marker='o', s=15, alpha=0.7)
                ax_3d.scatter(x[-1], y[-1], z[-1], color='red', marker='*', s=25, alpha=0.7)
                
            # Plot in 2D (X-Y plane)
            ax_2d.plot(x, y, color=colors[idx], alpha=0.5, linewidth=1.5)
            if idx == 0:
                ax_2d.scatter(x[0], y[0], color='green', marker='o', s=50, label='Start (t=0)')
                ax_2d.scatter(x[-1], y[-1], color='red', marker='*', s=70, label='End (t=T)')
            else:
                ax_2d.scatter(x[0], y[0], color='green', marker='o', s=15, alpha=0.7)
                ax_2d.scatter(x[-1], y[-1], color='red', marker='*', s=25, alpha=0.7)

    # Styling 3D Plot
    ax_3d.set_title("3D End-Effector Trajectories", fontsize=14, fontweight='semibold')
    ax_3d.set_xlabel("X (m)", fontsize=11)
    ax_3d.set_ylabel("Y (m)", fontsize=11)
    ax_3d.set_zlabel("Z (m)", fontsize=11)
    ax_3d.legend(loc='upper left')
    ax_3d.grid(True, linestyle='--', alpha=0.5)
    
    # Styling 2D Plot
    ax_2d.set_title("2D Projection (X-Y Plane / Top-down View)", fontsize=14, fontweight='semibold')
    ax_2d.set_xlabel("X (m)", fontsize=11)
    ax_2d.set_ylabel("Y (m)", fontsize=11)
    ax_2d.legend(loc='upper left')
    ax_2d.grid(True, linestyle='--', alpha=0.5)
    ax_2d.set_aspect('equal', adjustable='box')
    
    # Title and saving
    dataset_basename = os.path.splitext(os.path.basename(args.dataset))[0]
    plt.suptitle(f"End-Effector Trajectory Analysis\nDataset: {dataset_basename}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if args.output is None:
        args.output = f"{dataset_basename}_eef_trajectories.png"
        
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"Successfully saved trajectory plot to: {os.path.abspath(args.output)}")
    
    # Print quantitative statistics
    lengths = [len(traj) for traj in all_ee_pos]
    starts = np.array([traj[0] for traj in all_ee_pos])
    ends = np.array([traj[-1] for traj in all_ee_pos])
    
    print("\n" + "="*40)
    print("        TRAJECTORY STATISTICS")
    print("="*40)
    print(f"Total trajectories amount : {len(all_ee_pos)}")
    print(f"Trajectory length (steps)   : Min={np.min(lengths)}, Max={np.max(lengths)}, Mean={np.mean(lengths):.1f}, Std={np.std(lengths):.1f}")
    print(f"Start EEF Position Bounds   : X=[{starts[:, 0].min():.4f}, {starts[:, 0].max():.4f}], Y=[{starts[:, 1].min():.4f}, {starts[:, 1].max():.4f}], Z=[{starts[:, 2].min():.4f}, {starts[:, 2].max():.4f}]")
    print(f"End EEF Position Bounds     : X=[{ends[:, 0].min():.4f}, {ends[:, 0].max():.4f}], Y=[{ends[:, 1].min():.4f}, {ends[:, 1].max():.4f}], Z=[{ends[:, 2].min():.4f}, {ends[:, 2].max():.4f}]")
    
    # Analyze gripper states and action range
    flat_grippers = np.concatenate(all_grippers, axis=0)
    flat_actions = np.concatenate(all_actions, axis=0)
    print("\n" + "="*40)
    print("        GRIPPER & ACTION ANALYSIS")
    print("="*40)
    print(f"Gripper States (Finger positions) Range : Left=[{flat_grippers[:, 0].min():.4f}, {flat_grippers[:, 0].max():.4f}], Right=[{flat_grippers[:, 1].min():.4f}, {flat_grippers[:, 1].max():.4f}]")
    print(f"Action Limits (dims 0-6)                :")
    for d in range(7):
        print(f"  dim {d} range: [{flat_actions[:, d].min():.4f}, {flat_actions[:, d].max():.4f}]")
    print("="*40)

if __name__ == "__main__":
    main()
