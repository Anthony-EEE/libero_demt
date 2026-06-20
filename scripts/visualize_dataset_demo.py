import os
import argparse
import h5py
import imageio
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Visualize pre-recorded RGB images from a LIBERO HDF5 dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the HDF5 dataset file")
    parser.add_argument("--demo-id", type=int, default=0, help="ID of the demo/episode to visualize")
    parser.add_argument("--camera", type=str, default="agentview_rgb", choices=["agentview_rgb", "eye_in_hand_rgb"],
                        help="Camera view to visualize")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output MP4 video (default: <dataset_name>_demo_<id>_<camera>.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="FPS of the output video")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset path '{args.dataset}' does not exist.")
        return

    demo_key = f"demo_{args.demo_id}"
    
    with h5py.File(args.dataset, "r") as f:
        if "data" not in f:
            print("Error: 'data' group not found in HDF5 file.")
            return
        
        if demo_key not in f["data"]:
            available_demos = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[1]) if "_" in x else x)
            print(f"Error: Demo '{demo_key}' not found in the dataset.")
            print(f"Available demos are: {available_demos[:10]} ... (total {len(available_demos)})")
            return
        
        demo = f["data"][demo_key]
        if "obs" not in demo:
            print(f"Error: 'obs' group not found in {demo_key}.")
            return
        
        if args.camera not in demo["obs"]:
            available_cameras = list(demo["obs"].keys())
            print(f"Error: Camera '{args.camera}' not found in {demo_key}/obs.")
            print(f"Available keys in obs: {available_cameras}")
            return
        
        # Extract images: shape is (T, H, W, C)
        images = np.array(demo["obs"][args.camera])
        
    print(f"Loaded {images.shape[0]} frames from '{demo_key}/obs/{args.camera}' with shape {images.shape[1:]}.")
    
    if args.output is None:
        dataset_basename = os.path.splitext(os.path.basename(args.dataset))[0]
        args.output = f"{dataset_basename}_{demo_key}_{args.camera}.mp4"
        
    writer = imageio.get_writer(args.output, fps=args.fps)
    for i, img in enumerate(images):
        # Robosuite/Mujoco images are stored upside down (vertically flipped) in HDF5, flip them back
        writer.append_data(img[::-1])
    writer.close()
    
    print(f"Successfully saved video to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
