# HPC user instructions for `hpc_version`

This document records the current HPC setup steps for the `hpc_version` branch of `libero_demt`.

The examples below assume the King's CREATE-style HPC paths currently used for this project:

```bash
PROJECT_DIR=/scratch/prj/eng_demt_robot_learning/libero_demt
CONDA_ROOT=/scratch/users/k23114984/conda
ENV_PATH=${CONDA_ROOT}/envs/libero
CUDA_MODULE=cuda/11.7.0-gcc-13.2.0
LIBERO_CONFIG_PATH=/scratch/users/k23114984/.libero
MPLCONFIGDIR=/scratch/users/k23114984/.cache/matplotlib
LIBERO_DATASET_ROOT=/scratch/prj/eng_demt_robot_learning/dataset/datasets
```

## 1. Go to the project and select the branch

```bash
cd /scratch/prj/eng_demt_robot_learning/libero_demt
git status --short --branch
git switch hpc_version
git pull
```

If the branch does not exist locally yet:

```bash
git fetch origin
git switch -c hpc_version origin/hpc_version
```

## 2. Create the conda environment on scratch

Keep the conda environment under `/scratch/users/k23114984/conda/envs` rather than inside the repository.

```bash
conda create -p /scratch/users/k23114984/conda/envs/libero python=3.8.13 -y
```

Activate it with the full path:

```bash
conda activate /scratch/users/k23114984/conda/envs/libero
```

Check the active Python:

```bash
which python
python --version
```

Expected Python version:

```text
Python 3.8.13
```

## 3. Install LIBERO dependencies

Load the CREATE CUDA module before installing or running GPU-related packages:

```bash
module load cuda/11.7.0-gcc-13.2.0
which nvcc
nvcc --version
```

The expected CUDA compiler version from this module is CUDA 11.7. The PyTorch wheel below still uses the CUDA 11.3 runtime because that is the stack pinned by the original LIBERO instructions.

From the repository root and activated conda environment:

```bash
cd /scratch/prj/eng_demt_robot_learning/libero_demt
conda activate /scratch/users/k23114984/conda/envs/libero
export LIBERO_CONFIG_PATH=/scratch/users/k23114984/.libero
export MPLCONFIGDIR=/scratch/users/k23114984/.cache/matplotlib

pip install "cmake<4"
pip install -r requirements.txt
```

`cmake<4` is needed because `robomimic==0.2.0` pulls in `egl-probe`, whose build is not compatible with CMake 4.x.

Install the CUDA 11.3 PyTorch stack:

```bash
pip install torch==1.11.0+cu113 \
            torchvision==0.12.0+cu113 \
            torchaudio==0.11.0 \
            --extra-index-url https://download.pytorch.org/whl/cu113
```

Then install the local package in editable mode:

```bash
pip install -e .
```

Disable robosuite's numba disk cache in the installed environment. On this HPC setup, the default `CACHE_NUMBA = True` can fail during robosuite imports with `RuntimeError: cannot cache function ... no locator available`.

```bash
perl -0pi -e 's/CACHE_NUMBA = True/CACHE_NUMBA = False/g' \
    /scratch/users/k23114984/conda/envs/libero/lib/python3.8/site-packages/robosuite/macros.py
```

## 4. Verify the installation

Run these checks from the activated environment, with the CUDA module loaded:

```bash
module load cuda/11.7.0-gcc-13.2.0
conda activate /scratch/users/k23114984/conda/envs/libero
export LIBERO_CONFIG_PATH=/scratch/users/k23114984/.libero
export MPLCONFIGDIR=/scratch/users/k23114984/.cache/matplotlib

mkdir -p "${LIBERO_CONFIG_PATH}" "${MPLCONFIGDIR}"

python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import torchvision, torchaudio; print(torchvision.__version__); print(torchaudio.__version__)"
python -c "import libero; print('libero import ok')"
python -c "import robosuite, robomimic, bddl; print('core deps import ok')"
python -m pip check
```

Expected package versions after the PyTorch install:

```text
torch: 1.11.0+cu113
torch CUDA runtime: 11.3
torchvision: 0.12.0+cu113
torchaudio: 0.11.0+cu113
```

Check the installed command-line entry points:

```bash
lifelong.main --help
lifelong.eval --help
```

On first run, LIBERO may ask where to store datasets. For the current project setup, answer with:

```text
/scratch/prj/eng_demt_robot_learning/dataset
```

LIBERO appends `datasets`, so the resulting dataset path in `/scratch/users/k23114984/.libero/config.yaml` should be:

```text
/scratch/prj/eng_demt_robot_learning/dataset/datasets
```

The current verified config path is:

```text
/scratch/users/k23114984/.libero/config.yaml
```

The current verified `libero_object` dataset path is:

```text
/scratch/prj/eng_demt_robot_learning/dataset/datasets/libero_object
```

It contains the expected task HDF5 demo files, for example:

```text
pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5
pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5
pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5
pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5
pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5
pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5
pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5
pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5
pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5
```

On a login node, `torch.cuda.is_available()` may print `False`, and `nvidia-smi` may fail because GPUs are normally only visible inside an allocated GPU job. This does not mean the conda environment or repo install failed.

## 5. GPU job notes

Inside an interactive GPU allocation or batch job, load the same CUDA module and activate the same conda environment:

```bash
module load cuda/11.7.0-gcc-13.2.0
conda activate /scratch/users/k23114984/conda/envs/libero
```

Verify GPU visibility inside the GPU job:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Inside a proper GPU allocation, `torch.cuda.is_available()` should print `True`.

The current GPU verification on `erc-hpc-comp194` showed:

```text
NVIDIA A40
torch: 1.11.0+cu113
torch CUDA runtime: 11.3
torch.cuda.is_available(): True
```

For LIBERO training or evaluation, export the GPU IDs inside the job:

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
```

Example training command:

```bash
python libero/lifelong/main.py seed=0 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base
```

Use your HPC batch script or interactive allocation command to request a GPU before running training.

Small smoke-test training command for the verified `libero_object` dataset:

```bash
cd /scratch/prj/eng_demt_robot_learning/libero_demt

module load cuda/11.7.0-gcc-13.2.0
conda activate /scratch/users/k23114984/conda/envs/libero
export LIBERO_CONFIG_PATH=/scratch/users/k23114984/.libero
export MPLCONFIGDIR=/scratch/users/k23114984/.cache/matplotlib
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

lifelong.main benchmark_name=LIBERO_OBJECT \
    policy=bc_rnn_policy \
    lifelong=base \
    seed=0 \
    train.n_epochs=1 \
    train.batch_size=4 \
    train.num_workers=1 \
    eval.n_eval=1 \
    eval.num_procs=1 \
    eval.use_mp=false \
    eval.max_steps=10 \
    use_wandb=false
```

This is only a smoke test. It should load the `libero_object` HDF5 files and enter training.

Do not set `train.num_workers=0` for this code path. `libero/lifelong/algos/base.py` creates the PyTorch `DataLoader` with `persistent_workers=True`, and PyTorch requires `num_workers > 0` when persistent workers are enabled.

The `eval.eval=false` flag is not enough to disable evaluation in `libero/lifelong/algos/base.py`; that code path still calls evaluation on epoch 0. Keep the smoke test evaluation small with `eval.n_eval=1`, `eval.num_procs=1`, `eval.use_mp=false`, and `eval.max_steps=10`. The default evaluation settings spawn up to 20 offscreen render workers, which can fail on a shared GPU node with `EGL_BAD_ALLOC`.

## 6. Save branch updates to GitHub

Check local changes:

```bash
git status --short --branch
```

Commit documentation or setup changes:

```bash
git add docs/hpc_user_instruct.md
git commit -m "Add HPC setup instructions"
```

Push to the GitHub branch:

```bash
git push -u origin hpc_version
```

After pushing, the branch should be visible on GitHub as `hpc_version`.

codex resume 019ee444-eea1-7bb0-9f8c-478a8812e948
