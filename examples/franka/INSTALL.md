# realsense
uv pip install pyrealsense2-2.53.1.4623-cp39-cp39-manylinux1_x86_64

# xensesdk
uv pip install nvidia-cudnn-cu11==8.9.2.26
uv pip install nvidia-cuda-runtime-cu11
uv pip install onnxruntime-gpu==1.18.0
pip install xensesdk==1.6.0 -i https://repo.huaweicloud.com/repository/pypi/simple/

# franka controller
cd ~/workspace/yhx/franka_control
source /home/mpi/workspace/yhx/openpi/examples/franka/.venv/bin/activate
uv pip install -e .

# openpi-client
uv pip install -e packages/openpi-client
uv pip install tyro