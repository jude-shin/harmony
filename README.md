# harmony
- Tensorflow and Python
- Card Recognition

- pip, python 3.11
- performed on unix device

# NOTE:
- python and python3 are interchangeable here, and it may be the case that some of the commands will have to be run with one or the other, depending on your machine

# VENV 
### step 0: Create the New venv
``` bash 
python3 -m venv .venv
 ```

### step 1: Activate the venv
- if there is no .venv file in the directory, go to step 0
- activate the venv
``` bash
source .venv/bin/activate
```
- note: when you are done using the venv, you may choose to deactivate it with the following command
``` bash
deactivate
```

# Project Structure (where to find what)

# How to use
### API for storepass endpoint

### Training models

#### manual

#### automated scripts


### Checking status of automated scripts

# MISC notes (for dev)
### nvidia-container-toolkit
#### 1) Add NVIDIA’s repo once (skip if you already have it)
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
#### 2) Install (or replace any older nvidia-docker2 install)
```
sudo apt update
sudo apt install -y nvidia-container-toolkit
```
### 3) Register the runtime with Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker


# ScANN idea
https://www.tensorflow.org/recommenders/examples/basic_retrieval#item-to-item_recommendation


### GPU usage
https://www.tensorflow.org/install/pip#linux_setup
### other frustrating tensorflow and NVIDIA 
sudo echo 0 | sudo tee -a /sys/bus/pci/devices/0000:01:00.0/numa_node
### CUDA cuDNN and TensorRT
https://www.youtube.com/watch?v=1Tr1ifuSh6o
#### cuDNN
https://developer.nvidia.com/rdp/cudnn-archive
#### CUDA
https://developer.nvidia.com/cuda-12-1-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local
#### TensorRT
https://developer.nvidia.com/nvidia-tensorrt-8x-download
### anaCONDA environments
https://medium.com/@dev-charodeyka/tensorflow-conda-nvidia-gpu-on-ubuntu-22-04-3-lts-ad61c1d9ee32

