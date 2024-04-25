# Event-Guide-Object-Detection-Under-Spiking-Transmission
***
![EGSTNet](https://github.com/wang1239t/myimg/blob/main/EGSTNet_1.png)
## Installation
### Conda
Our environment runs on CUDA 11.6, eight RTX 3090 GPUs and Ubuntu 22.04
```
conda create -y -n egstnet python=3.7
conda activate egstnet
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
### Venv
Install the dependencies for the environment.
```
pip install -r requirements.txt
pip install torch-scatter==2.1.1
```
### CUDA operators
Compiling CUDA operators
```
cd models/dino/ops
python setup.py build install
```
## Date preparation
We used the [PKU-DAVIS-SOD Dataset](https://openi.pcl.ac.cn/LiDianze/PKU-DAVIS-SOD) dataset.
The directory structure after processing is as follows:
```
- dataset/
  - train/
    - annotations
      - train_annotions.json
    - images/
      - image1.jpg
      - image2.jpg
      ...
    - events/
      - event1.npy
      - event2.npy
      ...
  - val/
      ...
  - train_normal/
      ...
```
## Training
Select the appropriate paths and storage location, and run the train file.
```
python train.py --dataset_type='' --gpu=0 --sod_path='/home/dataset/SOD' --backbone='spikenext-B'
--output_dir='/home/output/EGSTNet-B' --use_pre_event='RME'
--config_file='../config/SOD/EGSTNet_5scale.py' --batch_size=4
```

## Test
To test using a pre-trained checkpoint, you need to modify the path information in the config_args_raw.json file.
```
train.py --dataset_type='' --gpu=0 --sod_path='/home/dataset/SOD' --backbone='spikenext-B'
--output_dir='/home/output/EGSTNet-B' --config_file='../config/SOD/EGSTNet_5scale.py' --resume='/home/output/EGSTNet-B'
```

## Code Acknowledgments
We used the code from the following project. <br>
[DINO-DETR](https://github.com/IDEA-Research/DINO) for network model framework. <br>
[EGRO-12](https://github.com/uzh-rpg/event_representation_study) for event dealing method.
