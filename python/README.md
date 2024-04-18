# Python code

## Installation

```
conda create -y -n hghoi python=3.8
conda activate hghoi
pip install -r requirements.txt
```

## Exported training data and pretrained models.

You could use Unity to export the training data by yourself. We also provide our exported data. You can access the data and the pretrained models from [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/hjpi_zju_edu_cn/ErSIBkSOwSFJh6Fk1Ml6BNABqulxpY_mEcGhnxxtQ0l2Bg?e=AzNZkD).

For the training data, please download `MotionWorld.zip` and unzip it in `python/datasets/samp/`.

For the pretrained models, please download `work_dirs.zip` and unzip it in `python/`.

## Training

### Goal pose
Train the part VQ-VAE.
```
python train.py --config configs/goalpose/vqvae.yaml
```

Extract the codebook of goal poses.
```
python tools/postprocess.py --config work_dirs/gp_d4_n8/config.yaml,configs/goalpose/extract.yaml --epoch 50 --func extract_code --forward-func forward_step
```

Train the part VQ-VAE generator.
```
python train.py --config configs/goalpose/generator.yaml
```

### Milestone
Get the statistics.
```
python tools/post_calculate.py --config configs/motion/traj.yaml --name Milestone
```

Train the milestone point model.
```
python train.py --config configs/motion/milestone.yaml
```

Train the milestone pose model.
```
python train.py --config configs/motion/milestone_pose.yaml
```

### Motion
Get the statistics.
```
python tools/post_calculate.py --config configs/motion/traj.yaml --name Traj
```

Train the trajectory completion model.
```
python train.py --config configs/motion/traj.yaml
```

Train the motion infilling model.
```
python train.py --config configs/motion/motion.yaml
```

## Inference (Build servers to connect with Unity)

We build Python servers and connect with Unity by socket. Please assign correct port in the Unity.

### Goal pose
```
python server.py --server-type PoseContactServer --hostname localhost --client-type c --config ./work_dirs/gp_d4_n8_code/config.yaml --test-epoch 1 --external-config work_dirs/gp_d4_n8/config.yaml --external-epoch 50 --port 3414
```

### Milestone
Milestone point.
```
python server.py --server-type MilestoneServer --hostname localhost --client-type c --config ./work_dirs/milestone/config.yaml --test-epoch 250 --port 3452
```

Milestone pose.
```
python server.py --server-type PoseServer --hostname localhost --client-type c --config ./work_dirs/milestone_pose/config.yaml --test-epoch 100 --port 3466
```

### Motion
Trajectory completion.
```
python server.py --server-type TrajCompletionServer --hostname localhost --client-type c --config ./work_dirs/traj/config.yaml --test-epoch 250 --port 3446
```

Motion infilling.
```
python server.py --server-type MotionServer --hostname localhost --client-type c --config ./work_dirs/motion/config.yaml --test-epoch 10 --port 3494
```

