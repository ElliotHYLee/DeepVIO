# Deep Visual Inertial Odometry

Deep learning based visual-inertial odometry project.

pros:
- Lighter CNN structure. No RNNs -> much lighter.
- Training images together with inertial data using exponential mapping.
- Rotation is coming from external attitude estimation.
- No RNN but Kalman filter: Accleration and image fusion for frame-to-frame displacement.

cons:
- no position correction: drift in position: But SLAM can correct the position drfit.


## Please Cite:
Hongyun Lee, James W. Gregory, Matthew McCrink, and Alper Yilmaz. "Deep Learning for Visual-Inertial Odometry: Estimation of Monocular Camera Ego-Motion and its Uncertainty" The Ohio State University, Master Thesis, https://etd.ohiolink.edu/pg_10?0::NO:10:P10_ACCESSION_NUM:osu156331321922759


## Tested System
- OS
  - Ubuntu 22.04
  - VSCode

- Hardware
  - CPU: i9-7940x
  - RAM: 128G, 3200Hz
  - GPU: 2 x Gefore 1080 ti

## Prereq.s
1. Docker
2. NVIDIA GPUs and driver. To test the correct installation on ubuntu, try:
    ```
    nvidia-smi
    ```

3. NVIDIA docker tool kit. (installation script provided.)
   ```
   bash nvidia_container_toolkit.bash
   ```
4. Your own KITTI's odometry dataset.
5. VSCode extensions
   - Docker by Microsfot
   - Remote Development by Microsoft

## Usage:
   
1. Clone this repo.
    ```
    git clone -- recursive https://github.com/ElliotHYLee/Deep_Visual_Inertial_Odometry 
    ```
2. A folder structure for KITTI dataset is provided. Fill in the actual data (images) accordingly. Or, provide custom bind mount path. To investigate the folder structure, see ~/datasets/KITTI/odom/sequences/00/image_0/
3. Build docker image
   ```
    docker compose build
   ```
4. Run docker container
   ```
    docker compose up -d
   ```
5. Using the VSCode extensions, attach the VSCode for the created container.




## Traing Results
description

<ul>
<img src="https://github.com/ElliotHYLee/Deep_Visual_Inertial_Odometry/blob/docker/src/Results/Figures/docker_kitti_none0_results.png" width="400">
</ul>

## Test Results
<ul>
<img src="https://github.com/ElliotHYLee/Deep_Visual_Inertial_Odometry/blob/docker/src/Results/Figures/master_kitti_none5_results.png" width="400">
</ul>

## Correction Result

<ul>
<img src="" width="400">
</ul>
