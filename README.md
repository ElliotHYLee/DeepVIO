## Update Notes: 04.25.2024

1. For simpler experience, the code assumes KITTI odom. datasets only at the moment.
2. Docker based environment is considered to be main branch for consistent experience.
3. The ground truth and psuedo-imu data is added.
4. The actual image data need to be put in the folder.
5. Weihts zip file is linked.
6. Lots of codes need to be organized and refactored further.
7. For training, still reuquires big RAM capacity to read the data.

# Deep Visual Inertial Odometry

Deep learning based visual-inertial odometry project.

Youtube Overview: 

https://youtu.be/T8hH6Q6KIrc?si=aMZP8SQk5q0PdzRC

pros:
- Lighter CNN structure. No RNNs -> much lighter.
- Training images together with inertial data using exponential mapping.
- Rotation is coming from external attitude estimation.
- No RNN but Kalman filter: Accleration and image fusion for frame-to-frame displacement.

cons:
- no position correction: drift in position: But SLAM can correct the position drfit.


## Please Cite:
Hongyun Lee, James W. Gregory, Matthew McCrink, and Alper Yilmaz. "Deep Learning for Visual-Inertial Odometry: Estimation of Monocular Camera Ego-Motion and its Uncertainty" The Ohio State University, Master Thesis, http://rave.ohiolink.edu/etdc/view?acc_num=osu156331321922759


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

## Qucik View: Results
To quickly view the results, you do not need to run any model. The result data is provided in ~/src/src/Results/. To plot the result data, you can simply run:


```bash
# @ In the docker container,
export PYTHONPATH="/workspace/dvio$PYTHONPATH" >> ~/.bashrc
```

```bash
# @ src/Results in the docker container,
python plot_all.py
```

### For CNN_KF result
You'd still need to run the model. See Step 8 in the "Training and Testing Usage:" section.

## Training and Testing Usage:
   
1. Clone this repo.
    ```bash
    git clone https://github.com/ElliotHYLee/Deep_Visual_Inertial_Odometry --recursive
    ```
2. A folder structure for KITTI dataset is provided. Fill in the actual data (images) accordingly. Or, provide custom bind mount path. To investigate the folder structure, see ~/datasets/KITTI/odom/sequences/00/image_0/
3. Build docker image
   ```bash
    docker compose build
   ```
4. Run docker container
   ```bash
    docker compose up -d
   ```
5. Using the VSCode extensions, attach the VSCode for the created container.

6. Put the sample weight uner the src/Weights. Sample weight link: https://drive.google.com/file/d/1DZL-MtMenJb23uQdjwQ_XuyonehuoovE/view?usp=drive_link

7. Training & Testing
Make sure to turn on/off the training @ main_cnn.py. To run training, set TestOnly=False. Default is true. This will trian and generate weights if you don't have one.

   ```bash
   # main_cnn.py
   runTrainTest('kitti', 'none', seq=[0, 2, 4], seqRange=[0, 11], TestOnly=False) 
   ```

   ```bash
   python main_cnn.py
   ```

   ```bash
   python main_realtime.py
   ```

8. Testing CNN-KF output   
   ```bash
   python main_kf.py
   ```

## Traing Results

Training result on KITTI odometry dataset on sequence 0 (Trained on seq: 0, 2, 4)

<img src="https://github.com/ElliotHYLee/Deep_Visual_Inertial_Odometry/blob/docker/src/Results/Figures/docker_kitti_none0_results.png" width="400">


## Test Results

Test result on KITTI odom. data: seq.5

<img src="https://github.com/ElliotHYLee/Deep_Visual_Inertial_Odometry/blob/docker/src/Results/Figures/master_kitti_none5_results.png" width="400">


## Correction Result
From left to right, velocity, position XYZ, position 2D. Red: ground truth, blue: CNN output, green: Kalman-Filter(CNN + Accelerometer)

<img src="https://github.com/ElliotHYLee/Deep_Visual_Inertial_Odometry/blob/docker/src/Results/Figures/correction_screen_shot.png" width="400">

**Note** As mentioned earlier, this project corrects the CNN output of the velocity using accelerometer's integration by Kalman filter. The positions are simple integrations on XYZ velocity. From 2nd graph's z axis, if lucky, the simple position integration may result in less drift.

To ultimately reduce the position, you will need to bring your own position measurement. i.e SLAM.


