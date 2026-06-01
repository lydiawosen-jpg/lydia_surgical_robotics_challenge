# AMBF Surgical Robotics Challenge for ICRA 2026

This branch provides specific instructions for the [ICRA 2026 Surgical Robotics Challenge](https://surgical-robotics-ai.github.io/icra-competition-2026/index.html), which implements a peg transfer task using ROS2.

Please check out the [Discussions tab](https://github.com/surgical-robotics-ai/surgical_robotics_challenge/discussions) to ask questions, post suggestions, connect with the community, and stay up to date with the challenge.

## Installation

1. Install ROS2 Jazzy, which can be found [here](https://docs.ros.org/en/jazzy/Installation.html).

2. Clone, build, and source `ambf-3.0` using these [instructions](https://github.com/WPI-AIM/ambf/wiki/Installing-AMBF).

3. Clone this repository (`icra2026-challenge` branch) to your local machine (recommended) OR use a Dockerfile

   a. Option 1: Clone repo to your local machine: Please refer to the [README](./scripts/README.md) in the [scripts](./scripts) folder for instructions on installing the Python package for system-wide access.

   b. Option 2: Use Dockerfile: You can create Docker images by following the instructions [here](https://github.com/surgical-robotics-ai/docker_surgical_robotics_challenge).

## Running the simulation

1. Open a terminal and set up for ROS 2, either in every terminal window that interacts with SRC or once in your `.bashrc` file, as described [here](https://github.com/WPI-AIM/ambf/wiki/Installing-AMBF#step-3).

2. Navigate to the `surgical_robotics_challenge` folder, which is `~/surgical_robotics_challenge` if you cloned it in your home directory.

3. Run the following in your terminal:

```bash
./run_env_pegboard_symmetric_with_wall.sh
```

A pair of windows showing a pegboard environment should appear.

  <p align="center">
  <img src=Media/sample_scene_pegboard_symmetric.png width="600"/>
  </p>

## Teleoperating the simulated robots

After completing the steps above to run the simulation:

1. Enable the Collaborative Robotics Toolkit (CRTK) interface, by running the following script:

```bash
cd scripts/surgical_robotics_challenge
python launch_crtk_interface --scene False
```

The `scene` argument above is relevant for the suturing environment with entry and exit holes, so it should be disabled (`False`) for other environments, including the pegboard used for the ICRA 2026 challenge.

2. Change to the teleoperation directory:

```bash
cd surgical-robotics-challenge/scripts/surgical_robotics_challenge/teleoperation
```

3. Run the teleoperation control using the following command line:

```
python udp_crtk_bridge.py
```

This control script converts CRTK-compatible JSON commands sent/received via UDP from the master device (e.g., Quest 3) to CRTK-compatible ROS2 topics for AMBF.

## Technical Details

### Launch file
To understand the launch file, refer to this [link](https://github.com/WPI-AIM/ambf/wiki/Selecting-Robots).

### Simulated Cameras
The simulated camera(s) are defined in the world file ([`world_stereo.yaml`](./ADF/world/world_stereo.yaml)), which is selected in [`launch.yaml`](./launch.yaml).
To enable the camera(s) to publish scene images or depth data, follow the [instructions](https://github.com/WPI-AIM/ambf/wiki/Camera-feed-and-depth-camera) on this page.

### Camera Coordinate frames
Camera coordinate frames, and the difference between the AMBF and `OpenCV` camera conventions, are described in [camera_conventions.md](./docs/camera_conventions.md).

### Resetting the Simulation
You can press `CTRL+R` to reset the rigid bodies in the simulation and `CTRL+V` to reset the camera pose.

### Launch Arguments
To manually control which objects are spawned in the scene, review the `.sh` scripts in this folder. For a full list of arguments that can be passed to AMBF, refer to these [instructions](https://github.com/WPI-AIM/ambf/wiki/Command-Line-Arguments).

## Citation
If you find this work useful, please cite it as:

```bibtex
@article{munawar2022open,
  title={Open Simulation Environment for Learning and Practice of Robot-Assisted Surgical Suturing},
  author={Munawar, Adnan and Wu, Jie Ying and Fischer, Gregory S and Taylor, Russell H and Kazanzides, Peter},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={2},
  pages={3843--3850},
  year={2022},
  publisher={IEEE}
}
```
