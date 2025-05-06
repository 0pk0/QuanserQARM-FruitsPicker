# QuanserQARM-FruitsPicker
Here’s a professional and structured **README.md** you can use for your GitHub repository to present your fruit-sorting robotic system project clearly to collaborators, recruiters, or researchers:

---

# 🍓🍌 Robotic Fruit Sorting System Using Quanser QArm

This repository contains the code, simulation models, and documentation for a modular, vision-guided robotic fruit-sorting system using Quanser QArm manipulators. The project supports both **teleoperation** and **autonomous operation**, integrated with **YOLOv8 object detection**, **RGB-D perception**, and **ROS + MATLAB Simulink-based control**.

---

## 📦 Project Overview

* **Purpose**: Automate fruit sorting in an industrial environment using robotic manipulators.
* **Fruits Sorted**: Strawberries, Bananas, and Tomatoes (including quality control for rotten/unripe fruit).
* **Modes**:

  * 👨‍💻 Teleoperation via keyboard GUI
  * 🤖 Autonomous pick-and-place via real-time object detection + trajectory planning
* **Simulation Platforms**: MATLAB Simulink, ROS2 (Foxy), Gazebo Classic
* **Hardware**: 3× Quanser QArms + Intel RealSense D415 + Tekscan FlexiForce sensors

---

## 🛠️ Features

* YOLOv8 object detection with grasp point and orientation estimation
* RealSense D415 RGB-D camera-based 3D vision pipeline
* Force-sensitive gripping with compliant feedback
* ROS2 nodes for IK, trajectory generation, and motor control
* MATLAB Simulink interface for trajectory block simulation
* Factory-style conveyor belt layout with modular arm placement
* Simulations in both Gazebo and Quanser Interactive Labs
* Autonomous QA station for rejecting unripe/rotten fruits

---

## 📂 Repository Structure

```
├── datasets/                  # YOLOv8 fruit dataset (link or reference)
├── detection/                 # YOLOv8 training scripts and config
├── ros_ws/                    # ROS2 workspace: launch files, nodes, URDF, sdf, etc.
│   ├── src/
│   └── install/
├── simulink_models/          # MATLAB Simulink trajectory and control models
├── teleop/                   # Python-based keyboard control interface
├── vision_module/            # Real-time grasp angle estimation and depth integration
├── images/                   # Architecture diagrams, simulation screenshots
├── docs/                     # Report, CAD layout, PDF deliverables
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone [https://github.com/0pk0/fruit-sorting-qarm.git]
cd fruit-sorting-qarm
```

### 2. Install Dependencies

* Python 3.8+
* YOLOv8 (`pip install ultralytics`)
* OpenCV (`pip install opencv-python`)
* RealSense SDK (`pyrealsense2`)
* ROS2 Foxy + Gazebo Classic
* MATLAB Simulink (with Quanser QLabs if available)

### 3. Run the System

#### ➤ YOLOv8 Detection

```bash
cd detection/
python detect.py --weights yolo.pt --source live --save-txt
```

#### ➤ ROS2 Simulation (Gazebo)

```bash
cd ros_ws/
source install/setup.bash
ros2 launch qarm_fruit_sorter main.launch.py
```

#### ➤ MATLAB Simulation

Open `simulink_models/main_model.slx` and click *Run*.

#### ➤ Teleoperation Interface

```bash
cd teleop/
python qarm_teleop.py
```

---

## 🖼️ Diagrams and Architecture

* ✅ System Architecture Diagram
* ✅ Factory Layout CAD
* ✅ Deployment Flowchart
* ✅ Grasp Angle Visualization

> See `images/` for high-res visuals

---

## 📈 Performance Metrics

* **Placement Error**: ±3.2 mm (20 cycles)
* **Average Cycle Time**: 19 seconds
* **Throughput**: 1,900 fruits/hr (scalable with 5-arm layout)
* **Energy Use**: <2 kWh per shift
* **Waste Reduction**: <2% bruised/misclassified

---


## 🧠 Future Work

* Train custom lightweight CNN for improved QA sorting
* Add soft robotic grippers for delicate fruit
* Real-time GUI for operator-friendly control
* Cloud-based fruit defect traceability system

---

## 📬 Contact

**Reuben Mathew**
📧 [pxk407@student.ac.uk](mailto:pxk407@student.bham.ac.uk)
🔗 [LinkedIn](https://www.linkedin.com/praveenkathirvel)

---

Let me know if you'd like this as a downloadable `.md` file or if you'd like to generate GitHub Pages documentation from it!
