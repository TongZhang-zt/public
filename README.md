# **DOFBot Ore Automatic Classification System**

### **Project Overview**

DOFBot is an intelligent ore classification system based on computer vision and robotic arm control, capable of automatically identifying, grasping, and classifying various gemstones. The system employs the YOLOv8 object detection model for high-precision recognition combined with robotic arm motion control for automated operations.

### **Functional Features**

* Multi-object Detection: Real-time detection of 8 gemstone types (White Turquoise, Spotted Stone, Red Jasper, etc.)
* Precision Grasping: Robotic arm control based on inverse kinematics calculations
* Automatic Classification: Automatic sorting into designated areas based on identification results
* Interactive Interface: Visual control panel integrated with Jupyter Notebook
* Logging System: Comprehensive runtime logging and export functionality

### **System Architecture**

|--dofbot_ws/

    |--dofbot_gem_yolov5/
    |    |--kinematics/
    |    |    |--models/
    |    |    |    |--dofbot_info/  # Robotic arm model
    |    |    |--inverse.py         # Inverse kinematics solver
    |    |--models/
    |    |    |--yolov8n_detect_bayese_640x640_nv12.bin   # Model file
    |    |    |--experimental.py
    |    |--dofbot_config.py         # System configuration utility
    |    |--gem_grap.py              # Robotic arm grasping control
    |    |--gem_yolov8_play.ipynb    # Main control interface
    |    |--logger.py                # Log management system
    |    |--XYT_config.txt           # Robotic arm initial position parameters

### **Dependencies**

* Python 3.8+
* OpenCV 4.5+
* NumPy
* PyEasyDNN (Horizon Robotics)
* Pinocchio (for inverse kinematics calculations)
* Jupyter Notebook (for control interface)

### **Installation Guide**
**1. Clone Repository:**

bash

`git clone https://github.com/TongZhang-zt/public.git
cd dofbot_gem_yolov5`

**2. Install Dependencies:**

bash

`pip install -r requirements.txt`

**3. Deploy Robotic Arm Driver:**

Arm_Lib files are in the py_install directory. After downloading, run setup.py:

bash

`python setup.py`

### **Usage Instructions**

**1. Launch Jupyter Notebook:**

bash

`jupyter notebook gem_yolov8_play.ipynb`

**2. Operation Workflow:**

* Click the "Object Detection" button to start identification
* After target detection, click "Grasp" to execute classification
* Or enable "Auto Grasp" mode for full automation

**3. Parameter Configuration:**

* Modify XYT_config.txt to adjust robotic arm initial positions
* Replace yolov8n_detect_bayese_640x640_nv12.bin to update the model

### **Development Guide**

#### **Adding New Gemstone Types**

**1. Update names list in gem_identify.py:**

python

`names = ['New_Gem_Type', ...]`

**2. Add corresponding target position in gem_grap.py:**

python

`self.target_pos9 = [x, y, z]`

**3. Update grasping logic:**

python

`elif name == "New_Gem_Type":
    joints_down = self.target_pos9`

#### **Model Optimization**
**1. Prepare custom dataset**

**2. Retrain using YOLOv8:**

bash

`yolo detect train data=your_dataset.yaml model=yolov8n.pt`

**3. Convert model to .bin format and replace original model**

### **FAQ**

**Q: Robotic arm fails to grasp correctly**

A: Check the following:

1.Verify TOOL_FRAME_ID in inverse.py matches URDF model

2.Inspect robotic arm joint limit settings

3.Validate camera calibration parameter accuracy

**Q: Detection accuracy decreases**

A: Try these solutions:

1.Recalibrate camera intrinsic matrix (camera_matrix)

2.Increase training data diversity

3.Adjust detection confidence threshold

**Q: Logging system reports errors**

A: Ensure:

1.Write permissions for /home/sunrise/ directory

2.Jupyter Notebook runtime has sufficient permissions

3.Log files aren't locked by other processes

### **Contribution Guidelines**

* Welcome improvements via Issues or Pull Requests. Please follow:
* Create feature branches for new functionality
* Create hotfix branches for bug repairs
* Use English for commit messages
* Maintain code style consistency

### **Acknowledgments**

* NPU acceleration support from Horizon Robotics
* Pinocchio robotics dynamics library
* Ultralytics YOLOv8 object detection framework