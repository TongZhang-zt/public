# **DOFBot 矿石自动分类系统**

### **项目概述**

DOFBot 是一个基于计算机视觉和机械臂控制的智能矿石分类系统，能够自动识别、抓取并分类多种宝石。系统采用YOLOv8目标检测模型实现高精度识别，结合机械臂运动控制完成自动化操作。

### **功能特性**

多目标检测：支持8种宝石的实时检测（白松石、斑点石、红碧玉等）

精准抓取：基于逆运动学计算的机械臂控制

自动分类：根据识别结果自动分拣到指定区域

交互界面：集成Jupyter Notebook的可视化控制面板

日志系统：完备的运行日志记录和导出功能

### **系统架构**

|--dofbot_ws/

    |--dofbot_gem_yolov5/
    |    |--kinemarics/
    |    |    |--models/
    |    |    |    |--dofbot_info/ #机械臂模型
    |    |    |--inverse.py   #逆运动学求解
    |    |--models/
    |    |    |--yolov8n_detect_bayese_640x640_nv12.bin   #模型文件
    |    |    |--experimental.py
    |    |--dofbot_config.py #系统配置工具
    |    |--gem_grap.py      #机械臂抓取控制
    |    |--gem_yolov8_play.ipynb    #主控制界面
    |    |--logger.py                #日志管理系统
    |    |--XYT_config.txt           #机械臂初始位置参数

### **依赖环境**

* Python 3.8+
* OpenCV 4.5+
* NumPy
* PyEasyDNN (Horizon Robotics)
* Pinocchio (用于逆运动学计算)
* Jupyter Notebook (用于控制界面)

### **安装指南**

**1.克隆仓库：**

bash

`git clone https://github.com/TongZhang-zt/public.git
cd dofbot_gem_yolov5`

**2.安装依赖：**

bash

`pip install -r requirements.txt`

**3.部署机械臂驱动：**

Arm_Lib在py_install文件中，将文件下载打开，执行setup.py文件

bash

`python setup.py`

### **使用说明**

**1.启动Jupyter Notebook：**

bash

`jupyter notebook gem_yolov8_play.ipynb`

**2.操作流程：**

* 点击"目标检测"按钮开始识别
* 检测到目标后点击"抓取"执行分类
* 或启用"自动抓取"模式实现全自动化

**3.参数配置：**

* 修改XYT_config.txt调整机械臂初始位置
* 更新yolov8n_detect_bayese_640x640_nv12.bin替换模型

### **开发指南**

#### **扩展新宝石类型**

**1.在gem_identify.py中更新names列表：**

python

`names = ['新宝石类型', ...]`

**2.在gem_grap.py中添加对应的目标位置：**

python

`self.target_pos9 = [x, y, z]`

**3.更新抓取逻辑：**

python

`elif name == "新宝石类型":
    joints_down = self.target_pos9`

#### **模型优化**

**1.准备自定义数据集**

**2.使用YOLOv8重新训练：**

bash

`yolo detect train data=your_dataset.yaml model=yolov8n.pt`

**3.转换模型为.bin格式后替换原模型**

### **常见问题**

**Q: 机械臂无法正确抓取**

A: 检查以下方面：

1.确认inverse.py中的TOOL_FRAME_ID与URDF模型匹配

2.检查机械臂关节限位设置

3.验证摄像头标定参数准确性

**Q: 检测精度下降**

A: 尝试以下解决方案：

1.重新校准摄像头内参矩阵camera_matrix

2.增加训练数据多样性

3.调整检测置信度阈值

**Q: 日志系统报错**

A: 确保：

1.对/home/sunrise/目录有写入权限

2.Jupyter Notebook运行环境有足够权限

3.日志文件未被其他进程占用

### **贡献指南**

欢迎通过Issue或Pull Request提交改进建议，请遵循以下规范：

* 新功能开发创建feature分支
* Bug修复创建hotfix分支
* 提交信息使用英文描述
* 保持代码风格一致性

### **致谢**

* Horizon Robotics 提供的NPU加速支持
* Pinocchio 机器人动力学库
* Ultralytics YOLOv8 目标检测框架
