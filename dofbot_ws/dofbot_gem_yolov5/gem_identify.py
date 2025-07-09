#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import cv2 as cv
from time import sleep
from numpy import random
from gem_grap import gem_grap_move
from hobot_dnn import pyeasy_dnn as dnn
import Arm_Lib
# 模型路径
model_path = '/home/sunrise/yolov8/yolov8n_detect_bayese_640x640_nv12.bin'

# 类别名称
names = ['baisongshi', 'bandianshi', 'hongbiyu',
         'huangshuijing', 'huyanshi', 'juyanshi', 'qingjinshi', 'shamomeigui']

# 生成随机颜色
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

class gem_identify():
    def __init__(self):
        self.frame = None
        self.arm = Arm_Lib.Arm_Device()
        self.xy = [90, 130]
        self.gem_index = 0
        self.grap_move = gem_grap_move()
        self.service_ready = False
        #相机内参矩阵
        self.camera_matrix=np.array([
            [967.97119525,0.0,262.01912843],
            [0.0,970.21889476,278.26639629],
            [0.0,0.0,1.0]
        ])
        self.depth=0.04 #高度

        # 机械臂参数
        self.arm_range_x = [-200, 200]  # X轴移动范围(mm)
        self.arm_range_y = [0, 400]  # Y轴移动范围(mm)
        self.image_width = 640
        self.image_height = 480

        print(f"机械臂X轴范围: {self.arm_range_x}")
        print(f"机械臂Y轴范围: {self.arm_range_y}")
        # 加载模型
        self.model = self.load_bin_model(model_path)
        print("模型加载成功")

    def print_properties(pro):
        print("tensor type:", pro.tensor_type)
        print("data type:", pro.dtype)
        print("layout:", pro.layout)
        print("shape:", pro.shape)

    def load_bin_model(self, model_path):
        """加载 bin 格式的模型"""
        try:
            print(f"开始加载模型: {model_path}")
            models = dnn.load(model_path)
            print(f"模型加载成功: {len(models)} 个模型")
            # 打印模型信息
            model = models[0]
            # 打印输入 tensor 的属性
            gem_identify.print_properties(model.inputs[0].properties)
            # 打印输出 tensor 的属性
            print(len(model.outputs))
            for output in model.outputs:
                gem_identify.print_properties(output.properties)
            # 返回加载的模型
            return model
        except Exception as e:
            print("model load fail")
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def gem_grap(self, msg, xy=None):
        print(f"进入抓取方法，消息长度: {len(msg)}")

        if xy is not None:
            self.xy = xy
            print(f"设置机械臂位置: {self.xy}")

        if len(msg) != 0:
            print("触发蜂鸣器")
            self.arm.Arm_Buzzer_On(1)
            sleep(0.5)
        print(f"开始处理 {len(msg)} 个目标")

        for index, name in enumerate(msg):
            print(f"处理目标 {index + 1}/{len(msg)}: {name}")
            try:
                # 记录坐标信息
                pos = msg[name]
                print(f"目标坐标: x={pos[0]}, y={pos[1]},z={pos[2]}")

                # 调用抓取方法
                self.grap_move.arm_run(str(name), pos)
            except Exception as e:
                print(f"抓取错误: {str(e)}")
        print("所有目标处理完成")
        joints_0 = [self.xy[0], self.xy[1], 0, 0, 90, 30]
        self.arm.Arm_serial_servo_write6_array(joints_0, 1000)
        sleep(1)
    #目标函数检测主入口
    def gem_run(self, image):
        self.frame = cv.resize(image, (640, 480))   #将图像调整为宽640高480的大小
        txt0 = 'Model-Loading...'
        msg = {}
        if self.gem_index < 3:
            cv.putText(self.frame, txt0, (190, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.gem_index += 1
            return self.frame, msg
        if self.gem_index >= 3:
            print("目标开始检测...")
            try:
                print(f"输入图像尺寸：{image.shape}")
                msg = self.get_pos()
                print(f"检测结果:{msg}")
            except Exception as e:
                print("识别错误")
            return self.frame, msg

    def get_pos(self):
        #打印模型输入尺寸
        print(f"模型输入尺寸: {self.model.inputs[0].properties.shape}")

        img = self.frame.copy()
        # 图像预处理
        h, w = self.model.inputs[0].properties.shape[2], self.model.inputs[0].properties.shape[3]
        des_dim = (w, h)
        resized_img = cv.resize(img, des_dim, interpolation=cv.INTER_AREA)

        # 添加图像增强
        resized_img = cv.convertScaleAbs(resized_img, alpha=1.2, beta=20)  # 增加对比度

        # 转换为NV12格式
        nv12_img = self.bgr2nv12_opencv(resized_img)

        # 模型推理
        outputs = self.model.forward(nv12_img)
        # 使用YOLOv8后处理
        detections = self.yolov8_postprocess(outputs, conf_threshold=0.5, iou_threshold=0.45)

        # 处理检测结果
        msg = {}
        for result in detections:
            bbox = result['bbox']
            class_id = int(result['id'])
            label = names[class_id]
            # 添加坐标转换：从模型输入尺寸(640x640)转回原始图像尺寸(640x480)
            scale_x = self.image_width / w
            scale_y = self.image_height / h
            # 转换边界框坐标
            bbox[0] = int(bbox[0] * scale_x)
            bbox[1] = int(bbox[1] * scale_y)
            bbox[2] = int(bbox[2] * scale_x)
            bbox[3] = int(bbox[3] * scale_y)

            # 坐标转换
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            #计算相机坐标系中的位置
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            z = self.depth  # 使用预设深度值

            #计算相机坐标系坐标
            x_cam = 10*(center_x - cx) * z / fx
            y_cam = 10*(center_y - cy) * z / fy
            if x_cam>0: x_cam = x_cam-0.006
            elif x_cam<0: x_cam = x_cam+0.005
            y_cam = 0.285 - y_cam   #偏移

            msg[label]=(x_cam,y_cam,z)

            """a = ((center_x - self.image_width / 2) / self.image_width) * \
                (self.arm_range_x[1] - self.arm_range_x[0])
            b = ((self.image_height - center_y) / self.image_height) * \
                self.arm_range_y[1]
            # 添加坐标范围检查
            if not (self.arm_range_x[0] <= a <= self.arm_range_x[1]) or \
                    not (self.arm_range_y[0] <= b <= self.arm_range_y[1]):
                print(f"坐标超出机械臂范围: a={a}, b={b}")
                continue  # 跳过无效坐标

            msg[label] = (a, b)"""

            print(f"检测到目标: {label} 在位置 ({x_cam}, {y_cam},{z})")
            # 在图像上绘制检测结果
            self.draw_bbox(img, bbox, label, result['score'])

        self.frame=img  #更新为带检测框的图像

        return msg

    def yolov8_postprocess(self, outputs, conf_threshold=0.5, iou_threshold=0.5):
        """
        YOLOv8 后处理函数
        参数:
            outputs: 模型输出列表
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IOU阈值

        返回:
            detections: 检测结果列表，每个元素是一个字典，包含bbox, score, id, name
        """
        detections = []
        num_classes = len(names)

        # 假设第一个输出包含检测结果 (调整根据实际模型输出)
        output = outputs[0].buffer
        output = output.squeeze(axis=(0, 3))  # 从 (1,12,8400,1) -> (12,8400)
        output = output.transpose()  # 转置为 (8400,12)

        for i in range(output.shape[0]):
            cx, cy, w, h = output[i, 0:4]
            class_scores = output[i, 4:4 + num_classes]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence > conf_threshold:
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': float(confidence),
                    'id': int(class_id),
                    'name': names[class_id]
                })

        # 应用非极大值抑制 (NMS)
        if detections:
            boxes = [d['bbox'] for d in detections]
            scores = [d['score'] for d in detections]
            indices = cv.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

            # 返回过滤后的检测结果
            return [detections[i] for i in indices]
        return []

    def draw_bbox(self, image, bbox, label, score):
        """在图像上绘制边界框和标签"""
        # 限制坐标在图像范围内
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1 = max(0, min(x1, image.shape[1]))
        y1 = max(0, min(y1, image.shape[0]))
        x2 = max(0, min(x2, image.shape[1]))
        y2 = max(0, min(y2, image.shape[0]))

        # 获取类别颜色
        class_id = names.index(label)
        color = colors[class_id]

        # 绘制边界框
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 绘制标签背景
        label_text = f"{label}: {score:.2f}"
        (text_width, text_height), baseline = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        # 标签框位置：左上角上方
        text_y = max(0, y1 - text_height - 5)  # 确保不会超出图像顶部

        cv.rectangle(image, (x1, text_y),
                     (x1 + text_width, y1), color, -1)

        # 绘制标签文本（修正位置）
        cv.putText(image, label_text, (x1, y1 - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 绘制中心点（可选）
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv.circle(image, (center_x, center_y), 4, (0, 255, 0), -1)

        # 绘制坐标文本
        coord_text = f"({center_x},{center_y})"
        cv.putText(image, coord_text, (x1, y2 + 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)


    # 将RGB图像转换为NV12格式
    @staticmethod
    def bgr2nv12_opencv(image):
        height, width = image.shape[0], image.shape[1]
        area = height * width
        yuv420p = cv.cvtColor(image, cv.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12
