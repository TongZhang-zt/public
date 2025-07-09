#!/usr/bin/env python3
# coding: utf-8
import Arm_Lib
from time import sleep
import numpy as np
import math
from kinemarics.inverse import inverse

class gem_grap_move():
    def __init__(self):
        # 设置移动状态
        self.move_status = True
        self.arm = Arm_Lib.Arm_Device()

        self.grap_angle = 140   #闭合
        self.release_angle = 30 #张开

        # 坐标定义
        #self.grap_origin = [0.0,0.265,0]    # 抓取原点
        self.target_pos1 = [-0.075, 0.11, 0.04] # 目标点1
        self.target_pos2 = [-0.075, 0.2, 0.04]  # 目标点2
        self.target_pos3 = [-0.155, 0.11, 0.04]
        self.target_pos4 = [-0.155, 0.2, 0.04]
        self.target_pos5 = [0.155, 0.11, 0.04]
        self.target_pos6 = [0.155, 0.2, 0.04]
        self.target_pos7 = [0.075, 0.11, 0.04]
        self.target_pos8 = [0.075, 0.2, 0.04]

    def move(self,origin,joints_down_pos):
        # 抓取后的抬起动作
        joints_grap_up = [0.0, -0.67, -0.566, -0.2989, 0, self.grap_angle]
        grap_origin=np.append(origin[0:2],[0])
        print("移动到抓取点")
        joints = inverse(grap_origin)
        joints=np.append(joints,[self.release_angle])

        self.arm.Arm_serial_servo_write6_array(self.convert_joints(joints),1000)
        sleep(1.5)
        #夹紧夹爪
        self.arm.Arm_serial_servo_write (6,self.grap_angle,1000)
        sleep(1.5)
        #提起夹爪
        self.arm.Arm_serial_servo_write6_array(self.convert_joints(joints_grap_up),1000)
        sleep(1.5)

        print("放置到指定位置")
        joints_down = inverse(joints_down_pos)
        joints_down=np.append(joints_down,[self.grap_angle])
       
        self.arm.Arm_serial_servo_write6_array(self.convert_joints(joints_down),1000)
        sleep(1.5)
        #松开夹爪
        self.arm.Arm_serial_servo_write(6,self.release_angle,1000)
        sleep(1.5)
        print("完成抓取动作")

    # 角度预处理
    def convert_joints(self,joints):
        return [
            math.degrees(joints[0]) + 90,
            math.degrees(joints[1]) + 90,
            math.degrees(joints[2]) + 90,
            math.degrees(joints[3]) + 90,
            math.degrees(joints[4]) + 90,
            joints[5]
        ]

    def arm_run(self, name, pos):
        if self.move_status:
            self.move_status = False

            try:
                # 根据宝石类型设置放置位置的角度
                if name == "hongbiyu":
                    joints_down = self.target_pos1
                elif name == "qingjinshi":
                    joints_down = self.target_pos2
                elif name == "bandianshi":
                    joints_down = self.target_pos3
                elif name == "huyanshi":
                    joints_down = self.target_pos4
                elif name == "juyanshi":
                    joints_down = self.target_pos5
                elif name == "baisongshi":
                    joints_down = self.target_pos6
                elif name == "huangshuijing":
                    joints_down = self.target_pos7
                elif name == "shamomeigui":
                    joints_down = self.target_pos8
                else:
                    print(f"未知宝石: {name}")
                    self.move_status = True
                    return
                # 执行移动过程
                self.move(pos,joints_down)

            except Exception as e:
                print(f"宝石放置位置出错{str(e)}")
            finally:
                self.move_status = True