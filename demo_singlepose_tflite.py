#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import copy
import time
import argparse

import ctypes
import cv2 as cv
import numpy as np
import tensorflow as tf

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

from pykinect2.PyKinectV2 import _CameraSpacePoint
from pykinect2.PyKinectV2 import _DepthSpacePoint

import mapper


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--mirror', action='store_true')

    parser.add_argument("--model_select", type=int, default=1)
    parser.add_argument("--keypoint_score", type=float, default=0.4)

    args = parser.parse_args()

    return args

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
frame_type = 'depth'

def get_last_depth():
    frame = kinect.get_last_depth_frame()
    frame = frame.astype(np.uint8)
    dep_frame = np.reshape(frame, [424, 512])
    return cv.cvtColor(dep_frame, cv.COLOR_GRAY2RGB)

def get_last_rbg():
    frame = kinect.get_last_color_frame()
    return np.reshape(frame, [1080, 1920, 4])[:, :, 0:3]


def run_inference(interpreter, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # 前处理
    input_image = cv.resize(image, dsize=(input_size, input_size))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB变换
    input_image = input_image.reshape(-1, input_size, input_size, 3)
    input_image = tf.cast(input_image, dtype=tf.uint8)  # 转换格式为uint8

    # 推理
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # 关键点提取
    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)

    return keypoints, scores


def main():
    # 参数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.file is not None:
        cap_device = args.file

    mirror = args.mirror
    model_select = args.model_select
    keypoint_score_th = args.keypoint_score

    # 相机画面获取 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 加载模型 #############################################################
    if model_select == 0:
        model_path = 'tflite/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite'
        input_size = 192
    elif model_select == 1:
        model_path = 'tflite/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite'
        input_size = 256
    elif model_select == 2:
        model_path = 'tflite/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite'
        input_size = 192
    elif model_select == 3:
        model_path = 'tflite/lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite'
        input_size = 256
    else:
        sys.exit(
            "*** model_select {} is invalid value. Please use 0-3. ***".format(
                model_select))

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    while True:
        start_time = time.time()

        # 自带相机 #####################################################
        # ret, frame = cap.read()
        # if not ret:
        #     break
        # if mirror:
        #     frame = cv.flip(frame, 1)  # 镜像


        # frame01 = kinect.get_last_depth_frame()
        # frame = np.reshape(frame01, [424, 512])
        # frame_test = frame1.

        frame = get_last_depth()
        debug_image = copy.deepcopy(frame)

        # 关键点预测 ##############################################################
        keypoints, scores = run_inference(
            interpreter,
            input_size,
            frame,
        )

        #实时刷新的关键点二维数据 ———————————————————————————————————————————————————————— 无深度！！！！！！！！！！
        # keypoints_str = [str(x) for x in keypoints]
        # write2txt =  open("data01.txt","w+")
        # for i in keypoints_str:
        #     write2txt.writelines(i+"\n")

        # depth2txt = open("depth01.txt","w+")
        # mapper中depth2world函数  ——————————————————————————————————————————         深度数据不变 20.000415
        # world_points = mapper.depth_points_2_world_points(kinect,_DepthSpacePoint,keypoints)
        # world_points_str = [str(x) for x in world_points]
        # for i in world_points_str:
        #     depth2txt.writelines(i+"\n")


        # ————————————————————————————————————————————————————————————————————————————————同上
        # for i in range(len(keypoints)):
        #     depth_x , depth_y = keypoints[i][0],keypoints[i][1]
        #     world_x, world_y, world_z = mapper.depth_point_2_world_point(kinect, _DepthSpacePoint, [depth_x, depth_y])
        #     depth_z = mapper.depth_space_2_world_depth(frame01,depth_x,depth_y)
        #     world_x_str = str(world_x)
        #     world_y_str = str(world_y)
        #     world_z_str = str(world_z)
        #     print(depth_z)
        '''
        # pyqt中生成点云————————————————————————————————————————————————————————
        world_points = mapper.depth_2_world(kinect, kinect._depth_frame_data, _CameraSpacePoint)
        world_points = ctypes.cast(world_points, ctypes.POINTER(ctypes.c_float))
        world_points = np.ctypeslib.as_array(world_points, shape=(512 * 424, 3))
        world_points *= 1000  # transform to mm

        # transform the point cloud to np (424*512, 3) array
        dynamic_point_cloud = np.ndarray(shape=(len(world_points), 3), dtype=np.float32)
        dynamic_point_cloud[:, 0] = world_points[:, 0]
        dynamic_point_cloud[:, 1] = world_points[:, 2]
        dynamic_point_cloud[:, 2] = world_points[:, 1]

        # if self._cloud_file[-4:] == '.txt':
        #     # remove zeros from array
        #     dynamic_point_cloud = dynamic_point_cloud[dynamic_point_cloud[:, 1] != 0]
        #     dynamic_point_cloud = dynamic_point_cloud[np.all(dynamic_point_cloud != float('-inf'), axis=1)]
        #
        # if self._cloud_file[-4:] == '.ply' or self._cloud_file[-4:] == '.pcd':
        #     # update color for .ply file only
        #     self._color = np.zeros((len(self._dynamic_point_cloud), 3), dtype=np.float32)
        #     # get color image
        #     color_img = self._color_frame.reshape(
        #         (self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4)).astype(np.uint8)
        #     color_img = color_img.reshape(
        #         (self._kinect.color_frame_desc.Height * self._kinect.color_frame_desc.Width, 4))
        #     color_img = color_img[:, :3:]  # remove the fourth opacity channel
        #     color_img = color_img[..., ::-1]  # transform from bgr to rgb
        #     # update color with rgb color
        #     self._color[:, 0] = color_img[:, 0]
        #     self._color[:, 1] = color_img[:, 1]
        #     self._color[:, 2] = color_img[:, 2]

        # write points for txt file
        row = ''.join(','.join(str(point).strip('[]') for point in xyz) + '\n' for xyz in dynamic_point_cloud)
        with open("depth02.txt", 'a') as txt_file:
            txt_file.write(row)
        '''

        elapsed_time = time.time() - start_time

        # 绘制关键点画面
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            keypoint_score_th,
            keypoints,
            scores,
        )

        # 退出(ESC) ##################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 显示画面 #############################################################
        cv.imshow('MoveNet(singlepose) Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    image,
    elapsed_time,
    keypoint_score_th,
    keypoints,
    scores,
):
    debug_image = copy.deepcopy(image)

    # 0:鼻 1:左目 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手腕
    # 10:右手腕 11:左股关节 12:右股关节 13:左膝 14:右膝 15:左足首 16:右足首
    # Line：鼻 → 左目
    index01, index02 = 0, 1
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 0, 0), 4)
        cv.line(debug_image, point01, point02, (255, 0, 0), 2)
    # Line：鼻 → 右目
    index01, index02 = 0, 2
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (0, 255, 0), 4)
        cv.line(debug_image, point01, point02, (0, 255, 0), 2)
    # Line：左目 → 左耳
    index01, index02 = 1, 3
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 0, 0), 4)
        cv.line(debug_image, point01, point02, (255, 0, 0), 2)
    # Line：右目 → 右耳
    index01, index02 = 2, 4
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (0, 255, 0), 4)
        cv.line(debug_image, point01, point02, (0, 255, 0), 2)
    # Line：鼻 → 左肩
    index01, index02 = 0, 5
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 0, 0), 4)
        cv.line(debug_image, point01, point02, (255, 0, 0), 2)
    # Line：鼻 → 右肩
    index01, index02 = 0, 6
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (0, 255, 0), 4)
        cv.line(debug_image, point01, point02, (0, 255, 0), 2)
    # Line：左肩 → 右肩
    index01, index02 = 5, 6
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (255, 255, 255), 2)
    # Line：左肩 → 左肘
    index01, index02 = 5, 7
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 0, 0), 4)
        cv.line(debug_image, point01, point02, (255, 0, 0), 2)
    # Line：左肘 → 左手腕
    index01, index02 = 7, 9
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 0, 0), 4)
        cv.line(debug_image, point01, point02, (255, 0, 0), 2)
    # Line：右肩 → 右肘
    index01, index02 = 6, 8
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (0, 255, 0), 4)
        cv.line(debug_image, point01, point02, (0, 255, 0), 2)
    # Line：右肘 → 右手腕
    index01, index02 = 8, 10
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (0, 255, 0), 4)
        cv.line(debug_image, point01, point02, (0, 255, 0), 2)
    # Line：左股关节 → 右股关节
    index01, index02 = 11, 12
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (255, 255, 255), 2)
    # Line：左肩 → 左股关节
    index01, index02 = 5, 11
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 0, 0), 4)
        cv.line(debug_image, point01, point02, (255, 0, 0), 2)
    # Line：左股关节 → 左膝
    index01, index02 = 11, 13
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 0, 0), 4)
        cv.line(debug_image, point01, point02, (255, 0, 0), 2)
    # Line：左膝 → 左足首
    index01, index02 = 13, 15
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 0, 0), 4)
        cv.line(debug_image, point01, point02, (255, 0, 0), 2)
    # Line：右肩 → 右股关节
    index01, index02 = 6, 12
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (0, 255, 0), 4)
        cv.line(debug_image, point01, point02, (0, 255, 0), 2)
    # Line：右股关节 → 右膝
    index01, index02 = 12, 14
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (0, 255, 0), 4)
        cv.line(debug_image, point01, point02, (0, 255, 0), 2)
    # Line：右膝 → 右足首
    index01, index02 = 14, 16
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (0, 255, 0), 4)
        cv.line(debug_image, point01, point02, (0, 255, 0), 2)

    # Circle：各点
    for keypoint, score in zip(keypoints, scores):
        if score > keypoint_score_th:
            cv.circle(debug_image, keypoint, 6, (255, 255, 255), -1)
            cv.circle(debug_image, keypoint, 3, (0, 0, 0), -1)

    # 处理时间
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4,
               cv.LINE_AA)
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
               cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()