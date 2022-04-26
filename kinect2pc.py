from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
import tensorflow

# 获取深度图, 默认尺寸 424x512
def get_last_depth():
    frame = kinect.get_last_depth_frame()
    frame = frame.astype(np.uint8)
    dep_frame = np.reshape(frame, [424, 512])
    return cv2.cvtColor(dep_frame, cv2.COLOR_GRAY2RGB)

#获取rgb图, 1080x1920x4
def get_last_rbg():
    frame = kinect.get_last_color_frame()
    return np.reshape(frame, [1080, 1920, 4])[:, :, 0:3]

# 运行模式选择读取深度和rgb
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
# 二选一使用opencv显示
frame_type = 'rgb'
while True:

    # RGB
    last_frame_rgb = get_last_rbg()
    #深度
    last_frame_depth = get_last_depth()

    # 使用opencv显示图片
    # cv2.namedWindow('rgb', 0)
    # cv2.imshow('rgb', last_frame_rgb)

    cv2.namedWindow('depth', 0)
    cv2.imshow('depth',last_frame_depth)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyWindow()


