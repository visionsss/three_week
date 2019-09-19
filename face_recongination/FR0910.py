# %% 一、人脸采集
import cv2
import matplotlib.pyplot as plt
import os
import time

numpic = 64  # 拍照的数量
someone = 'yanghui'  # 拍摄的人名


def take_piture(someone, numpic, path='../image'):
    '''
    :param someone: 拍照人名
    :param numpic: 拍照数量
    :param path: 路径
    :return: None
    '''
    # 保存路径设置
    file_path = os.path.join(path, someone)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    capture = cv2.VideoCapture(0)  # 调用摄像头
    time.sleep(2)  # 让摄像头预热2秒，以免出现曝光不足的情况
    for i in range(numpic):
        _, img = capture.read()
        cv2.imwrite(f'{file_path}/tmp{i}.jpg', img,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])  # 保存的jpg文件质量，默认95（0-100）
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.show()
    capture.release()  # 释放摄像头


take_piture(someone, numpic, path='../image')
# %% 二、人脸检测
from mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector

size = 64  # 图片大小
load_path = '../image'  # 读取的路径
save_path = '../gray'  # 灰度图保存路径


def get_gray(save_path, load_path, size, someone):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_names = os.listdir(load_path)
    detector = MtcnnDetector(model_folder='./mxnet_mtcnn_face_detection/model')
    for f in file_names:
        tmp_path = os.path.join(load_path, f)
        for t in os.listdir(tmp_path):
            img = cv2.imread(os.path.join(tmp_path, t))
            res = detector.detect_face(img)  # 如果图片没有人，会返回None
            if res is not None:  # img里面必须有检测到人
                faceboxes, points = res
                plt.imshow(img[:, :, [2, 1, 0]])
                for i in range(len(faceboxes)):
                    x1, y1, x2, y2, score = faceboxes[i]
                    # plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1])
                    # plt.scatter(points[i, : 5], points[i, 5:], marker='D', c='r')
                    # plt.show()
                    face = img[int(y1): int(y2 + 1), int(x1): int(x2 + 1)]  # 取出人脸区域图像
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # 灰度处理
                    face = cv2.resize(face, (size, size))  # 统一数据大小
                    cv2.imwrite(os.path.join(save_path, someone, t),
                                face, [cv2.IMWRITE_JPEG_QUALITY, 100])  # 数据保存
                    plt.imshow(face, cmap='gray')
                    plt.show()


get_gray(save_path, load_path, 64, 'yanghui')
# %% 三、建模前预处理
# %% 四、模型构建
# %% 五、模型训练
# %% 六、模型测试
