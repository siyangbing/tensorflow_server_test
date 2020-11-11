import cv2
import math
# from lxml import etree, objectify
import os
import numpy as np
import tensorflow as tf
import cv2
import time

"""
此脚本的输入为labelimg标记好的xml文件夹、原始图片文件夹
输出为裁剪后的图片文件夹、新生成的图片文件夹

"""

import time
def echoRuntime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        endTime = time.time()
        msecs = (endTime - startTime)
        print(func.__name__+" running time is %.2f s" %msecs)
        return result
    return wrapper

# 原始图片的缩放比例
resize_shape = 0.5
# 原始img文件夹路径
img_path = "/home/db/PycharmProjects/FastAPI/tensorflow_server_test/img_src"
# 裁剪的大小
crop_size = (480, 480)
# 重合的边界
border = 90

saved_model_dir = "/home/db/PycharmProjects/FastAPI/tensorflow_server_test/saved_model"

result_img_path = "/home/db/PycharmProjects/FastAPI/tensorflow_server_test/img_result"


class CropImageLabel():
    """
    把大分辨率的图片裁剪成小分辨率的图片
    """
    # @echoRuntime
    def __init__(self, img_path, crop_size, border, resize_shape, interpreter):
        self.interpreter = interpreter
        self.img_path = img_path
        self.crop_size = crop_size
        self.border = border
        self.img = cv2.imread(img_path)
        self.resize_shape = resize_shape
        self.img = cv2.resize(self.img,
                              (int(self.img.shape[1] * self.resize_shape), int(self.img.shape[0] * self.resize_shape)))

    # @echoRuntime
    def mat_inter(self, box1, box2):
        # 判断两个矩形是否相交
        # box=(xA,yA,xB,yB)
        # a = box1
        # b = box2
        # print(a[0], a[1], a[2], a[3], a[4])
        # print(b)
        x01 = box1[0]
        y01 = box1[1]
        x02 = box1[2]
        y02 = box1[3]
        score_1 = box1[4]

        x11 = box2[0]
        y11 = box2[1]
        x12 = box2[2]
        y12 = box2[3]
        score_2 = box2[4]
        # x11, y11, x12, y12, score = box2

        lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
        ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
        sax = abs(x01 - x02)
        sbx = abs(x11 - x12)
        say = abs(y01 - y02)
        sby = abs(y11 - y12)
        if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
            return True
        else:
            return False

    # @echoRuntime
    def solve_coincide(self, box1, box2):
        # box=(xA,yA,xB,yB)
        # 计算两个矩形框的重合度
        if self.mat_inter(box1, box2) == True:
            x01 = box1[0]
            y01 = box1[1]
            x02 = box1[2]
            y02 = box1[3]
            score_1 = box1[4]

            x11 = box2[0]
            y11 = box2[1]
            x12 = box2[2]
            y12 = box2[3]
            score_2 = box2[4]



            # x01, y01, x02, y02 = box1
            # x11, y11, x12, y12 = box2
            col = min(x02, x12) - max(x01, x11)
            row = min(y02, y12) - max(y01, y11)
            intersection = col * row
            area1 = (x02 - x01) * (y02 - y01)
            area2 = (x12 - x11) * (y12 - y11)
            coincide = intersection / (area1 + area2 - intersection)
            return coincide
        else:
            return False

    # @echoRuntime
    def tuiduan_img(self, img, sess1):
        # 深度学习模型推断图像结果
        # t3 = time.time()
        # cv2.imshow('po',img)
        # cv2.waitKey(0)
        input = sess1.graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = sess1.graph.get_tensor_by_name('detection_boxes:0')
        detection_score = sess1.graph.get_tensor_by_name('detection_scores:0')
        detection_classes = sess1.graph.get_tensor_by_name('detection_classes:0')
        num_detections = sess1.graph.get_tensor_by_name('num_detections:0')

        img_array = np.array(img, dtype=float)
        img_array = img_array[np.newaxis, :, :, :]
        input_data = np.array(img_array, dtype=np.float32)

        feed_dict = {input: input_data, }
        y = sess1.run([detection_boxes, detection_score, detection_classes, num_detections], feed_dict=feed_dict)

        # input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details()
        # img_array = np.array(img, dtype=float)
        # img_array = img_array[np.newaxis, :, :, :]
        # input_data = np.array(img_array, dtype=np.float32)
        # input_data = ((input_data / 127.5) - 1)
        # t4 = time.time()
        # print("图像预处理时间{}".format(t4-t3))
        # print(type(input_data[0][0][0][0]))
        # interpreter.set_tensor(input_details[0]['index'], input_data)
        # t3 = time.time()
        # # interpreter.invoke()
        # t4 = time.time()
        # print("图像推断时间{}".format(t4 - t3))

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        location_list = y[0]
        score_list = y[1]
        class_list = y[2]


        points_list = []
        # for x in range(len(output_data2[0])):
        for x in range(20):
            if score_list[0][x] > 0.6 and class_list[0][x] == 1.0:
                # print(output_data2[0][x], output_data1[0][x])
                point = location_list[0][x] * self.crop_size[1]
                score = score_list[0][x]
                points_list.append([point[1], point[0], point[3], point[2], score])
        # points_list = []
        # print(points_list)
        return points_list

    # @echoRuntime
    def del_iou_boxes(self, pingjie_points_list):
        # 删除多余的重合边框
        result_list = pingjie_points_list
        while 1:
            pingjie_points_list = result_list
            if_ok = 1
            for point1_1 in pingjie_points_list:
                for point2_2 in pingjie_points_list:
                    if point1_1 != point2_2:
                        if self.solve_coincide(point1_1, point2_2) > 0.2:
                            # 如果重合面大于0.2就只保留得分高的检测框，删除得分低的检测框
                            if_ok = 0
                            if point1_1[4] > point2_2[4]:
                                result_list.remove(point2_2)
                            else:
                                if point1_1 in result_list:
                                    result_list.remove(point1_1)
                                else:
                                    pass
                                    # print(point1_1)
                                    # print(result_list)
            if if_ok:
                return result_list
            else:
                continue
        return result_list

    @echoRuntime
    def pingjie(self):
        # 裁剪图片
        h, w = self.img.shape[:2]
        h_num = math.floor((h - self.crop_size[1]) / (self.crop_size[1] - self.border)) + 2
        w_num = math.floor((w - self.crop_size[1]) / (self.crop_size[1] - self.border)) + 2

        img = cv2.copyMakeBorder(self.img, 0, h_num * self.crop_size[0] + self.border - h, 0,
                                 w_num * self.crop_size[1] + self.border - w, cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])

        h, w = img.shape[:2]

        # print(h_num, w_num)
        pingjie_points_list = []
        for x_l in range(h_num):
            for y_l in range(w_num):
                # print(x_l, y_l)
                x_min = x_l * (self.crop_size[0] - self.border)
                x_max = x_min + self.crop_size[0]
                y_min = y_l * (self.crop_size[1] - self.border)
                y_max = y_min + self.crop_size[1]

                if x_max >= h:
                    x_max = h
                if y_max >= w:
                    y_max = w

                img_result = img[x_min:x_max, y_min:y_max]
                # img_result_path = self.result_path_img + self.img_path.split('/')[-1].split('.')[0] + "_{}".format(
                #     str(x_l) + "_" + str(y_l)) + ".jpg"

                points_list = self.tuiduan_img(img_result, self.interpreter)
                if points_list:
                    for point in points_list:
                        # 计算boxes对应的原图位置
                        # px_min = point[0] / (self.crop_size[0] - self.border)
                        # py_min = point[1] / (self.crop_size[1] - self.border)
                        # px_max = point[2] + px_min
                        # py_max = point[3] + py_min

                        px_min = point[0] + y_l*(self.crop_size[0] - self.border)
                        py_min = point[1] + x_l*(self.crop_size[1] - self.border)
                        px_max = point[2] + y_l*(self.crop_size[0] - self.border)
                        py_max = point[3] + x_l*(self.crop_size[1] - self.border)

                        pingjie_points_list.append([px_min, py_min, px_max, py_max, point[4]])

        result_list_points = self.del_iou_boxes(pingjie_points_list)

        for last_point in result_list_points:
            # 绘制最终拼接的检测结果
            cv2.rectangle(self.img, (int(last_point[0]), int(last_point[1])), (int(last_point[2]), int(last_point[3])),
                          (0, 255, 0), 1, 8)
            cv2.putText(self.img, str(last_point[4])[:6], (int(last_point[0]), int(last_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        # return self.img
        cv2.imshow("result", self.img)
        cv2.waitKey(0)



if __name__ == "__main__":
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess1:

        # load pb
        meta_graph_def = tf.saved_model.loader.load(sess1, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        # get 输入输出
        # input = sess1.graph.get_tensor_by_name('image_tensor:0')
        # detection_boxes = sess1.graph.get_tensor_by_name('detection_boxes:0')
        # detection_score = sess1.graph.get_tensor_by_name('detection_scores:0')
        # detection_classes = sess1.graph.get_tensor_by_name('detection_classes:0')
        # num_detections = sess1.graph.get_tensor_by_name('num_detections:0')

        # 遍历文件夹内的文件，逐个裁剪
        for root, folders, files in os.walk(img_path):
            # index = 0
            t0 = time.time()
            for file in files:
                t0= time.time()
                img_path_one = os.path.join(root, file)
                # xml_path_one = os.path.join(xml_path, file.split('/')[-1].split('.')[-2] + ".xml")
                # print(img_path_one)
                # print(xml_path_one)
                crop_image_label = CropImageLabel(img_path_one, crop_size,
                                                  border, resize_shape, sess1)
                result_img = crop_image_label.pingjie()
                cv2.imwrite("{}/{}".format(result_img_path,img_path_one.split("/")[-1]),result_img)
                t1 = time.time()
                print("每张图片用时{}秒".format(t1-t0))






# saved_model_dir = "/home/sucom/dbing/tensor_offical/models-master/research/pingjie_test/saved_model"
# img_dir = "/home/sucom/dbing/tensor_offical/models-master/research/pingjie_test/20200414_141751_1_2.jpg"
# img = cv2.imread(img_dir)
# img = cv2.resize(img, (480, 480))
# img_array = np.array(img,dtype=float)
# img_array = img_array[np.newaxis,:,:,:]
# input_data = np.array(img_array, dtype=np.float32)
# input_data = ((input_data/127.5)-1)

# with tf.Session() as sess1:
#   # load pb
#     meta_graph_def = tf.saved_model.loader.load(sess1, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
#     # get 输入输出
#     input = sess1.graph.get_tensor_by_name('image_tensor:0')
#     detection_boxes = sess1.graph.get_tensor_by_name('detection_boxes:0')
#     detection_score = sess1.graph.get_tensor_by_name('detection_scores:0')
#     detection_classes = sess1.graph.get_tensor_by_name('detection_classes:0')
#     num_detections = sess1.graph.get_tensor_by_name('num_detections:0')
#     #
#     feed_dict = {input: input_data,}
#     y=sess1.run([detection_boxes,detection_score,detection_classes,num_detections], feed_dict=feed_dict)
#     print(y[0])
#     print(y[1])
#     print(y[2])
#     print(y[3])
#
#     # print(y[0])
#     # print(np.exp(y[0]/6))
