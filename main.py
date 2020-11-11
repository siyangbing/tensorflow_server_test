from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import time
import cv2

from tensorflow_server_test.pingjie_class import CropImageLabel



# # 原始图片的缩放比例
# resize_shape = 0.5
# # 原始img文件夹路径
# # img_path_one = "/home/db/PycharmProjects/FastAPI/tensorflow_server_test/img_src"
# # 裁剪的大小
# crop_size = (480, 480)
# # 重合的边界
# border = 90
#
# saved_model_dir = "/home/db/PycharmProjects/FastAPI/tensorflow_server_test/saved_model"
#
# result_img_path = "/home/db/PycharmProjects/FastAPI/tensorflow_server_test/img_result"
#
#
app = FastAPI()
#
# config = tf.ConfigProto(allow_soft_placement=True)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# config.gpu_options.allow_growth = True
#
# with tf.Session(config=config) as sess1:
#
#     meta_graph_def = tf.saved_model.loader.load(sess1, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

# 原始图片的缩放比例
resize_shape = 1.0
# 原始img文件夹路径
# img_path_one = "/home/db/PycharmProjects/FastAPI/tensorflow_server_test/img_src"
# 裁剪的大小
crop_size = (480, 480)
# 重合的边界
border = 90

saved_model_dir = "/home/sucom/Desktop/FastAPI/tensorflow_server_test/saved_model"

result_img_path = "/home/sucom/Desktop/FastAPI/tensorflow_server_test/img_result"

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
with sess.as_default():
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)

    @app.post("/uploadfile/")
    async def create_upload_file(file: UploadFile = File(...)):
        t1 = time.time()
        contents = await file.read()
        saved_img_path = "/home/sucom/Desktop/FastAPI/tensorflow_server_test/receive_files/{}".format(file.filename)
        with open(saved_img_path,'wb') as f:
            f.write(contents)

        # t1 = time.time()
        # for x in range(100):
        #     t1 = time.time()
        print("传输一张图片1600*1200需要时间 {}".format(time.time()-t1))
        crop_image_label = CropImageLabel(saved_img_path, crop_size,
                                      border, resize_shape, sess)
        point_list = crop_image_label.pingjie()
        point_list1 = [(float(x2) for x2 in x1) for x1 in point_list]
        t2 = time.time()-t1
        print("处理一张图片时间：{}".format(t2))
        # print(t2,point_list)


        # print(point_list)
        # show_img = cv2.imread(saved_img_path)
        # for last_point in point_list:
        #     # 绘制最终拼接的检测结果
        #     cv2.rectangle(show_img, (int(last_point[0]), int(last_point[1])), (int(last_point[2]), int(last_point[3])),
        #                   (0, 255, 0), 1, 8)
        #     cv2.putText(show_img, str(last_point[4])[:6], (int(last_point[0]), int(last_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        #
        # # cv2.imshow("result",show_img)
        # cv2.imwrite("/home/db/PycharmProjects/FastAPI/tensorflow_server_test/img_result/"+file.filename,show_img)

        # cv2.waitKey(0)




        return {"point_list": point_list1}