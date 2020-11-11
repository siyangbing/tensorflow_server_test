import cv2
import os

img_path = "/home/db/PycharmProjects/FastAPI/tensorflow_server_test/img_src"
img_path_1600 = "/home/db/PycharmProjects/FastAPI/tensorflow_server_test/img_src_1600"

for root, folders, files in os.walk(img_path):
    # index = 0
    # t0 = time.time()
    for file in files:
        # t0 = time.time()
        img_path_one = os.path.join(root, file)
        img = cv2.imread(img_path_one)
        img = cv2.resize(img,(1600,1200))
        cv2.imwrite(os.path.join(img_path_1600, file),img)