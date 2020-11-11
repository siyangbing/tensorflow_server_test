import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

img_path = "/home/sucom/Desktop/FastAPI/tensorflow_server_test/img_src_1600"


def send_data(img_url):
    os.system(
        'curl -X POST "http://192.168.3.124:8000/uploadfile/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@{};type=image/jpeg"'.format(
            img_url))


for root, folders, files in os.walk(img_path):
    # index = 0
    # t0 = time.time()
    img_path_list = []
    futures = []
    executor = ProcessPoolExecutor(max_workers=10)
    for file in files:
        # t0 = time.time()
        img_path_one = os.path.join(root, file)
        img_path_list.append(img_path_one)
    for x in range(len(files)):
        future = executor.submit(send_data, img_path_list[x])
        futures.append(future)

    executor.shutdown(True)

    # a = os.system('curl -X POST "http://192.168.3.41:8000/uploadfile/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@{};type=image/jpeg"'.format(img_path_one))
    # print(a)
    # # break
