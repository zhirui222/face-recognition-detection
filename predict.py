# ----------------------------------------------------#
#   对视频中的predict.py进行了修改，
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time
import argparse  # 命令行模块
import cv2
import numpy as np
import configparser
from retinaface import Retinaface
import pymysql
import base64
import os
import multiprocessing
import logging

import logging.config

logging.config.fileConfig('logging.config')
logging = logging.getLogger("applog")


class frame_object(object):
    def __init__(self, frame, number, rate):
        self.frame = frame
        self.frame_num = number
        self.rate = rate


# 将数据插入sql
def insert_value(taskId, name, appear_time, msg=None):
    con = configparser.ConfigParser()
    con.read('./config.ini', encoding='utf-8')
    section = 'SQL'
    host = con.get(section, 'host')
    port = con.getint(section, 'port')
    user = con.get(section, 'user')
    password = con.get(section, 'password')
    database = con.get(section, 'database')
    charset = con.get(section, 'charset')
    conn = pymysql.connect(host=host,
                           port=port,
                           user=user,
                           password=password,
                           database=database,
                           charset=charset,
                           )
    cursor = conn.cursor()
    try:
        # 查询表目标人物出现时间为空，子进程第一次插入数据或者主进程结束未检测到目标人物
        logging.debug("捕获目标{}出现时间{}插入数据库".format(name, appear_time))
        sql = "insert into face_identify_result_live (taskId , name,  appear_time) " \
              "VALUES (%s, %s, %s)"
        values = (taskId, name, appear_time)
        cursor.execute(sql, values)
    except Exception as e:
        logging.error(f"MYSQL ERROR: {e} with sql: {sql}")
    conn.commit()
    conn.close()


# 参数检验
def paramater_check(video_path, npy_target_path, taskId):
    if npy_target_path == None:
        raise ValueError("未指定npy文件存储路径npy_target_path")
    if video_path == None:
        raise ValueError("未指定视频文件地址video_path")
    if taskId == None:
        raise ValueError("未指定taskId")


def image_compress_encoding(image):
    # 等比例压缩
    height, width, channel = image.shape
    size_decrease = (int(width / 2), int(height / 2))
    image = cv2.resize(image, size_decrease, interpolation=cv2.INTER_AREA)
    image_encode = cv2.imencode('.jpg', np.asarray(image))[1]
    target_image = str(base64.b64encode(image_encode))[2:-1]
    return target_image


def video_spilt(taskId, queue, stop_event, error_event, video_path):
    try:
        logging.debug("进程{}拆分视频".format(os.getpid()))
        capture = cv2.VideoCapture(video_path)
        # 视频保存到本地
        """if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)"""
        # rate 播放速率
        ref, frame = capture.read()
        rate = capture.get(5)
        if not ref:
            raise ValueError("未能正确读取视频/摄像头,请检查地址")
        frame_num = 0  # frame_num 总帧数
        while True:
            if error_event.is_set():
                logging.info("进程{}检测到错误标志,拆分进程结束".format(os.getpid()))
                break
            for i in range(5):
                if i == 0:
                    ref, frame = capture.read()
                    frame_o = frame_object(frame, frame_num, rate)
                    if not ref:
                        return
                    if error_event.is_set():
                        logging.info("进程{}检测到错误标志,拆分进程结束".format(os.getpid()))
                        return
                    queue.put(frame_o)
                else:
                    ref, frame = capture.read()
                    if not ref:
                        return
                frame_num += 1
        logging.debug("进程{}图像拆分完毕".format(os.getpid()))
        stop_event.set()
        capture.release()
    except Exception as e:
        error_event.set()
        logging.error(e)
        task_status = -1
        wrong_message = str(e)
        insert_value(taskId=taskId, name=None, appear_time=None)


def predict_function(taskId, queue, stop_event, error_event, mode, video_path=None,
                     facenet_threhold=0.9, npy_target_path=None, device=False,
                     ):
    try:
        logging.debug("子进程{}启动".format(os.getpid()))
        hit = 0
        index = 0
        logging.debug("npy:{}".format(npy_target_path))
        paramater_check(video_path, npy_target_path, taskId)
        retinaface = Retinaface(npy_save_path=npy_target_path, device=device)
        # ----------------------------------------------------------------------------------------------------------#
        #   mode用于指定测试的模式：
        #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
        #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
        #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
        #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
        # ----------------------------------------------------------------------------------------------------------#
        # ----------------------------------------------------------------------------------------------------------#
        #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
        #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
        #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
        #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
        #   video_fps用于保存的视频的fps
        #   video_path、video_save_path和video_fps仅在mode='video'时有效
        #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
        # ----------------------------------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------#
        #   test_interval用于指定测量fps的时候，图片检测的次数
        #   理论上test_interval越大，fps越准确。
        # -------------------------------------------------------------------------#
        # -------------------------------------------------------------------------#
        #   video_path指定了用于检测的图片的文件夹路径
        #   dir_save_path指定了检测完图片的保存路径
        #   video_path和dir_save_path仅在mode='dir_predict'时有效
        # -------------------------------------------------------------------------#

        if mode == "predict":
            '''
            predict.py有几个注意点
            1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用cv2.imread打开图片文件进行预测。
            2、如果想要保存，利用cv2.imwrite("img.jpg", r_image)即可保存。
            3、如果想要获得框的坐标，可以进入detect_image函数，读取(b[0], b[1]), (b[2], b[3])这四个值。
            4、如果想要截取下目标，可以利用获取到的(b[0], b[1]), (b[2], b[3])这四个值在原图上利用矩阵的方式进行截取。
            5、在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
            '''

            image = queue.get()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            r_image, names = retinaface.detect_image(image, facenet_threhold)
            save_image = r_image
            r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)

            # cv2.imshow("after", r_image)
            # cv2.imwrite(dir_save_path, r_image)
            for name in names:
                if name != "Unknown":
                    name = name
                    appear_time = 0
                    hit = 1
                    target_image = image_compress_encoding(r_image)
                    stop_event = True
                    break
        elif mode == "video":
            """if video_save_path != "":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)"""
            # rate 播放速率
            while True:
                if error_event.is_set():
                    logging.info("进程{}检测到错误标志,检测进程结束".format(os.getpid()))
                    break
                if stop_event.is_set():
                    logging.info("进程{}检测到停止标志目标已捕获,检测进程结束".format(os.getpid()))
                    break
                frame_object = queue.get(True, 10)
                frame = frame_object.frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 进行检测
                frame, names = retinaface.detect_image(frame, facenet_threhold)
                frame = np.array(frame)
                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                for name in names:
                    if name != "Unknown":
                        task_status = 1
                        # cv2.imwrite(dir_save_path, frame)
                        hit = 1
                        # 将文件由cv2数组编码为字符串
                        # target_image = image_compress_encoding(frame)
                        appear_time = time.time()
                        logging.info("进程{}捕获目标人物{},时间{}s".format(os.getpid(), name, appear_time))
                        insert_value(taskId, name, appear_time)
                        break
                # 展示视频
                # cv2.imshow("video", frame)
                # cv2.waitKey(1) & 0xff
            logging.debug("子进程{}结束".format(os.getpid()))
        else:
            raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
    except Exception as e:
        logging.error(e)
        error_event.set()
        task_status = -1
        wrong_message = str(e)
        insert_value(taskId=taskId, name=None, appear_time=None)


if __name__ == "__main__":
    # 创建ArgumentParser对象
    try:
        task_status = 0
        st = time.time()
        parser = argparse.ArgumentParser(description='start the face recognition')
        # 添加参数
        parser.add_argument('-video_path', '--video_path', type=str, default=None,
                            help='This is the video_path or image path')
        parser.add_argument('-taskId', '--taskId', type=str, default=None, help='This is taskId')
        parser.add_argument('-mode', '--mode', type=str, default="video", help='mode include video/predict')
        parser.add_argument('-facenet_threhold', '--facenet_threhold', type=float, default=0.9,
                            help='facenet_threhold high:0.5 middle:0.7 low:0.9')
        parser.add_argument('-npy_target_path', '--npy_target_path', type=str, default=None, help='the path of npy')
        parser.add_argument('-device', '--device', type=str, default="cuda", help='指定显卡 "cuda:显卡编号",指定cpu "cpu"')
        # 接受传入的参数
        args = parser.parse_args()
        logging.info(
            "任务id{},清晰度阈值{},设备{}".format(args.taskId, args.facenet_threhold, args.device))
        manager = multiprocessing.Manager()
        queue = manager.Queue(200)
        stop_event = multiprocessing.Event()
        error_event = multiprocessing.Event()
        split_pro = multiprocessing.Process(target=video_spilt,
                                            args=(args.taskId, queue, stop_event, error_event, args.video_path))
        split_pro.start()
        workers = []
        worker = multiprocessing.Process(target=predict_function,
                                         args=(args.taskId,
                                               queue, stop_event, error_event, args.mode, args.video_path,
                                               args.facenet_threhold, args.npy_target_path, args.device))
        worker.start()
        split_pro.join()
        logging.debug("视频拆分完毕")
        worker.join()
        et = time.time()
        logging.debug("总预测时间{}".format(et - st))
        logging.debug(stop_event.is_set())
        # if not error_event.is_set() and not stop_event.is_set():
        #     logging.info("任务{},未找到目标,预测结束".format(taskId))
        #     task_status = 1
        #     logging.debug(stop_event.is_set())
        #     insert_value(taskId=taskId, name=None, appear_time=None)
        logging.info("Complete the task")
    except Exception as e:
        task_status = -1
        wrong_message = str(e)
        logging.error("Program error:{}".format(wrong_message))
        insert_value(taskId=args.taskId, name=None, appear_time=None)
