import logging
import os
import time

def facelog_config(logger,taskId):
    #创建目录
    log_name = str(time.strftime('%Y-%m-%d'))
    #绝对路径
    # current_path=os.getcwd()
    # log_path='log/{}'.format(log_name)
    # log_path = os.path.join(current_path,log_path)
    #相对路径
    log_path='log/{}'.format(log_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    file_path = '{}/{}.log'.format(log_path, str(taskId))

    # 设置日志权限
    logger.setLevel(logging.DEBUG)
    # 日志文件路径
    fh = logging.FileHandler(file_path)  # 生成文件处理器对象
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()  # 生成控制台处理器对象
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s|%(levelname)8.5s|%(filename)15s|%(lineno)s|%(message)s',
                                  datefmt="%Y-%m-%d %H:%M:%S")  # 设置日志模式
    fh.setFormatter(formatter)  # 添加模式
    sh.setFormatter(formatter)  # 添加模式
    logger.addHandler(fh)  # 添加文件处理器
    logger.addHandler(sh)  # 添加控制台处理器
