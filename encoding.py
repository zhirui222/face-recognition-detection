import os
import argparse  # 命令行模块
from retinaface import Retinaface
import logging.config

logging.config.fileConfig('logging.config')
logging = logging.getLogger("applog")


# 在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
def paramater_check(facedataset_path, npy_save_path):
    if facedataset_path is None:
        raise ValueError("未指定目标图片路径:facedataset_path")
    if not os.path.exists(npy_save_path):
        os.mkdir(npy_save_path)
    if npy_save_path is None:
        raise ValueError("未指定npy文件存储路径:npy_save_path")


def encoding(facedataset_path, npy_save_path, device):
    # 参数检测
    paramater_check(facedataset_path, npy_save_path)
    # retinaface编码人脸库
    retinaface = Retinaface(1, npy_save_path=npy_save_path, device=device)
    list_dir = os.listdir(facedataset_path)
    image_paths = []
    names = []
    for name in list_dir:
        image_paths.append(facedataset_path + "/" + name)
        names.append(name.split("_")[0])
    retinaface.encode_face_dataset(image_paths, names, npy_save_path)


if __name__ == "__main__":
    # 创建ArgumentParser对象
    try:
        parser = argparse.ArgumentParser(description='encoding the facedataset')
        # 添加参数
        parser.add_argument('-facedataset_path', '--facedataset_path', type=str, default=None,
                            help='the save folder of facedataset path')
        parser.add_argument('-npy_save_path', '--npy_save_path', type=str, default=None,
                            help='the save folder of npy files')
        parser.add_argument('-device', '--device', type=str, default="cuda",
                            help='指定显卡 "cuda:显卡编号",指定cpu "cpu"')
        # 接受传入的参数
        args = parser.parse_args()
        encoding(facedataset_path=args.facedataset_path, npy_save_path=args.npy_save_path,
                 device=args.device)
    except Exception as e:
        logging.error(e)
