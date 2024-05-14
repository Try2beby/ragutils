import torch
import sys
import os
from .arguments import log_filename
import dotenv
from rich.console import Console


_device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


def get_device():
    return _device


def set_device(num: int):
    global _device
    _device = torch.device(f"cuda:{num}" if torch.cuda.is_available() else "cpu")


CONSOLE = Console()
VERBOSE = True
USE_FIRST_N_DATA = 5000
QA_DATASET = "./data/qa_squad_dataset_sub_uuid.json"
LOG_DIR = "./log"
DATA_DIR = "./data"
STORE_DIR = "./store"


import logging

# 获取或创建一个logger
logger = logging.getLogger("my_logger")

# 设置日志级别
logger.setLevel(logging.INFO)

# 创建一个处理器，将日志输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# 创建一个处理器，将日志输出到文件
fh = logging.FileHandler(os.path.join(LOG_DIR, log_filename))
fh.setLevel(logging.INFO)

# 定义日志格式
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s"
)

# 设置处理器的格式
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# 添加处理器到logger
logger.addHandler(ch)
logger.addHandler(fh)


dotenv.load_dotenv(dotenv.find_dotenv())

IS_JUPYTER = "ipykernel_launcher" in sys.argv[0]
