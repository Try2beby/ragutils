from loguru import logger
import sys
import os
from .arguments import log_filename
import dotenv
from rich.console import Console
import torch


_device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


def get_device():
    return _device


def set_device(num: int):
    global _device
    _device = torch.device(f"cuda:{num}" if torch.cuda.is_available() else "cpu")


CONSOLE = Console()
VERBOSE = True
USE_FIRST_N_DATA = 5000
# QA_DATASET = "./data/qa_squad_dataset_sub_uuid.json"
LOG_DIR = "./log"
DATA_DIR = "./data"
STORE_DIR = "./store"

dotenv.load_dotenv(dotenv.find_dotenv())

IS_JUPYTER = "ipykernel_launcher" in sys.argv[0]

# 设置日志级别
logger.remove()
logger.add(sys.stderr, level="INFO")

# 创建一个处理器，将日志输出到文件
logger.add(os.path.join(LOG_DIR, log_filename), level="INFO")
