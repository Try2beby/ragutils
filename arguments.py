import argparse
import datetime
import os
import sys

# # 解析命令行参数
# parser = argparse.ArgumentParser()
# parser.add_argument("--log_filename", help="The filename of the log file.")
# args = parser.parse_args()

# # 如果提供了文件名参数，使用它，否则使用日期和时间作为文件名
# if args.log_filename:
#     log_filename = args.log_filename
# else:
#     now = datetime.datetime.now()
#     log_filename = now.strftime("%Y%m%d_%H%M%S.log")

# 获取脚本的文件名（不包括扩展名）
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# 使用脚本的文件名作为日志文件的文件名
log_filename = script_name + ".log"
