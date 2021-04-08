# import logging
# import sys
# import os
# from datetime import datetime


# class _Logger:
#     date = datetime.now()
#     current_time = date.strftime("%H-%M-%S")

#     file = "./logs/log--{day}-{month}-{year}--{time}.log".format(
#         day=date.day,
#         month=date.month,
#         year=date.year,
#         time=current_time,
#     )

#     logging.basicConfig(
#         filename=file,
#         level=logging.INFO,
#         format="%(asctime)s %(levelname)s : %(message)s",
#     )

#     def __getattr__(self, key):
#         return getattr(logging, key)


# logger = _Logger()
