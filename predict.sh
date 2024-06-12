#!/bin/bash

# 1、bge+minicpm召回重排
python3 predict-reranker-minicpm.py

# 2、bge+NV召回重排
python3 predict-reranker-nv-embed-v1.py

# 3、融合排序
python3 merge.py