# 1、基本环境
    Python版本： Python 3.10.12
    cuda版本： 12.2
    cudnn： 8.3.1
    显卡： RTX-3090-24G
    安装依赖包：
        pip install -r requirements.txt

# 2、下载预训练模型，放置在pretrain_models目录下
    bge-large-en-v1.5 下载链接:https://huggingface.co/BAAI/bge-large-en-v1.5/tree/main
    bge-reranker-v2-minicpm-layerwise 下载链接:https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise/tree/main
    NV-Embed-v1 下载链接:https://huggingface.co/nvidia/NV-Embed-v1/tree/main

# 3、下载微调好的预训练模型，解压放置在output目录下
    链接：https://pan.baidu.com/s/1pePtt0UgW0VCn0Ocrq3zLA?pwd=nref  提取码：nref

# 4、训练脚本
    ① 下载官方数据集放置data目录下
    ② sh train.sh

# 5、推理脚本
    sh predict.sh
    最终的结果为result/result.txt
   
# 6、方法介绍
    ① 构造数据，利用bge-large-en-v1.5进行微调
    ② 利用微调的模型进行召回topN, 然后利用排序模型进行重排
    ③ 排序模型重排的排名融合

# 7、若过程有疑问/问题，请联系：1506025911@qq.com