LLaDA_DRFT
==
Python 3.11.0

cuda 12.4

需要对mteb库中的llm2vec文件进行修改。需要修改mteb库中的测试数据集加载方式。

不直接使用llm2vec库，而是进行了魔改。

训练权重、推理测试结果的输出路径不在该文件夹下。需要修改。

cache/echo-data和cache/echo-data-50k被git忽略了。需要手动添加。
