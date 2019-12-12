# 三维重建大作业

## 简介
探究和实现了头部追踪以及人脸三维重建的算法，其中头部追踪采用了YOLO+deep SORT，三维重建使用了3DDFA网络。实现简单的换脸功能。

## 环境要求
* Python 3.7 (numpy, opencv, scipy, pillow, pytorch, pycuda)
* GPU, cuda

推荐使用Anaconda：
`conda env create -f environment.yml`
`conda activate 3df`
即可配置完毕

## 运行方式
`python change_face.py <input_path> <output_path>`
比如可以测试`python change_face example.mp4 output.mp4`
根目录下的output.mp4是已经生成好的视频

## 速度
目前在GTX XP上约10FPS，可优化

## 预训练数据
Data目录下为所有模型数据