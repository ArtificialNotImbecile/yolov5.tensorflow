# 把YoloV5(目前支持v4.0)转换成tensorflow格式
1. 填写对应的运行 `save_yolov5_weight.py`, 当前路径下会生成`params.dict`, `meta.dict`
2. 运行`yolov5_convert.py`, 当前路径下会生成一个`yolov5.pb`文件和`result.jpg`(输入名字`input:0`，输出名字`output:0`)



# Ref
https://github.com/facerless/yolov5-tensorflow

