视频人脸算法检测

## 1视频人脸算法检测



识别视频中的人脸，如果与人脸库中的匹配，将结果插入数据库

Retinaface：识别视频中的目标人物，并用框标记出来

Facenet：将人脸编码，与人脸库人脸编码对比，并给出分数

 

**算法运行顺序：**

encoding.py-->retinaface.py-->初始化，generate方法， encode\_face\_dataset方法，返回两个npy文件（内容为人脸库中人脸编码）

 运行encoding.py，对face\_dataset里面的图片进行编码，face\_dataset的命名规则为XXX\_1.jpg、XXX\_2.jpg。最终在model\_data文件夹下生成对应的数据库人脸编码数据文件。

predict.py-->retinaface.py-->初始化，generate方法, detect\_image方法，将视频中检测到的人脸与npy文件编码比对，返回检测到的图片，人物

**注意事项**

本项目自带主干为mobilenet的retinaface模型与facenet模型。可以直接运行，如果想要使用主干为resnet50的retinafa和主干为inception\_resnetv1的facenet模型需要。

该库中包含了两个网络，分别是retinaface和facenet。二者使用不同的权值。     在使用网络时一定要注意权值的选择，以及主干与权值的匹配。  &#x20;



**运行命令/配置：**

编码：python encoding.py -facedataset\_path \<filename>/face\_dataset  -npy\_save\_path \<filename>/model\_data -device cuda:0

\-facedataset\_path 人脸图片文件夹地址

\-npy\_save\_path    保存的人脸编码文件夹地址

\-device           指定使用的设备 可选 cpu ，cuda， cuda：n 指定第n个显卡运行

预测：python predict.py -video\_path  \<filename>/zjl.mp4 -taskId 1 -mode video -facenet -threhold 0.7 -npy\_target\_path \<filename>/model\_data  -num\_worker 8 -device cuda:4

\-video\_path 待检测视频地址

\-taskId 指定任务id

\-mode 指定mode 为 video

\-threhold 为检测阈值，推荐0.5-0.9 阈值越高容易检测到目标人物，同时容易误报，阈值越低更难检测到目标人物，但不容易误报

\-npy\_target\_path 读取的人脸编码文件夹地址

\- num\_worker 指定要开启的进程数

\-device 指定使用的设备 可选 cpu ，cuda， cuda：n 指定第n个显卡运行   

&#x20;参考

<https://github.com/biubug6/Pytorch_Retinaface>

<https://github.com/bubbliiiing/facenet-pytorch>



