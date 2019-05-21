# nn_with_python			

#####&emsp;&emsp;本项目使用python实现基于神经网络的mnist手写数字识别。适合初学python和神经网络或机器学习的人使用。代码中实现了完整的前向和反向传播过程，实现激活函数和softmax等及其导数，使用生成器循环读取数据。			

###说明：			

1. master分支使用softmax和交叉熵损失，ave-square-loss分支使用sigmoid和均方差损失。				

2. 两个分支的激活函数都实现了sigmoid和relu，其中relu无法收敛，留待解决。			

3. 在初始化网络时可以设置任意层数和任意神经元个数。			

4. 在此目录下新建mnist/mldata文件夹，然后从 （链接：https://pan.baidu.com/s/1E2GGXVOvpkaYSwaxo9Q68Q 
提取码：q0zl ）下载mnist数据集置于该文件夹下。			

5. 第三方库使用了numpy和sklearn，自行下载。代码在python3.6下开发。
