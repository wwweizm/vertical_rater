2024/12/22
1.通过官网安装freemocap，如有conda环境直接pip install freemocap(或通过freemocap main压缩包安装)
2.找到SimplX官网下载模型参数，导入freemocap
3.导入对应的配置文件(.json)
4.在终端运行 freemocap -m 唤起图形界面并完成视频的捕捉
5.将trajectory的csv文件复制到根目录下，名称保存为trajectory.csv
6.运行data_process.py结果会直接输出