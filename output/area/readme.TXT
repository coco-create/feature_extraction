此时采用的是读取实时相机的视频流，而没有采取数据集
电脑端相机的帧率为30fps，分辨率为480*640

时间是处理每一帧画面用时


只能对运动目标进行跟踪，不能对目标种类进行检测和识别
采用pycharm 使用python-opencv 

算法基本思路：
利用两帧之间物体（area）运动后像素间的差值来框定区域，主要利用：
 #cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]     ---------给两幅图的差异设定阈值
# cv2.dilate(thresh, None, iterations=2)               --------对输入图像用特定结构元素进行膨胀操作，该结构元素确定膨胀操作过程中的邻域的形状，各点像素值将被替换为对应邻域上的最大值
# cv2.boundingRect(thresh)     ----获取目标区域的bbox

