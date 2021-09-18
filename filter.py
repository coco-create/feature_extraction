import cv2
import numpy as np
import time
import xlwt
#写表格
book = xlwt.Workbook()
sheet = book.add_sheet(u'time1')
#参数
camera = cv2.VideoCapture(0)
gray = None
firstframe = None
filtered_frame = 0 # 被过滤的帧的数量
pixel_width = 768
pixel_height = 576
threshold = 40 # 差帧后 用于检测动态物体的阈值 ，阈值越大 检测效果越小，即更不易检测到动态物体

#读视频
capture = cv2.VideoCapture("C:\\Users\\老万选教授\\Desktop\\数据源-视频\\View_001.mp4")
output_path = 'output/filter_reducto/filtered_view_1'
#写视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = capture.get(cv2.CAP_PROP_FPS)
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
vedio_out = cv2.VideoWriter('camera_out.avi', fourcc, fps, size)

start = time.time()
if capture.isOpened():
    while True:
        ret, frame = capture.read()#如果要读取实时视频 则将此处的capture 改成 camera即可
        if not ret:
            break

        firstframe = gray   #在调用摄像头的实时视频数据时 采用每一帧与前一帧作差，采用视频数据集的时候 用每一帧与第一帧作差
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if firstframe is None:
            firstframe = gray
            continue

        frameDelta = cv2.absdiff(firstframe, gray)#两幅图的差的绝对值输出到另一幅图上面来
        thresh = cv2.threshold(frameDelta, threshold, 255, cv2.THRESH_BINARY)[1]
        #给两幅图的差异设定阈值
        thresh = cv2.dilate(thresh, None, iterations=2)
        #dilate函数可以对输入图像用特定结构元素进行膨胀操作，该结构元素确定膨胀操作过程中的邻域的形状，各点像素值将被替换为对应邻域上的最大值
        # cnts= cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #x, y, w, h = cv2.boundingRect(thresh)
        #frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #提取运动物体的轮廓 contours为物体轮廓列表
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 根据二值图找轮廓

        #计算area面积（area为检测到轮廓的每个运动物体的面积），all_area 为总面积
        all_area = 0
        for obj in contours:
            area = cv2.contourArea(obj)
            all_area += area
            print('area is ',area)
        print('all area in a frame is ', all_area)

        cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)  # 把轮廓画在原图上（0,0,255） 表示 RGB 三通道，红色

        ratio = all_area / (pixel_height * pixel_width )
        print('所占比例为'+ str(ratio)+'%')

        if ratio :
            filtered_frame += 1
            sheet.write(filtered_frame, 0, ratio)

        cv2.imshow("frame", frame)
        if ratio > 0.025:
            cv2.imshow("filtered frame ", frame)
            cv2.imwrite(output_path+'/'+str(filtered_frame)+'.jpg', frame)
            vedio_out.write(frame)

        cv2.imshow("Thresh", thresh)
        cv2.imshow("frame2", frameDelta)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

end = time.time()
area_time = end - start
print('时间为'+ str(area_time))


book.save('filter_ratio.xls')

capture.release()
cv2.destroyAllWindows()