import cv2
import time
import xlwt
#写表格
book = xlwt.Workbook()
sheet = book.add_sheet(u'time1')

#视频初始化
gray = None
firstframe = cv2.imread('input/images/bkgrd/v1_bg.jpg')
firstframe = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
firstframe = cv2.GaussianBlur(firstframe, (21, 21), 0)#要做模糊处理（以忽略细微处的变化） 不然在检测一些细微变化时 会影响很大

threshold = 25 # 差帧后 用于检测动态物体的阈值 ，阈值越大 检测效果越小，即更不易检测到动态物体

#读视频
camera = cv2.VideoCapture(0)
capture = cv2.VideoCapture("input/video/temporal-spatial/View_001.mp4")
#output_path = 'output/filter_reducto/filtered_view_1'
#input_path = 'output/filter_reducto/filtered_view_1'


if capture.isOpened():
    while True:
        ret, frame = capture.read()#如果要读取实时视频 则将此处的capture 改成 camera即可
        if not ret:
            break

        #firstframe = gray   #在调用摄像头的实时视频数据时 采用每一帧与前一帧作差，采用视频数据集的时候 用每一帧与第一帧作差
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if firstframe is None:
            firstframe = gray
            continue


        cv2.imshow("gray", gray)
        cv2.imshow("firstframe", firstframe)
        frameDelta = cv2.absdiff(firstframe, gray)#两幅图的差的绝对值输出到另一幅图上面来
        thresh = cv2.threshold(frameDelta, threshold, 255, cv2.THRESH_BINARY)[1]
        #给两幅图的差异设定阈值
        thresh = cv2.dilate(thresh, None, iterations=2)
        #dilate函数可以对输入图像用特定结构元素进行膨胀操作，该结构元素确定膨胀操作过程中的邻域的形状，各点像素值将被替换为对应邻域上的最大值
        # cnts= cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        #提取运动物体的轮廓 contours为物体轮廓列表
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 根据二值图找轮廓
        print('people is ', len(contours))

        cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)  # 把轮廓画在原图上（0,0,255） 表示 RGB 三通道，红色

        cv2.imshow("frameDelta", frameDelta)
        cv2.imshow("frame with box", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


#book.save('filter_ratio.xls')

capture.release()
cv2.destroyAllWindows()