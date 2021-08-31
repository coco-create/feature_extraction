import cv2
import time
import xlwt
import os

book = xlwt.Workbook()
sheet = book.add_sheet(u'time',cell_overwrite_ok=True)


path = 'cut_video/'
folders = os.listdir(path)
print(folders)

# 实例化:
# Initiate FAST detector
star = cv2.xfeatures2d.StarDetector_create()
# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

video_num = 1
j = 0
for img_path in folders:
    img = path + img_path
    print(img)
    capture = cv2.VideoCapture(img)

    firstframe = cv2.imread('FirstFrame/' + str(video_num) + '.jpg')
    firstframe = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
    i = 1
    if capture.isOpened():
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            cv2.imshow("frame", frame)
            start = time.time()

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化
            # 作差帧后再提取
            frameDelta = cv2.absdiff(firstframe, gray_frame)  # 两幅图的差的绝对值输出到另一幅图上面来
            Diff = time.time()
            Diff_time = Diff - start

            # 计算关键点和描述子,其中kp为关键点keypoints,des为描述子descriptors
            # find the keypoints with STAR
            kp = star.detect(frameDelta, None)
            # compute the descriptors with BRIEF
            kp, des = brief.compute(frameDelta, kp)
            print(des)

            end = time.time()
            BRIEF_time = end - Diff
            print('Diff_time is ', Diff_time, 'BRIEF_time is', BRIEF_time)
            #print('BRIEF_time is', BRIEF_time)

            KP_img = cv2.drawKeypoints(frameDelta, kp, frameDelta, color=(255, 0, 255))

            #cv2.imshow("gray_frame", gray_frame)
            cv2.imshow("frameDelta", frameDelta)
            #cv2.imshow("frame", frame)
            cv2.imshow("img with key point", KP_img)  # imshow后面一定要跟一个waitkey否则无效！！！！！

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if BRIEF_time and Diff_time:
                sheet.write(i, j, Diff_time)
                sheet.write(i, j + 1, BRIEF_time)
                i += 1

    j = j + 2
    video_num += 1

book.save('BRIEF_time.xls')

capture.release()
cv2.destroyAllWindows()