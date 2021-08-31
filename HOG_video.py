import cv2
import numpy as np
import time
import xlwt
import os
from skimage import feature as ft


book = xlwt.Workbook()
sheet = book.add_sheet(u'time',cell_overwrite_ok=True)


path = 'cut_video/'
folders = os.listdir(path)
#print(folders)

orit = 3
cell_size = 10
block_size = 3
j = 0
video_num = 1

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

            start = time.time()

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化

            # 作差帧后再提取
            frameDelta = cv2.absdiff(firstframe, gray_frame)  # 两幅图的差的绝对值输出到另一幅图上面来
            cv2.imshow('firstframe',firstframe)

            Diff = time.time()
            Diff_time = Diff - start

            #features, hog_img = ft.hog(frame, orientations=orit, pixels_per_cell=[cell_size, cell_size], cells_per_block=[block_size, block_size],visualize=True)
            features, hog_img = ft.hog(frameDelta, visualize=True)
            #print('ft is ', features)

            end = time.time()
            HOG_time = end - start
            print('Diff_time is', Diff_time, 'HOG_time is', HOG_time)

            cv2.imshow("frame", frame)
            cv2.imshow("frameDelta", frameDelta)
            cv2.imshow("hog_img", hog_img)  # imshow后面一定要跟一个waitkey否则无效！！！！！

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if HOG_time and Diff_time:
                sheet.write(i, j, Diff_time)
                sheet.write(i, j + 1, HOG_time)
                i += 1

    j = j + 2
    video_num += 1

book.save('HOG_time.xls')

capture.release()
cv2.destroyAllWindows()