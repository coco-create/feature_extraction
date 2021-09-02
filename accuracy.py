#%%将txt中的信息转为bbox形式  filename  ----->bboxes
def make_bbox(filename):
    bboxes = []
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            line = line.strip().split(" ") # 按照空格分段
            if len(line) > 1:
                bbox = [ line[1], line[2], line[3], line[4] ]
                bbox = list(map(float,bbox))
                bboxes.append(bbox)
            else:
                bboxes.append([])
           #print("bbox =",bbox)
        #print(bboxes)
    return bboxes

# %%计算两个bbox的IoU中I 和U的面积
def  Cal_IoU(box1,box2):    #box=[x1,y1,x2,y2,s]  -----> inter_s, union_s
    if box1 and box2:
        iou_w  = max(0,  min( box1[2], box2[2] ) - max(box1[0],box2[0]))
        #print(iou_w)
        iou_h = max(0,  min(box1[3], box2[3]) - max(box1[1],box2[1]))
        #print(iou_h)
        inter_s = iou_h * iou_w
        #print(inter_s)
        union_s = box1[4] + box2[4] - inter_s
         #print(union_s)
        return inter_s, union_s
    elif box1:
        return 0, box1[4]
    elif box2:
        return 0, box2[4]
    else:
        return 0, 0

#%%定义bbox转换函数 : bbox=[x,y,w,h]  ---->  box=[x1,y1,x2,y2,s]
#resolution = (x,y)
def change_box(bbox,resolution):
    if bbox:
        x1 =  bbox[0] * resolution[0]
        x2 = ( bbox[0] + bbox[2] ) * resolution[0]
        y1 = bbox[1] * resolution[1]
        y2 = ( bbox[1] + bbox[3] ) * resolution[1]
        s = bbox[2] * bbox[3] * resolution[0] * resolution[1]
        return [x1, y1, x2, y2, s]
    else:
        return []

#%%计算before和after 的 两张图中 所有bbox的IoU   
def txt_IoU(file1,file2):
    bboxes_a = make_bbox(file1)
    bboxes_b = make_bbox(file2)

    resolution = (360, 288)
    iou =[]
    inter = 0
    union = 0 

    for i in range(len(bboxes_a)):
        bbox_a = change_box( bboxes_a[i] , resolution )
        bbox_b = change_box( bboxes_b[i] , resolution )
        inter_s , union_s = Cal_IoU(bbox_a, bbox_b)
        inter += inter_s
        union += union_s
        #print(inter_s,union_s)
        #print(inter, union)
        #print( "bbox_a", bbox_a)
        #print( "bbox_b", bbox_b)

    return inter/union

# %%遍历before 和after中所有txt计算IoU
path_after  = "/home/coco/Project/python_try/txt/time/after"
path_before  = "/home/coco/Project/python_try/txt/time/before"

all_score = []
files_after = os.walk(path_after)
for root,dirs, afiles in files_after:         # 递归遍历及输出
    for afile in afiles:
        afile_name = os.path.join(root, afile)
        print('\n after = ',afile, '\n' )

        score_list = []
        files_before = os.walk(path_before)
        for root,dirs, bfiles in files_before:         # 递归遍历及输出
            for bfile in bfiles:
                bfile_name = os.path.join(root, bfile)
                score = txt_IoU(afile_name, bfile_name)
                score_list.append(score)
                print('   before = ', bfile,'     score =', score)
        #print(score_list)
        all_score.append(score_list)

#print(all_score)


# %%遍历.txt并排序
    path = "/home/coco/Project/new/yolov3-master/output"
    txt_files = os.walk(path)
    for root,dirs,files in txt_files:         # 递归遍历及输出
        print("root:%s" % root)
        for dir in dirs:
        print(os.path.join(root,dir))
        for file in files:
            file_name = os.path.join(root,file)
            file_item = os.path.splitext(file_name)
            if '.txt' == file_item[1]:
                print('file name is ', file_name)
                
                with open(file_name, "r") as f:
                    data = f.readlines()
                    print('\n lines is ',data)
                    data.sort()
                    print('\n sorted lines is ',data)

                with open( (file_item[0] + "_sorted.txt") , "w") as f:
                    f.writelines(data)



#%%
bbox1 = [84.99996, 179.500032, 171.0, 390.500064, 18146.01119200128]
bbox2 = [83.00016000000001, 180.0, 173.00016, 390.000096, 18900.00864]
score = Cal_IoU(bbox1,bbox2)
print(score)

# %%一行一行的读取（删除/n)
lines = []
with open("1501.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            lines.append(line)
            print(line)

        print('\n lines is ',lines)
        lines.sort()#此函数无返回值，不会返回对象，会改变原有的list
        print('\n sorted lines is ',lines)

# %%写所有行
    with open("1501_sorted.txt", "w") as f:
        f.writelines(lines)

# %%一起读 + 排序
    with open("1501.txt", "r") as f:
        data = f.readlines()
        print('\n lines is ',data)
        data.sort()
        print('\n sorted lines is ',data)
# %%写txt
    with open("1501_sorted.txt", "w") as f:
        f.writelines(data)
# %%遍历
    import os
    folders = os.listdir('.')
    print(folders)
    for file_path in folders:
        print(file_path)

    data = os.walk(".")               # 遍历test目录
    for root,dirs,files in data:         # 递归遍历及输出
        print("root:%s" % root)
        for dir in dirs:
        print(os.path.join(root,dir))
        for file in files:
            file_name = os.path.join(root,file)
            file_item = os.path.splitext(file_name)#将文件名和扩展名拆开
            print('file name is ', file_name)
            print('file item is ', file_item)
      



# %%
path_after  = "/home/coco/Project/python_try/txt/time/after"
path_before  = "/home/coco/Project/python_try/txt/time/before"

files_before = os.walk(path_before)
files_after = os.walk(path_after)

for root,dirs, bfiles in files_before:         # 递归遍历及输出
    for bfile in bfiles:
        bfile_name = os.path.join(root, bfile)

        with open(bfile_name, "r") as f:
            data = f.readlines()
            print(len(data))
            #print('\n lines is ',data)


