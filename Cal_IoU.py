def  Cal_IoU(box1,box2):    #box=[x1,y1,x2,y2,s]
    iou_w  = max(0,  min( box1[2], box2[2] ) - max(box1[0],box2[0]))
    iou_h = max(0,  min(box1[3], box2[3]) - max(box1[1],box2[1]))
    inter_s = iou_h * iou_w
    union_s = box1[4] + box2[4] - inter_s
    return inter_s/union_s


#bbox=[x,y,w,h]  ---->  box=[x1,y1,x2,y2,s]
#resolution = (x,y)
def change_box(bbox,resolution):
    x1 =  bbox[0] * resolution[0]
    x2 = ( bbox[0] + bbox[2] ) * resolution[0]
    y1 = bbox[1] * resolution[1]
    y2 = ( bbox[1] + bbox[3] ) * resolution[0]
    s = bbox[2] * bbox[3] * resolution[0] * resolution[1]
    return x1, y1, x2, y2, s