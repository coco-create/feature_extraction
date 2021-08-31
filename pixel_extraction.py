import cv2


image=cv2.imread('images/OIP-C.jpg')

for pixel in image:
    avg = (pixel.red + pixel.green + pixel.blue)/ 3 ;
    if pixel.red > 1.5 * avg:
        pixel.red = 0
        pixel.blue = 0
        pixel.green = 0
    else:
        pixel.red = 255
        pixel.blue = 255
        pixel.green = 255
cv2.imshow('img', image)



