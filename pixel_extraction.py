
def colorless(filename):
    image=SimpleImage(filename)
    for pixel in image:
        avg=(pixel.red + pixel.green + pixel.blue)/3;
        if pixel.red > 1.5*avg:
            pixel.red = 0
            pixel.blue = 0
            pixel.green = 0
        else:
            pixel.red = 255
            pixel.blue = 255
            pixel.green = 255
    return image


colorless('pic3.jpg')
