from PIL import Image, ImageDraw, ImageFont

def drawText(filepath, fontsize, text):
    image = Image.open(filepath)

    draw = ImageDraw.Draw(image)
    #font = ImageFont.load("arial.pil")
    font = ImageFont.truetype(font='arial.ttf', size=fontsize)

    (x,y) = (50,50)
    message = text # text needs to be String
    color = 'rgb(0,0,0)' #black

    draw.text((x,y), message, fill=color, font=font)
    image.save('InpaintedImage.png')

drawText('testingImages/22982871191_ec61e36939_n.jpg', 45, 'hello')
