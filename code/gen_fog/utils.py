from PIL import Image

def convert_png2jpg(src, dest):
    im1 = Image.open(src)
    im1.save(dest, dpi=(72, 72))
    print('Convert done!')