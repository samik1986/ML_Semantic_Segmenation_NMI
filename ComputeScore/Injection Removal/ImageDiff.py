import os
from PIL import Image


# image_path = 'data/FINALIZE/Injection_84.tif'
#
#
# image = Image.open(image_path)
# cropped = image.crop((0, 0, 8192, 11264))
# cropped.save('data/FINALIZE/Injection_84_cropped.tif')

image_path = '001.tif'
mask_path = 'StitchedImage_Z084_L001_InjectionMask.tif'
output_path = 'Samik_84_masked.tif'

image2 = Image.open(image_path)
image1 = Image.new('L', image2.size, 0)
mask = Image.open(mask_path)
mask = mask.crop((0, 0, 8192, 11264))

new_image = Image.composite(image1, image2, mask)
new_image.save(output_path)