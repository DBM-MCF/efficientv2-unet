import math
from glob import glob
from time import sleep

import cv2
import keras
from skimage.io import imread

from efficient_v2_unet.model.predict import predict_single_image, predict

print('best' in 'fdafda_fobest-.h5')

list_of_list = [[] for x in range(5)]
print('list of list:', list_of_list)

thresholds = [x/10 for x in range(0, 11)]
print(thresholds)

resoltuions = {i: None for i in range(1, 4)}
print(resoltuions)
end


print('find np array type:')
img_path = 'G:/20231006_Martin/images/Slide2-25_ChannelBrightfield_Seq0003_XY1-1.tif'
img = imread(img_path)
print('list?', isinstance(img, list))
print('type=', type(img), '\n\n')

print('img')
predict(img, 'fdsafda')
print('list[img]')
predict([img], 'fdsaf')
print('img_path')
predict(img_path, 'fdsa')
end


print('Continue in ', end='')
for i in range(3, 0, -1):
    print(i, end='...')
    sleep(1)
print('0')
end

test_mask_dir = None
print('not None=', not test_mask_dir)
test_mask_dir = 'fdsa'
print('not (str)=', not test_mask_dir)
print('not not (str)=', not not test_mask_dir)
end

print('test zip and enumerate')
dir_img_unsplit = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_images"
dir_mask_unsplit = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_masks"
imgs_ = glob(dir_img_unsplit + '/*.tif')
masks_ = glob(dir_mask_unsplit + '/*.tif')
for i, (img, mask) in enumerate(zip(imgs_, masks_)):
    print(i, img, mask)
    if i == 2:
        break
end





print('checking image padding')
shape_y = 4875
img_size = 256
pad_y = math.ceil(shape_y / img_size)
pad_y = pad_y * img_size
print(shape_y, pad_y, pad_y-img_size)
end


print('checking cv2 resize')
img_ = 'G:/20231006_Martin/EfficientUNet/resources/test_files/stitch_test_image.tif'
img_ = imread(img_)
print(img_.name)
print(img_.shape)
img_ = cv2.resize(img_, (1000, 500))
print(img_.shape)
end


'''
# this was just to create a model, to plot it in jupyter notebook
model_temp = build_efficient_unet('b0', (256, 256, 3))

model_temp.save('models/my_efficientUNet-B0_model/empty_model.h5')
'''

model_path = './models/my_efficientUNet-B3_allIMGs/my_efficientUNet-B3_allIMGs.h5'

#model_type = 'b0' # b0 seems not too bad
#final_name = 'my_efficientUNet-' + model_type.upper() + '_model'
#model = keras.models.load_model(final_name + '.h5')
model = keras.models.load_model(model_path)

print("in playground")
img_path = 'G:/20231006_Martin/images/Slide2-25_ChannelBrightfield_Seq0003_XY1-1.tif'
img = imread(img_path)
predict_single_image(img, model, factor=1, batch_size=0)
print("playground end")

'''
path = 'G:/20231006_Martin/images/Slide2-26_ChannelBrightfield_Seq0007_XY1.ome.tif'
img = imread(path)
img_crop = img[6000:10048, 5500:9548, :]
print(img.shape)
img = cv2.resize(img_crop, (2048, 2048))
#pred = model.predict(np.asarray([img]))

#show_2images(img, pred)
#show_3images(img, pred, thresh=0.9)

tiles = create_tiles(img_crop, 256)

pred_tiles = []
for img in tiles:
    pred_tiles.append(
        model.predict(np.asarray([img]))
    )


row_col = int(len(tiles)**0.5)
print('img', np.asarray(tiles).shape)
#print('mask', np.asarray(pred_tiles).shape)
img_montage = montage(tiles, grid_shape=(row_col, row_col), channel_axis=-1)
#mask_montage = montage(pred_tiles, grid_shape=(row_col, row_col))

plt.imshow(img_montage)
plt.show()
#show_3images(img_montage, img_montage, thresh=0.9)
'''