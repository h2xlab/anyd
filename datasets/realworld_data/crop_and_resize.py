
import cv2

'''
对于 av2 的图片
大幅度裁剪，只保留中间扁平的部分，使其比例和 nuscene 原图一致。
然后 resize 成 height 225, width 400

'''

def crop_central_and_resize(image_bgr):
    
    if image_bgr.shape==(2048, 1550, 3):
    
        x=0
        crop_width=image_bgr.shape[1]
        crop_height=1550*900//1600
        y=(2048-crop_height)//2

        image_bgr=image_bgr[y:y+crop_height,x:x+crop_width]
        
    dimension=(400,225)
    resized_image_bgr = cv2.resize(image_bgr, dimension, interpolation=cv2.INTER_AREA)

    return resized_image_bgr
    



'''
对于 av2 的图片
只是稍微剪掉下面的 argo 车
然后 resize 成 height 225, width 400

'''

def crop_car_and_resize(image_bgr):
    
    if image_bgr.shape==(2048, 1550, 3):
    
        x=0
        crop_width=image_bgr.shape[1]
        crop_height=1850
        y=0
        image_bgr=image_bgr[y:y+crop_height,x:x+crop_width]

    
    dimension=(400,225)

    resized_image_bgr = cv2.resize(image_bgr, dimension, interpolation=cv2.INTER_AREA)

    return resized_image_bgr
    
    
    
    
    
    


