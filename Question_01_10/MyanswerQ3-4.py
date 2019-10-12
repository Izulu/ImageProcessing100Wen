import numpy as np
import cv2

img=cv2.imread('imori.jpg').astype(np.float)

def gray_scale(img):
    b = img[:,:,0].copy()
    g = img[:,:,1].copy()
    r = img[:,:,2].copy()
    result = 0.2126*r + 0.7152*g+0.0722*b
    return result.astype(np.uint8)

def img_bw(gray_scale_img,threshold=128):
    result = gray_scale_img.copy()
    result[result<threshold]=0
    result[result>=threshold]=255
    return result


def otsu_bw(gray_scale_img):
    img = gray_scale_img.copy()
    s = 0
    argmax_t = 0
    size = img.size
    for t in range(1,255):
        v0 = img[img<t]
        v1 = img[img>=t]
        if len(v0) >0:
            w0 = v0.size / size
            M0 = v0.mean()
        else
            w0, M0 = 0
        if len(v1) >0:
            w1 = v1.size / size
            M1 = v1.mean()
        else
            w1, M1 = 0
        s_current = w0 * w1 * ((M0 - M1)**2)
        if s_current > s:
            argmax_t = t
            s = s_current
    #
    #Sb^2 = s
    #X = Sb^2/Sw^2 = Sb^2 / (St^2-Sb^2)  ~ Sb^2 ~ Sb
    #
    return argmax_t

img_gray = gray_scale(img)
t = otsu_bw(img_gray)
result_img = img_bw(img_gray,t)
