import os.path
import numpy as np
import cv2
from datasets.SOD.EDI.src.event2video_final import event2video_final

def mat2gray(image):
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val != min_val:
        image = (image - min_val) / (max_val - min_val)
    return image


def blur_to_sharp(png_path, npy_path, v_length=100, delta=100):
    timescale = 1e6

    if isinstance(npy_path, str):
        data = np.load(npy_path)
    else:
        data = npy_path

    x_o, y_o, pol_o, t_o = data[:, 0], data[:, 1], data[:, 3].copy(), data[:, 2]
    pol_o[pol_o == 0] = -1
    t_o = np.mod(t_o, 1e8)
    t_o /= timescale
    x, y, pol, t = x_o, y_o, pol_o, t_o

    img = cv2.imread(png_path, 0)


    blur = np.asarray(img)
    blur = mat2gray(blur)

    eventstart = t_o[0]
    eventend = t_o[-1]
    exptime = eventend - eventstart
    I_video, deltaT = event2video_final(blur, x, y, pol, t, eventstart, eventend, exptime, v_length, delta)

    for i in range(len(I_video)):
        I_video[i] = mat2gray(I_video[i]) * 255

    return I_video, deltaT, delta


def gen_sharp_img(cI, delta, data):
    for i in range(len(cI)):
        array_2d = np.squeeze(cI[i])
        gray_image = cv2.cvtColor(array_2d, cv2.COLOR_GRAY2BGR)

        img_save_path = f'./{data}_v{len(cI)}_delta{delta}'
        if not os.path.exists(img_save_path):
            os.mkdir(img_save_path)

        cv2.imwrite(f'{img_save_path}/{i:04d}.png', gray_image)


if __name__ == '__main__':
    data = '013_train_low_light_156'
    png_path = f'D:/publicData\SOD/train/images/{data}.png'
    npy_path = f'D:/publicData\SOD/train/events/{data}.npy'
    cI, deltaT, delta = blur_to_sharp(png_path, npy_path, v_length=5, delta=110)
    gen_sharp_img(cI, delta, data)


