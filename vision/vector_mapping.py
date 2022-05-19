import numpy as np
import cv2


def feather_edges(image, blur_amount):
    mask = np.full(np.array(image.shape[:2]) - blur_amount, 255)
    mask = cv2.copyMakeBorder(mask,
                              int(blur_amount / 2),
                              round(blur_amount / 2),
                              int(blur_amount / 2),
                              round(blur_amount / 2),
                              cv2.BORDER_CONSTANT,
                              value=[0, 0, 0, 0])
    mask = cv2.blur(mask, [blur_amount, blur_amount])

    image[:, :, 3] = mask

    return image


def alpha_over(foreground, background):
    back_alpha = np.expand_dims(background[:, :, 3] / 255, axis=2)
    fore_alpha = np.expand_dims(foreground[:, :, 3] / 255, axis=2)

    foreground[:, :, 3] = np.max((background[:, :, 3], foreground[:, :, 3]))

    background = fore_alpha * foreground + back_alpha * background * (1 - fore_alpha)

    return background