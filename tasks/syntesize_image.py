from typing import List, Tuple

import numpy as np



def transparent_png_img_to_mask(png_img):
    ret_img = png_img.copy()
    gray = cv2.cvtColor(ret_img, cv2.COLOR_BGR2GRAY)
    mask = 1 - gray.astype(np.bool)
    return mask


retVector = Tuple[int, int, int, int] or Tuple[float, float, float, float]
def transform_top_image_for_synthesizing(
        logo_img: np.ndarray,
        base_img_shape,
        ratio: float
    ) -> Tuple[np.ndarray, retVector]:

    base_ratio = np.min(np.array(base_img_shape)/np.array(logo_img.shape))
    ratio = ratio*base_ratio
    resized_logo = cv2.resize(logo_img, dsize=None, fx=ratio, fy=ratio)

    ### synthesize
    shape = base_img_shape[1], base_img_shape[0]
    ret = align_top_img_shape_with_base(base_img_shape=shape, top_img=resized_logo)
    xmin, ymin, xmax, ymax = bounding_rect(resized_logo=resized_logo)

    return ret, (xmin, ymin, xmax, ymax)

def synthesize_image(background, top):
    if background.size != top.size:
        raise ValueError

    mask = transparent_png_img_to_mask(top)
    background = background * np.stack([mask]*3, axis=2)
    synthesized_img = background + top

    return synthesized_img

def align_top_img_shape_with_base(base_img_shape, top_img, x_start=0, y_start=0):
    composite_img = np.zeros((base_img_shape[1], base_img_shape[0], 3))

    top_sh = top_img.shape
    y_start = 0
    x_start = 0
    composite_img[y_start:y_start + top_sh[0], x_start:x_start + top_sh[1]] = top_img
    composite_img = composite_img.astype(np.uint8)

    return composite_img

def bounding_rect(resized_logo):
    thresh = (1 - transparent_png_img_to_mask(resized_logo)) * 255
    thresh = thresh.astype(np.uint8)
    contours, _ = cv2.findContours(thresh, 1, 2)
    coords = cv2.boundingRect(contours[0])
    x, y, w, h = coords
    xmin = x
    ymin = y
    xmax = x+w
    ymax = y+h
    return xmin, ymin, xmax, ymax


def main(base_img, logo_img):
    base_img_shape = (base_img.shape[1], base_img.shape[0])
    ratio = 0.5
    resized_logo, coordinates = transform_top_image_for_synthesizing(logo_img, base_img.shape, ratio)
    composite_image = synthesize_image(background=base_img, top=resized_logo)

    return composite_image, coordinates


if __name__ == '__main__':
    # from PIL import Image
    import cv2
    from pathlib import Path
    from m2det.errors import Errors
    from m2det import PROJECT_ROOT


    ### load data
    logo_img_path = Path("/raid/projects/logo_detection/M2Det/datasets/CocaCola/LogoWithBase/00_image00001.png")
    base_img_path = PROJECT_ROOT/"tests/data/imgs/cat.jpg"
    if not logo_img_path.exists():
        raise Errors().FileNotFound(logo_img_path)
    logo_img = cv2.imread(str(logo_img_path))
    base_img = cv2.imread(str(base_img_path))

    ###############################
    ### main function
    ###############################
    synthesized_img, coords = main(base_img=base_img, logo_img=logo_img)

    ### verification
    synthesized_img_path = PROJECT_ROOT/"tasks"/"sample_synthesized_img.png"

    xmin, ymin, xmax, ymax = coords
    synthesized_img = cv2.rectangle(synthesized_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    from matplotlib import pyplot as plt
    img = synthesized_img.get().astype(np.uint8)
    # img = synthesized_img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    cv2.imwrite(str(synthesized_img_path), synthesized_img)
    print("coords: ", coords)

