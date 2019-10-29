import math
import random
import numpy as np
import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology
from PIL import Image

from torchvision import transforms


def rgb_to_grayscale(np_img):
    """
    Convert an RGB NumPy array to a grayscale NumPy array.
    Shape (h, w, c) to (h, w).
    Args:
        np_img: RGB Image as a NumPy array.
    Returns:
        Grayscale image as NumPy array with shape (h, w).
    """
    # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
    grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
    grayscale = grayscale.astype(np.uint8)
    return grayscale


def obtain_complement(np_img):
    """
    Obtain the complement of an image as a NumPy array.
    Args:
        np_img: Image as a NumPy array.
    Returns:
        Complement image as Numpy array.
    """
    return 255 - np_img


def filter_hysteresis_threshold(np_img, low=50, high=100):
    """
    Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.
    Args:
        np_img: Image as a NumPy array.
        low: Low threshold.
        high: High threshold.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
    """
    hyst = sk_filters.apply_hysteresis_threshold(np_img, low, high)
    hyst = (255 * hyst).astype(np.uint8)
    return hyst


def filter_otsu_threshold(np_img):
    """
    Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.
    Args:
        np_img: Image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
    """
    otsu_thresh_value = sk_filters.threshold_otsu(np_img)
    otsu = np_img > otsu_thresh_value
    otsu = otsu.astype(np.uint8) * 255
    return otsu


def filter_local_otsu_threshold(np_img, disk_size=3):
    """
    Compute local Otsu threshold for each pixel and return binary image based on pixels being less than the
    local Otsu threshold.
    Args:
        np_img: Image as a NumPy array.
        disk_size: Radius of the disk structuring element used to compute the Otsu threshold for each pixel.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8) where local Otsu threshold values have been applied to original image.
    """
    local_otsu = sk_filters.rank.otsu(np_img, sk_morphology.disk(disk_size))
    local_otsu = local_otsu.astype(np.uint8) * 255
    return local_otsu


def filter_entropy(np_img, neighborhood=9, threshold=5):
    """
    Filter image based on entropy (complexity).
    Args:
        np_img: Image as a NumPy array.
        neighborhood: Neighborhood size (defines height and width of 2D array of 1's).
        threshold: Threshold value.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a measure of complexity.
    """
    entr = (
        sk_filters.rank.entropy(np_img, np.ones((neighborhood, neighborhood))) > threshold
    )
    entr = entr.astype(np.uint8) * 255
    return entr


def filter_canny(np_img, sigma=1, low_threshold=0, high_threshold=25):
    """
    Filter image based on Canny algorithm edges.
    Args:
        np_img: Image as a NumPy array.
        sigma: Width (std dev) of Gaussian.
        low_threshold: Low hysteresis threshold value.
        high_threshold: High hysteresis threshold value.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8) representing Canny edge map (binary image).
    """
    can = sk_feature.canny(
        np_img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold
    )
    can = can.astype(np.uint8) * 255
    return can


def filter_contrast_stretch(np_img, low=40, high=60):
    """
    Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in
    a specified range.
    Args:
        np_img: Image as a NumPy array (gray or RGB).
        low: Range low value (0 to 255).
        high: Range high value (0 to 255).
    Returns:
        Image as NumPy array with contrast enhanced.
    """
    low_p, high_p = np.percentile(np_img, (low * 100 / 255, high * 100 / 255))
    cons_stretch = sk_exposure.rescale_intensity(np_img, in_range=(low_p, high_p))
    return cons_stretch


def filter_histogram_equalization(np_img, nbins=256):
    """
    Filter image (gray or RGB) using histogram equalization to increase contrast in image.
    Args:
        np_img: Image as a NumPy array (gray or RGB).
        nbins: Number of histogram bins.
        output_type: Type of array to return (float or uint8).
    Returns:
        NumPy array (float or uint8) with contrast enhanced by histogram equalization.
    """

    # if uint8 type and nbins is specified, convert to float so that nbins can be a value besides 256
    if np_img.dtype is np.uint8 and nbins != 256:
        np_img = np_img / 255
    hist_equ = sk_exposure.equalize_hist(np_img, nbins=nbins)
    hist_equ = (hist_equ * 255).astype(np.uint8)
    return hist_equ


def filter_adaptive_equalization(np_img, nbins=256, clip_limit=0.01):
    """
    Filter image (gray or RGB) using adaptive equalization to increase contrast in image, where contrast in local regions
    is enhanced.
    Args:
        np_img: Image as a NumPy array (gray or RGB).
        nbins: Number of histogram bins.
        clip_limit: Clipping limit where higher value increases contrast.
        output_type: Type of array to return (float or uint8).
    Returns:
        NumPy array (float or uint8) with contrast enhanced by adaptive equalization.
    """
    adapt_equ = sk_exposure.equalize_adapthist(np_img, nbins=nbins, clip_limit=clip_limit)
    adapt_equ = (adapt_equ * 255).astype(np.uint8)
    return adapt_equ


def filter_local_equalization(np_img, disk_size=50):
    """
    Filter image (gray) using local equalization, which uses local histograms based on the disk structuring element.
    Args:
        np_img: Image as a NumPy array.
        disk_size: Radius of the disk structuring element used for the local histograms
    Returns:
        NumPy array with contrast enhanced using local equalization.
    """
    local_equ = sk_filters.rank.equalize(np_img, selem=sk_morphology.disk(disk_size))
    return local_equ


def filter_rgb_to_hed(np_img):
    """
    Filter RGB channels to HED (Hematoxylin - Eosin - Diaminobenzidine) channels.
    Args:
        np_img: RGB image as a NumPy array.
        output_type: Type of array to return (float or uint8).
    Returns:
        NumPy array (float or uint8) with HED channels.
    """
    hed = sk_color.rgb2hed(np_img)
    hed = (sk_exposure.rescale_intensity(hed, out_range=(0, 255))).astype(np.uint8)
    return hed


def filter_rgb_to_hsv(np_img):
    """
    Filter RGB channels to HSV (Hue, Saturation, Value).
    Args:
        np_img: RGB image as a NumPy array.
        display_np_info: If True, display NumPy array info and filter time.
    Returns:
        Image as NumPy array in HSV representation.
    """

    hsv = sk_color.rgb2hsv(np_img)
    return hsv


def filter_hsv_to_h(hsv):
    """
    Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float
    values are multiplied by 360 for their degree equivalents for simplicity. For more information, see
    https://en.wikipedia.org/wiki/HSL_and_HSV
    Args:
        hsv: HSV image as a NumPy array.
        output_type: Type of array to return (float or int).
        display_np_info: If True, display NumPy array info and filter time.
    Returns:
        Hue values (float or int) as a 1-dimensional NumPy array.
    """
    h = hsv[:, :, 0]
    h = h.flatten()
    h *= 360
    h = h.astype(np.uint8)
    return h


def filter_hsv_to_s(hsv):
    """
    Experimental HSV to S (saturation).
    Args:
        hsv:  HSV image as a NumPy array.
    Returns:
        Saturation values as a 1-dimensional NumPy array.
    """
    s = hsv[:, :, 1]
    s = s.flatten()
    return s


def filter_hsv_to_v(hsv):
    """
    Experimental HSV to V (value).
    Args:
        hsv:  HSV image as a NumPy array.
    Returns:
        Value values as a 1-dimensional NumPy array.
    """
    v = hsv[:, :, 2]
    v = v.flatten()
    return v


def filter_hed_to_hematoxylin(np_img):
    """
    Obtain Hematoxylin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
    contrast.
    Args:
        np_img: HED image as a NumPy array.
        output_type: Type of array to return (float or uint8).
    Returns:
        NumPy array for Hematoxylin channel.
    """
    hema = np_img[:, :, 0]
    hema = (sk_exposure.rescale_intensity(hema, out_range=(0, 255))).astype(np.uint8)
    return hema


def filter_hed_to_eosin(np_img):
    """
    Obtain Eosin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
    contrast.
    Args:
        np_img: HED image as a NumPy array.
        output_type: Type of array to return (float or uint8).
    Returns:
        NumPy array for Eosin channel.
    """
    eosin = np_img[:, :, 1]
    eosin = (sk_exposure.rescale_intensity(eosin, out_range=(0, 255))).astype(np.uint8)
    return eosin


def filter_binary_erosion(np_img, disk_size=5, iterations=1, output_type="bool"):
    """
    Erode a binary object (bool, float, or uint8).
    Args:
        np_img: Binary image as a NumPy array.
        disk_size: Radius of the disk structuring element used for erosion.
        iterations: How many times to repeat the erosion.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8) where edges have been eroded.
    """
    if np_img.dtype == "uint8":
        np_img = np_img / 255
    result = sc_morph.binary_erosion(
        np_img, sk_morphology.disk(disk_size), iterations=iterations
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result


def filter_binary_dilation(np_img, disk_size=5, iterations=1, output_type="bool"):
    """
  Dilate a binary object (bool, float, or uint8).
  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for dilation.
    iterations: How many times to repeat the dilation.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8) where edges have been dilated.
  """
    if np_img.dtype == "uint8":
        np_img = np_img / 255
    result = sc_morph.binary_dilation(
        np_img, sk_morphology.disk(disk_size), iterations=iterations
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result

def filter_threshold(np_img, threshold):
    """
    Return mask where a pixel has a value if it exceeds the threshold value.
    Args:
        np_img: Binary image as a NumPy array.
        threshold: The threshold value to exceed.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
        pixel exceeds the threshold value.
    """
    result = np_img > threshold
    result = result.astype(np.uint8) * 255
    return result


def uint8_to_bool(np_img):
    """
    Convert NumPy array of uint8 (255,0) values to bool (True,False) values
    Args:
        np_img: Binary image as NumPy array of uint8 (255,0) values.
    Returns:
        NumPy array of bool (True,False) values.
    """
    result = (np_img / 255).astype(bool)
    return result


def _transform(image, label):
    if np.random.random() > 0.5:
        image = transforms.functional.hflip(image)
        label = transforms.functional.hflip(label)

    if np.random.random() > 0.5:
        degree = random.choice([90, 180, 270])
        image = transforms.functional.rotate(image, degree)
        label = transforms.functional.rotate(label, degree)
    return image, label

def mask_rgb(rgb, mask):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

    Args:
        rgb: RGB image as a NumPy array.
        mask: An image mask to determine which pixels in the original image should be displayed.

    Returns:
        NumPy array representing an RGB image with mask applied.
    """
    result = rgb * np.dstack([mask, mask, mask])
    return result


def mask_percent(np_img):
    """
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
    Args:
        np_img: Image as a NumPy array.
    Returns:
        The percentage of the NumPy array that is masked.
    """
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    return mask_percentage


def filter_green_channel(
    np_img,
    green_thresh=200,
    avoid_overmask=True,
    overmask_thresh=90,
    output_type="bool",
):
    """
    Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
    and eosin are purplish and pinkish, which do not have much green to them.
    Args:
        np_img: RGB image as a NumPy array.
        green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
        avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
        overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
    """

    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (
        (mask_percentage >= overmask_thresh)
        and (green_thresh < 255)
        and (avoid_overmask is True)
    ):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        # print(
        #     "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d"
        #     % (mask_percentage, overmask_thresh, green_thresh, new_green_thresh)
        # )
        gr_ch_mask = filter_green_channel(
            np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type
        )
    np_img = gr_ch_mask

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    return np_img


def filter_remove_small_objects(
    np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"
):
    """
    Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
    is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
    reduce the amount of masking that this filter performs.
    Args:
        np_img: Image as a NumPy array of type bool.
        min_size: Minimum size of small object to remove.
        avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
        overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8).
    """

    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (
        (mask_percentage >= overmask_thresh)
        and (min_size >= 1)
        and (avoid_overmask is True)
    ):
        new_min_size = min_size / 2
        # print(
        #     "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d"
        #     % (mask_percentage, overmask_thresh, min_size, new_min_size)
        # )
        rem_sm = filter_remove_small_objects(
            np_img, new_min_size, avoid_overmask, overmask_thresh, output_type
        )
    np_img = rem_sm

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    return np_img


def filter_grays(rgb, tolerance=15, output_type="bool"):
    """
    Create a mask to filter out pixels where the red, green, and blue channel values are similar.
    Args:
        np_img: RGB image as a NumPy array.
        tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """

    rgb = rgb.astype(np.int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result


def filter_green(
    rgb,
    red_upper_thresh,
    green_lower_thresh,
    blue_lower_thresh,
    output_type="bool",
):
    """
    Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
    red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
    Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
    lower threshold value rather than a blue channel upper threshold value.
    Args:
        rgb: RGB image as a NumPy array.
        red_upper_thresh: Red channel upper threshold value.
        green_lower_thresh: Green channel lower threshold value.
        blue_lower_thresh: Blue channel lower threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.
    Returns:
        NumPy array representing the mask.
    """
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] > green_lower_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result


def filter_red(
    rgb,
    red_lower_thresh,
    green_upper_thresh,
    blue_upper_thresh,
    output_type="bool",
):
    """
    Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
    red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.
    Args:
        rgb: RGB image as a NumPy array.
        red_lower_thresh: Red channel lower threshold value.
        green_upper_thresh: Green channel upper threshold value.
        blue_upper_thresh: Blue channel upper threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.
    Returns:
        NumPy array representing the mask.
    """
    r = rgb[:, :, 0] > red_lower_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] < blue_upper_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result


def filter_blue(
    rgb,
    red_upper_thresh,
    green_upper_thresh,
    blue_lower_thresh,
    output_type="bool",
):
    """
    Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
    red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.
    Args:
        rgb: RGB image as a NumPy array.
        red_upper_thresh: Red channel upper threshold value.
        green_upper_thresh: Green channel upper threshold value.
        blue_lower_thresh: Blue channel lower threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.
    Returns:
        NumPy array representing the mask.
    """
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result


def filter_red_pen(rgb, output_type="bool"):
    """
    Create a mask to filter out red pen marks from a slide.
    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing the mask.
    """
    result = (
        filter_red(
            rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90
        )
        & filter_red(
            rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30
        )
        & filter_red(
            rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105
        )
        & filter_red(
            rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125
        )
        & filter_red(
            rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145
        )
        & filter_red(
            rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70
        )
        & filter_red(
            rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150
        )
        & filter_red(
            rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65
        )
        & filter_red(
            rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45
        )
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result


def filter_green_pen(rgb, output_type="bool"):
    """
    Create a mask to filter out green pen marks from a slide.
    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing the mask.
    """
    result = (
        filter_green(
            rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140
        )
        & filter_green(
            rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110
        )
        & filter_green(
            rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100
        )
        & filter_green(
            rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60
        )
        & filter_green(
            rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210
        )
        & filter_green(
            rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225
        )
        & filter_green(
            rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200
        )
        & filter_green(
            rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20
        )
        & filter_green(
            rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40
        )
        & filter_green(
            rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35
        )
        & filter_green(
            rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60
        )
        & filter_green(
            rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105
        )
        & filter_green(
            rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180
        )
        & filter_green(
            rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150
        )
        & filter_green(
            rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195
        )
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result


def filter_blue_pen(rgb, output_type="bool"):
    """
  Create a mask to filter out blue pen marks from a slide.
  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing the mask.
  """
    result = (
        filter_blue(
            rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190
        )
        & filter_blue(
            rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200
        )
        & filter_blue(
            rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230
        )
        & filter_blue(
            rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210
        )
        & filter_blue(
            rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160
        )
        & filter_blue(
            rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130
        )
        & filter_blue(
            rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180
        )
        & filter_blue(
            rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85
        )
        & filter_blue(
            rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65
        )
        & filter_blue(
            rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140
        )
        & filter_blue(
            rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120
        )
        & filter_blue(
            rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175
        )
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result


def apply_filters(rgb):
    """
    Apply filters to image as Pillow Image.
    Args:
        image: Image as Pillow.
    Returns:
        Resulting filtered image as a Pillow Image.
    """

    # rgb = np.array(image)

    mask_not_green = filter_green_channel(rgb)
    # rgb_not_green = mask_rgb(rgb, mask_not_green)

    mask_not_gray = filter_grays(rgb)
    # rgb_not_gray = mask_rgb(rgb, mask_not_gray)

    # mask_no_red_pen = filter_red_pen(rgb)
    # rgb_no_red_pen = mask_rgb(rgb, mask_no_red_pen)

    mask_no_green_pen = filter_green_pen(rgb)
    # rgb_no_green_pen = mask_rgb(rgb, mask_no_green_pen)

    mask_no_blue_pen = filter_blue_pen(rgb)
    # rgb_no_blue_pen = mask_rgb(rgb, mask_no_blue_pen)
        # & mask_no_red_pen

    mask_gray_green_pens = (
        mask_not_gray
        & mask_not_green
        & mask_no_green_pen
        & mask_no_blue_pen
    )
    rgb_gray_green_pens = mask_rgb(rgb, mask_gray_green_pens)

    # mask_remove_small = filter_remove_small_objects(
    #     mask_gray_green_pens, min_size=500, output_type="bool"
    # )
    # rgb_remove_small = mask_rgb(rgb, mask_remove_small)
    # return rgb_remove_small
    return rgb_gray_green_pens
    # return Image.fromarray(rgb_remove_small)
