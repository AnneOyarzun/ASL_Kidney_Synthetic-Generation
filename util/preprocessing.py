import numpy as np
import numpy as np
import SimpleITK as sitk
import numpy as np
from skimage.exposure import rescale_intensity


def specific_intensity_window(image, window_percent=0.15):
    image = image.astype('int64') 
    arr = np.asarray_chkfinite(image)
    min_val = arr.min()
    number_of_bins = arr.max() - min_val + 1
    hist = np.bincount((arr-min_val).ravel(), minlength=number_of_bins)
    hist_new = hist[1:]
    total = np.sum(hist_new)
    window_low = window_percent * total
    window_high = (1 - window_percent) * total
    cdf = np.cumsum(hist_new)
    low_intense = np.where(cdf >= window_low) + min_val
    high_intense = np.where(cdf >= window_high) + min_val
    res = rescale_intensity(image, in_range=(low_intense[0][0], high_intense[0][0]),out_range=(arr.min(), arr.max()))
    return res