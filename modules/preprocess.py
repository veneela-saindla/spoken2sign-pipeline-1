import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def clean_sequence(raw_data):
    data = np.array(raw_data, float)

    # Missing values → nan
    mask = (data[:,:,0] == 0) & (data[:,:,1] == 0)
    data[mask] = np.nan

    # Velocity outlier removal
    vel = np.diff(data, axis=0, prepend=data[:1])
    speed = np.linalg.norm(np.nan_to_num(vel[:,:,:2]), axis=2)
    data[speed > 0.15] = np.nan

    # Interpolation
    T, V, C = data.shape
    for v in range(V):
        for c in range(C):
            s = data[:,v,c]
            nan = np.isnan(s)
            if (~nan).sum() > T * 0.3:
                f = interp1d(np.where(~nan)[0], s[~nan], fill_value="extrapolate", bounds_error=False)
                data[:,v,c] = f(np.arange(T))
            else:
                data[nan,v,c] = 0.5

    # Hand smoothing
    data[:,0:42,:]  = gaussian_filter1d(data[:,0:42,:],  sigma=2.0, axis=0)

    # Body smoothing
    data[:,42:49,:] = gaussian_filter1d(data[:,42:49,:], sigma=1.0, axis=0)

    return data
