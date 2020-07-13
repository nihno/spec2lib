from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, argrelextrema,find_peaks
from scipy import interpolate, stats
import numpy as np
import matplotlib.pyplot as plt






def ir_smooth(x,y, _window = 360, _overlap = 60):
    N = int(np.trunc(len(x) / _overlap))
    Y = np.zeros([len(x),N])
    for i in range(N):
        xtmp = x[i*_overlap:(i+1)*_overlap + (_window - _overlap)]
        ytmp = y[i*_overlap:(i+1)*_overlap + (_window - _overlap)]
        x1,y1,y2,W = variable_smooth(xtmp,ytmp)
        Y[i*_overlap:(i+1)*_overlap + (_window - _overlap),i] = y2

    Y[Y==0] = np.nan
    Ymedian = np.nanmedian(Y,axis=1)
    return Ymedian

def variable_smooth(x,y):
    W = 100
    _window = 11
    while W > 3E-2:
        y2 = savgol_filter(y,_window,2)
        _fft = np.fft.fft(y2)
        _fft = np.sort(_fft.real)
        x1,y1,w = width_fft(_fft,x)
        W = w
        _window = _window + 2
    return x1,y1,y2,W

def width_fft(_fft,x):
    _fft = _fft[:-1]
    _fft,_idx = np.unique(_fft,return_index = True)
    _iqr = stats.iqr(_fft)
    x1 = x[:-1]
    x1 = x[_idx]
    x1 = (x1 - min(x1)) / (max(x1) - min(x1))

    _quart75 = np.percentile(_fft,75)
    _quart25 = np.percentile(_fft,25)
    _idx = (_fft > _quart25 - 1.5 * _iqr) * (_fft < _quart75 + 1.5 * _iqr)
    _fft = _fft[_idx]
    x1 = x1[_idx]

    f = interp1d(_fft,x1)
    x2 = np.linspace(min(_fft),max(_fft),2*len(_fft))
    y2 = f(x2)

    f = UnivariateSpline(x2,y2, k=5)
    y3 = f(x2)
    y3 = savgol_filter(y3,11,2,deriv=1)
    _idx = argrelextrema(y3, np.greater)
    _idx2 = argrelextrema(y3, np.less)
    try:
        w = abs(np.min(np.max(x2[_idx]) - x2[_idx2]))
    except:
        w = 0
    return x2,y3,w



# Initialize Peaks (MASTER FUNCTION) ##########################################
def peak_init(x,y,y_raw):
    # x:     x Data
    # y:     y Data (smoothed + baseline corrected)
    # y_raw: y Data (baseline corrected)

    # Preset Fit Values
    pre_y = np.array([])
    parameter = np.array([])
    peak_x = np.array([])
    peak_y = np.array([])
    pre_residues = y
    rel_residues = 1
    k = -1
    while rel_residues > 0.05:
        k = k + 1
        if k == 0:
            pre_residues = savgol_filter(pre_residues,11,2)
        # Main Peaks
        main_x, main_y, main_p, filt = peakProminence(x, pre_residues)
        peak_x = np.append(peak_x,main_x[filt])
        peak_y = np.append(peak_y,main_y[filt])
        peak_x, peak_y = correctPosition(x,y_raw,peak_x)
        peak_x, indices = np.unique(peak_x,return_index=True)
        peak_y = peak_y[indices]
        charc_x, charc_y = characteristicPoints(x,y_raw,peak_x,peak_y,'right')
        charc_x2, charc_y2 = characteristicPoints(x,y_raw,peak_x,peak_y,'left')
        pre_widths = (charc_x - charc_x2)/2
        # check if peak_x and charc_x are equal
        peak_x = peak_x[pre_widths > 0]
        peak_y = peak_y[pre_widths > 0]
        pre_widths = pre_widths[pre_widths > 0]
        # Filter Water Bands
        min_width = 1.5
        min_wavenumber = 1800
        _filter = (peak_x > min_wavenumber) * (pre_widths < min_width)
        peak_x = peak_x[~_filter]
        peak_y = peak_y[~_filter]
        pre_widths = pre_widths[~_filter]
        # Delete Double Bands
        _filter = np.array([True],dtype=bool)
        for band in peak_x:
            # check distances
            _distances = np.sort(abs(peak_x - band))
            if (_distances[1] < 2) * (_filter[-1]):
                _filter = np.append(_filter,[False])#False
            else:
                _filter = np.append(_filter,[True])
        _filter = _filter[1:]
        peak_x = peak_x[_filter]
        peak_y = peak_y[_filter]
        pre_widths = pre_widths[_filter]
        # Filter Bands with Width or Height = 0
        _filter = (peak_y < np.std(np.diff(y_raw))*3) + (pre_widths == 0)
        peak_x = peak_x[~_filter]
        peak_y = peak_y[~_filter]
        pre_widths = pre_widths[~_filter]
        # Set Parameters
        parameter = np.zeros(len(peak_x)*3)
        parameter[0::3] = peak_x
        parameter[1::3] = peak_y
        parameter[2::3] = pre_widths
        # Correct heights of overlayed peaks
        # peaks*heights = y
        peaks = []
        for i in range(len(peak_x)):
            peaks.append(np.exp( -.5 * ((peak_x[i] - peak_x) / pre_widths)**2))
        try:
            heights = np.linalg.solve(peaks,peak_y)
            print('peak initialize optimal')
        except:
            heights = peak_y
            print('peak initialize not optimal')
        parameter[1::3] = heights
        # Filter negative heights
        _filter = np.repeat(parameter[1::3] > 0, 3)
        parameter = parameter[_filter]
        # Parameter for residues calculation
        parameter_estimated = np.zeros(len(parameter))
        parameter_estimated[0::3] = parameter[0::3]
        parameter_estimated[1::3] = parameter[1::3]#*1.2
        parameter_estimated[2::3] = parameter[2::3]#*1.2
        # Calculate residues
        y_estimated = gaussian_func(x,parameter_estimated)
        pre_residues = y - y_estimated
        pre_residues[pre_residues < 0] = 0
        rel_residues = sum(pre_residues) / sum(y)
        parameter_fitted = parameter
        if k == 3:
            rel_residues = 0
    # Filter Water Spikes
    pos = parameter_fitted[0::3]
    heights = parameter_fitted[1::3]
    widths = parameter_fitted[2::3]
    _filter = (pos > min_wavenumber) * (widths < min_width)
    _filter = _filter + (pos > min_wavenumber) * (heights < 0.05)
    pos = pos[~_filter]
    heights = heights[~_filter]
    widths = widths[~_filter]
    # Set Parameters
    results_parameter = np.zeros(len(pos)*5)
    results_parameter[0::5] = pos
    results_parameter[1::5] = heights
    results_parameter[2::5] = widths
    results_parameter[3::5] = 0
    results_parameter[4::5] = 0
    return results_parameter

def peakProminence(x,y):
    # find all local maxima in a window of n = 5 datapoints
    n = 5
    b1 = (y[(n-5):-(n-1)] < y[(n-4):-(n-2)])
    b2 = (y[(n-4):-(n-2)] < y[(n-3):-(n-3)])
    b3 = (y[(n-3):-(n-3)] > y[(n-2):-(n-4)])
    b4 = (y[(n-2):-(n-4)] > y[(n-1):])

    B1 = b1
    B2 = ~b2
    B3 = ~b3
    B4 = b4
    B5 = y[(n-3):-(n-3)] > y[(n-5):-(n-1)]
    B6 = y[(n-3):-(n-3)] > y[(n-1):]

    peak_index = np.append([False, False],b1 * b2 * b3 * b4 + B1 * B2 * B3 * B4 * B5 * B6)
    peak_index = np.append(peak_index, [False, False])

    if len(x) == len(peak_index):
        peak_x = x[peak_index]
        peak_y = y[peak_index]
    else:
        peak_x = x[peak_index[0:-1]]
        peak_y = y[peak_index[0:-1]]
    peak_prominence = np.zeros(len(peak_x))

    for i in range(0,len(peak_x)):
        # current peak
        curr_peak_x = peak_x[i]
        curr_peak_y = peak_y[i]
        # peaks that are higher than current peak
        if curr_peak_y < max(peak_y):
            higher_peaks_x = peak_x[peak_y > curr_peak_y]
            # closest higher peak
            idx = np.argmin(abs(curr_peak_x - higher_peaks_x))
            next_peak_x = higher_peaks_x[idx]
            if next_peak_x != curr_peak_x:
                # calculate prominence
                tmp_start = min([curr_peak_x , next_peak_x])
                tmp_stop = max([curr_peak_x , next_peak_x])
                tmp_y = y[(x > tmp_start) * (x < tmp_stop)]
                peak_prominence[i] = curr_peak_y - min(tmp_y)
        else:
            peak_prominence[i] = curr_peak_y - min(y)

    # mean diff y
    mean_diff_y = np.mean(abs(np.diff(y)))
    std_diff_y = np.std(abs(np.diff(y)))
    _filter = peak_prominence > mean_diff_y + 2 * std_diff_y

    return peak_x, peak_y, peak_prominence, _filter

# Correct Peak Positions ######################################################
def correctPosition(x,y,peak_x):
    corr_x = np.zeros(len(peak_x))
    corr_y = np.zeros(len(peak_x))
    for j in range(len(peak_x)):
        i = np.argmin(abs(x - peak_x[j]))
        actual_pos = x[i]
        actual_y = y[i]
        if (i > 1) * (i<len(x)-3):
            actual_test_range = y[i-2:i+2]
            if max(actual_test_range) == actual_y:
                flag = False
            else:
                flag = True
        else:
            flag = False
        while flag:
            i = np.argmax(actual_test_range) - 2 + i
            actual_pos = x[i]
            actual_y = y[i]
            if (i > 1) * (i<len(x)-3):
                actual_test_range = y[i-2:i+2]
                if max(actual_test_range) == actual_y:
                    flag = False
                else:
                    flag = True
            else:
                flag = False
        corr_x[j] = actual_pos
        corr_y[j] = actual_y
    return corr_x,corr_y

# Find Characteristic Points ##################################################
def characteristicPoints(x,y,peak_x,peak_y,direction='right'):
    charc_x = np.zeros(len(peak_x))
    charc_y = np.zeros(len(peak_x))
    if direction == 'right':
        if len(peak_x) > 0:
            for i in range(len(peak_x)-1):
                x_range = x[(x >= peak_x[i]) * (x <= peak_x[i+1])]
                y_range = y[(x >= peak_x[i]) * (x <= peak_x[i+1])]
                fwhm_y = y_range[y_range <= peak_y[i]/2]
                fwhm_x = x_range[y_range <= peak_y[i]/2]
                try:
                    fwhm_y = fwhm_y[0]
                    fwhm_x = fwhm_x[0]
                except:
                    fwhm_y = min(y_range)
                    fwhm_x = x_range[np.argmin(y_range)]
                charc_x[i] = fwhm_x
                charc_y[i] = fwhm_y


            i = len(peak_x) - 1
            x_range = x[x >= peak_x[i]]
            y_range = y[x >= peak_x[i]]
            fwhm_y = y_range[y_range <= peak_y[i]/2]
            fwhm_x = x_range[y_range <= peak_y[i]/2]
            try:
                fwhm_y = fwhm_y[0]
                fwhm_x = fwhm_x[0]
            except:
                fwhm_y = min(y_range)
                fwhm_x = x_range[np.argmin(y_range)]
            charc_x[i] = fwhm_x
            charc_y[i] = fwhm_y

    if direction == 'left':
        if len(peak_x) > 0:
            for i in range(len(peak_x)):
                if i == 0:
                    charc_x[i] = 0
                    charc_y[i] = 0
                else:
                    x_range = x[(x <= peak_x[i]) * (x >= peak_x[i-1])]
                    y_range = y[(x <= peak_x[i]) * (x >= peak_x[i-1])]
                    fwhm_y = y_range[y_range <= peak_y[i]/2]
                    fwhm_x = x_range[y_range <= peak_y[i]/2]
                    try:
                        fwhm_y = fwhm_y[-1]
                        fwhm_x = fwhm_x[-1]
                    except:
                        fwhm_y = min(y_range)
                        fwhm_x = x_range[np.argmin(y_range)]
                    charc_x[i] = fwhm_x
                    charc_y[i] = fwhm_y


            i = 0
            x_range = x[x <= peak_x[i]]
            y_range = y[x <= peak_x[i]]
            fwhm_y = y_range[y_range <= peak_y[i]/2]
            fwhm_x = x_range[y_range <= peak_y[i]/2]
            try:
                fwhm_y = fwhm_y[-1]
                fwhm_x = fwhm_x[-1]
            except:
                fwhm_y = min(y_range)
                fwhm_x = x_range[np.argmin(y_range)]
            charc_x[i] = fwhm_x
            charc_y[i] = fwhm_y


    return charc_x, charc_y

# Gaussian Function ###########################################################
def gaussian_func(x,parameter):
    x0 = parameter[0::3]
    h = parameter[1::3]
    w = parameter[2::3]
    x = np.repeat(np.transpose(np.matrix(x)), len(x0), axis = 1)
    # Function
    x = x - x0
    gaussian = np.multiply(np.exp(np.square(np.divide(x,w)) * - 0.5), h)
    return np.squeeze(np.asarray(np.sum(gaussian, axis = 1)))


# BASELINE
def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def baseline_als_auto(x,y):
    y = (y-min(y))/(max(y)-min(y))
    idx = np.argmin(abs(x-2400))
    idx2 = np.argmin(abs(x-1700))
    idx3 = np.argmin(abs(x-2700))
    p = 0.0001
    lam = 1E5
    lam2 = 1E6
    lam3 = 1E8
    lam4 = 1E3
    baseline = np.zeros([len(x),4])
    baseline[baseline == 0] = np.nan
    baseline[0:idx+1,0] = baseline_als(y[0:idx+1],lam2,p)
    baseline[idx:,1] = baseline_als(y[idx:],lam,p)
    baseline[idx2:idx3,2] = baseline_als(y[idx2:idx3],lam,p)
    baseline[idx:,3] = baseline_als(y[idx:],lam3,p)
    baseline = np.nanmax(baseline,axis=1)
    baseline = baseline_als(baseline,lam4,p)
    # z2 = np.array([])
    # x_fast = x[0::2]
    # y_fast = y[0::2]
    # idx = np.argmin(abs(x_fast-2400))
    # LAM = 1.5**np.linspace(26,60,35)
    # for lam in LAM:
    #     p = 0.0001
    #     baseline = baseline_als(y_fast,lam, p)
    #     z2 = np.append(z2,abs(y_fast[0]-baseline[0])+abs(y_fast[np.argmin(y_fast[x_fast<1500])]-baseline[np.argmin(y_fast[x_fast<1500])]) + abs(y_fast[idx]-baseline[idx])+abs(y_fast[-np.argmin(y_fast[x_fast>3900])]-baseline[-np.argmin(y_fast[x_fast>3900])])+abs(y_fast[-1]-baseline[-1]))
    # loc_min, properties = find_peaks(-z2,prominence=0.0,width=0)
    # loc_min2, properties2 = find_peaks(np.diff(z2),prominence=0.0,width=0)
    # idx = np.argmax(properties["prominences"]*properties["widths"])
    # idx2 = np.argmax(properties2["prominences"]*properties2["widths"])
    # if np.argmin(z2[[loc_min[idx],loc_min2[idx2]]]) == 1:
    #     idx = idx2
    #     loc_min = loc_min2
    #     print('changed')
    # baseline = baseline_als(y,LAM[loc_min[idx]], p)
    return y,baseline

######################################################
### SAVE JCAMP-DX ####################################
######################################################
def save_jcamp(x,y,xraw,yraw,name,processing):
    FLAG = True
    iii = 1
    while True:
        if name[-iii] == '/':
            break
        else:
            iii += 1
    try:
        file = open(name,'w')
    except:
        print('permission denied - please change the output directory')
        FLAG = False
    if FLAG:
        file.write('##TITLE=' + name[-iii+1:-4] + '\n')
        file.write('##JCAMP-DX=1.0\n')
        file.write('##DATATYPE=LINK\n')
        file.write('##BLOCKS=2\n')

        file.write('##TITLE=processed: ' + name[-iii+1:-4] + '\n')
        file.write('##JCAMP-DX=1.0\n')
        file.write('##BLOCK_ID=1\n')
        file.write('##DATATYPE=INFRARED SPECTRUM\n')
        file.write('##ORIGIN=HS-Niederrhein\n')
        file.write('##OWNER=WELTBUNT\n')
        file.write('##LONGDATE=2019/12/12\n')
        file.write('##TIME=00:00:00\n')
        file.write('##SPECTROMETER/DATA SYSTEM=WELTBUNT SPEC-2-LIB\n')
        file.write('##DATA PROCESSING=' + processing + '\n')
        file.write('##COMMENTS=\n')

        file.write('##RESOLUTION=4.000\n')
        file.write('##XUNITS=1/CM\n')
        file.write('##YUNITS=ABSORBANCE\n')
        file.write('##FIRSTX=' + str(x[0]) + '\n')
        file.write('##LASTX=' + str(x[-1]) + '\n')
        file.write('##FIRSTY=' + str(y[0]) + '\n')
        file.write('##MAXX=' + str(max(x)) + '\n')
        file.write('##MINX=' + str(min(x)) + '\n')
        file.write('##MINY=' + str(min(y)) + '\n')
        file.write('##MAXY=' + str(max(y)) + '\n')

        file.write('##XFACTOR=1.000000\n')
        file.write('##YFACTOR=1.000000E-009\n')

        file.write('##NPOINTS=' + str(len(x)) + '\n')
        file.write('##DELTAX=' + str(abs(x[0]-x[1])) + '\n')

        file.write('##XYDATA=(X++(Y..Y))\n')
        for i in range(int(len(x)/6)):
            tmp_data_str = str(x[i*6])
            tmp_data_str = tmp_data_str[0:7]
            for j in range(7):
                tmp_data_str = tmp_data_str + ' ' + str(int(round(y[i*6+j]*1E9)))
            file.write(tmp_data_str+'\n')

        try:
            i = int(len(x)/6)+0
            tmp_data_str = str(x[i*6])
            tmp_data_str = tmp_data_str[0:7]
            for j in range(np.mod(len(x),6)):
                tmp_data_str = tmp_data_str + ' ' + str(int(round(y[i*6+j]*1E9)))
            file.write(tmp_data_str+'\n')
        except:
            pass
        file.write('##END=\n')

        file.write('##TITLE=raw: ' + name[-iii+1:-4] + '\n')
        file.write('##JCAMP-DX=1.0\n')
        file.write('##BLOCK_ID=2\n')
        file.write('##DATATYPE=INFRARED SPECTRUM\n')
        file.write('##ORIGIN=HS-Niederrhein\n')
        file.write('##OWNER=WELTBUNT\n')
        file.write('##LONGDATE=2019/12/12\n')
        file.write('##TIME=00:00:00\n')
        file.write('##SPECTROMETER/DATA SYSTEM=WELTBUNT SPEC-2-LIB\n')
        file.write('##DATA PROCESSING=' + processing + '\n')
        file.write('##COMMENTS=\n')

        file.write('##RESOLUTION=4.000\n')
        file.write('##XUNITS=1/CM\n')
        file.write('##YUNITS=ABSORBANCE\n')
        file.write('##FIRSTX=' + str(xraw[0]) + '\n')
        file.write('##LASTX=' + str(xraw[-1]) + '\n')
        file.write('##FIRSTY=' + str(yraw[0]) + '\n')
        file.write('##MAXX=' + str(max(xraw)) + '\n')
        file.write('##MINX=' + str(min(xraw)) + '\n')
        file.write('##MINY=' + str(min(yraw)) + '\n')
        file.write('##MAXY=' + str(max(yraw)) + '\n')

        file.write('##XFACTOR=1.000000\n')
        file.write('##YFACTOR=1.000000E-009\n')

        file.write('##NPOINTS=' + str(len(xraw)) + '\n')
        file.write('##DELTAX=' + str(abs(xraw[0]-xraw[1])) + '\n')

        file.write('##XYDATA=(X++(Y..Y))\n')
        for i in range(int(len(xraw)/6)):
            tmp_data_str = str(xraw[i*6])
            tmp_data_str = tmp_data_str[0:7]
            for j in range(7):
                tmp_data_str = tmp_data_str + ' ' + str(int(round(yraw[i*6+j]*1E9)))
            file.write(tmp_data_str+'\n')

        try:
            i = int(len(xraw)/6)+0
            tmp_data_str = str(xraw[i*6])
            tmp_data_str = tmp_data_str[0:7]
            for j in range(np.mod(len(xraw),6)):
                tmp_data_str = tmp_data_str + ' ' + str(int(round(yraw[i*6+j]*1E9)))
            file.write(tmp_data_str+'\n')
        except:
            pass
        file.write('##END=\n')
        file.write('##END=\n')
        file.close()

        """ SAVE PLOT """
        fig = plt.figure(figsize=[16,9],dpi=180)
        ax1 = fig.add_subplot(211)
        ax1.plot(xraw,yraw)
        ax1.set_xlabel('Wavenumber (1/cm)', fontsize=9, fontweight='bold')
        ax1.set_ylabel('Absorbance (a.u.)', fontsize=9, fontweight='bold')
        ax1.set_title('Raw Data (' + name[name.rfind('/')+1:-4] + ')',fontsize=9,fontweight='bold',loc='left',fontstyle='italic')
        ax1.set_xlim(np.max(xraw),np.min(xraw))
        # ax1.set_xticklabels('Wavenumber (1/cm)',fontsize=8)
        # ax1.set_yticklabels('Absorbance (a.u.)',fontsize=8)
        ax2 = fig.add_subplot(212)
        ax2.plot(x,y)
        ax2.set_xlabel('Wavenumber (1/cm)', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Absorbance (a.u.)', fontsize=9, fontweight='bold')
        ax2.set_title('Processed Data (' + processing + ')',fontsize=9,fontweight='bold',loc='left',fontstyle='italic')
        ax2.set_xlim(np.max(xraw),np.min(xraw))
        fig.tight_layout()
        fig.savefig(name + '.pdf',dpi=180)
        # fig.show()


#### LINEAR BASELINE Correction
def lin_base(y_processed):
    y2 = np.transpose(y_processed)
    baseline = np.linspace(y2[0],y2[-1],len(y2))
    tmp_residues = y_processed - baseline
    base_knots = np.array([0])
    base_flag = True
    k = 0
    while base_flag:
        if np.min(tmp_residues) < 0:
            tmp_idx = np.argmin(tmp_residues)
            base_knots = np.append(base_knots,tmp_idx)
            base_knots = np.sort(base_knots)
            tmp_baseline = np.array([])
            for i in range(k+1):
                tmp = np.linspace(y2[base_knots[i]],y2[base_knots[i+1]],int(base_knots[i+1]-base_knots[i]+0))
                tmp_baseline = np.append(tmp_baseline,tmp)
            tmp_baseline = np.append(tmp_baseline,np.linspace(y2[base_knots[i+1]],y2[-1],int(len(y2)-len(tmp_baseline))))
            baseline = tmp_baseline
            tmp_residues = y_processed - baseline
        else:
            base_flag = False
        k += 1
        if k == 10:
            base_flag = False
    return baseline
