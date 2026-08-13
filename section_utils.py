import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d, RegularGridInterpolator, griddata
from scipy import ndimage
import sys
from scipy.signal import stft, istft, hilbert

SMALL_VAL = 1.0e-10

def get_taper(t_range, t_arr):
  fac = np.ones_like(t_arr)
  if (t_range[0] is not None and t_range[1] is not None):
    fac[t_arr<=t_range[0]] = 0.0
    ind = np.logical_and(t_arr>=t_range[0], t_arr<=t_range[1])
    fac[ind] = fac[ind] * (1.0-np.cos((t_arr[ind]-t_range[0])/(t_range[1]-t_range[0])*np.pi)) / 2.0
  if (t_range[2] is not None and t_range[3] is not None):
    fac[t_arr>=t_range[3]] = 0.0
    ind = np.logical_and(t_arr>=t_range[2], t_arr<=t_range[3])
    fac[ind] = fac[ind] * (1.0-np.cos((t_arr[ind]-t_range[3])/(t_range[2]-t_range[3])*np.pi)) / 2.0
  return fac

def split_line(line):
  # split line with continuous multiple spaces
  # return a list of strings
  return [_ for _ in line.strip().split(' ') if _ != '']

def moving_average(x, n):
  return np.convolve(x, np.ones(n)/n, mode='same')

def moving_average_2d(x, kernel_size):
  norm = np.sum(np.ones(kernel_size))
  return signal.convolve2d(x, np.ones(kernel_size) / norm, mode='same')

class WaveformSection:
  def __init__(self):
    self.waveforms = None
    self.t = None
    self.coord_list = None
    self.t_type = 'time'
    self.coord_type = 'space'

  def from_numpy(self, waveforms, t_arr, coord_list, t_type='time', coord_type='space'):
    self.waveforms = waveforms
    self.t = t_arr
    self.coord_list = coord_list
    self.t_type = t_type
    self.coord_type = coord_type

  def __sub__(self, other):
    sub = other.interpolate(self.t)
    for i_wave in range(len(sub.waveforms)):
      sub.waveforms[i_wave] = self.waveforms[i_wave] - sub.waveforms[i_wave]
    return sub

  def __add__(self, other):
    sub = other.interpolate(self.t)
    for i_wave in range(len(sub.waveforms)):
      sub.waveforms[i_wave] = self.waveforms[i_wave] + sub.waveforms[i_wave]
    return sub

  def _zeros_like(self):
    sec_new = WaveformSection()
    sec_new.coord_list = np.copy(self.coord_list)
    sec_new.t = np.copy(self.t)
    sec_new.t_type = self.t_type
    sec_new.coord_type = self.coord_type
    sec_new.waveforms = np.zeros_like(self.waveforms)
    return sec_new

  def shift(self, dt):
    self.t += dt

  def scale(self, fac):
    for i_wave in range(len(self.waveforms)):
      self.waveforms[i_wave] = self.waveforms[i_wave] * fac

  def taper(self, taper_range, axis='t', x_axis_trans=(lambda coord:coord[0])):
    if (axis=='t'):
      fac_v = np.reshape(self.t, newshape=(1, len(self.t)))
    elif (axis=='x'):
      x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
      fac_v = np.reshape(x_arr, newshape=(len(x_arr), 1))
    fac = get_taper(taper_range, fac_v)
    #print(fac)
    self.waveforms = self.waveforms * fac

  def wiener(self, size):
    self.waveforms = signal.wiener(self.waveforms, mysize=size)

  def gaussian_filter(self, sigma):
    self.waveforms = ndimage.gaussian_filter(self.waveforms, sigma=sigma)

  def get_envelope(self):
    sec_new = self._zeros_like()
    sec_new.waveforms = abs(signal.hilbert(self.waveforms, axis=-1))
    return sec_new

  def get_stalta(self, tshort, tlong):
    dt = self.t[1] - self.t[0]
    nshort = int(tshort / dt)
    nlong = int(tlong / dt)
    ma_short = np.zeros_like(self.waveforms)
    ma_long = np.zeros_like(self.waveforms)
    sec_new = self._zeros_like()
    for iw, wave in enumerate(self.waveforms):
      ma_short = moving_average(abs(wave), nshort)
      ma_long = moving_average(abs(wave), nlong)
      norm = abs(ma_long).max() * SMALL_VAL
      sec_new.waveforms[iw, :] = ma_short / (ma_long + norm)
    return sec_new

  def interpolate(self, t_new):
    """
    not in-place
    """
    sec_new = self._zeros_like()
    for iw, wave in enumerate(self.waveforms):
      f = interp1d(self.t, wave, fill_value=0.0, bounds_error=False)
      sec_new.waveforms[iw, :] = f(t_new)
    sec_new.t = t_new
    return sec_new

  def slice(self, start, end, stride):
    sec_new = self._zeros_like()
    sec_new.waveforms = self.waveforms[start:end:stride]
    sec_new.coord_list = self.coord_list[start:end:stride]
    return sec_new

  def trace_normalize(self, normalize_by='max_abs'):
    if normalize_by == 'max_abs':
      norm = np.amax(abs(self.waveforms), axis=-1, keepdims=True)
    elif normalize_by == 'rms':
      norm = np.sqrt(np.mean(self.waveforms**2, axis=-1, keepdims=True))
    self.waveforms /= norm

  def cut_in_time(self, t1, t2):
    ind = np.logical_and(self.t >= t1, self.t <= t2)
    self.t = self.t[ind]
    #self.waveforms = [wave[ind] for wave in self.waveforms]
    self.waveforms = self.waveforms[:,ind]

  def cut_in_x(self, x1, x2, x_axis_trans=(lambda coord:coord[0])):
    x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
    ind = np.logical_and(x_arr >= x1, x_arr <= x2)
    self.waveforms = self.waveforms[ind,:]
    self.coord_list = self.coord_list[ind,:]

  def find_ntrace_closest_to_x(self, x, x_axis_trans=(lambda coord:coord[0])):
    x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
    return np.argmin(abs(x_arr-x))

  def filter(self, freq_range, verbose=1):
    if (not isinstance(freq_range, tuple)):
      sys.exit(f"{freq_range} is not a tuple\n")
    if (len(freq_range)!=2):
      sys.exit(f"incorrect frequency range {freq_range}\n")
    if (verbose==1): print(f'apply filter {freq_range}\n')
    s = self.t[1] - self.t[0]
    if (not (freq_range[0] >= 0.0)):
      sos = signal.butter(4, freq_range[1] * s * 2, 'lowpass', output='sos')
    else:
      sos = signal.butter(4, [freq_range[0] * s * 2, freq_range[1] * s * 2], 'bandpass', output='sos')
    #for i_wave in range(len(self.waveforms)):
    #  wave = self.waveforms[i_wave]
    #  self.waveforms[i_wave] = signal.sosfiltfilt(sos, wave, padtype=None)
    self.waveforms = signal.sosfiltfilt(sos, self.waveforms, padtype=None, axis=-1)

  def filter_median(self, kernel_size, filter_out=False):
    if filter_out:
      self.waveforms -= signal.medfilt2d(self.waveforms, kernel_size=kernel_size)
    else:
      self.waveforms = signal.medfilt2d(self.waveforms, kernel_size=kernel_size)

  def filter_moving_average(self, kernel_size, filter_out=False):
    if filter_out:
      self.waveforms -= moving_average_2d(self.waveforms, kernel_size=kernel_size)
    else:
      self.waveforms = moving_average_2d(self.waveforms, kernel_size=kernel_size)

  def filter_fk_median(self, kernel_size, verbose=1):
    # assuming uniform sampling in x and t
    nx, nt = self.waveforms.shape
    waveforms_fft = np.fft.fft(np.fft.fft(self.waveforms, n=nt*2, axis=1), nx*2, axis=0)
    waveforms_fft_spec = abs(waveforms_fft)
    waveforms_fft_spec_filter = signal.medfilt2d(waveforms_fft_spec, kernel_size=kernel_size)
    ind = (waveforms_fft_spec > waveforms_fft_spec.max() * SMALL_VAL)
    waveforms_fft_filter = np.copy(waveforms_fft)
    waveforms_fft_filter[ind] = waveforms_fft[ind] / waveforms_fft_spec[ind] * waveforms_fft_spec_filter[ind]
    self.waveforms = np.real(np.fft.ifft(np.fft.ifft(waveforms_fft_filter, axis=1), axis=0))[0:nx, 0:nt]
    #plt.pcolormesh(abs(waveforms_fft_filter-waveforms_fft), vmin=0.0, vmax=0.01*waveforms_fft_spec.max(), cmap='gray_r')
    #plt.show()
    #plt.close()

  def filter_fk(self, v_range, x_axis_trans=(lambda coord:coord[0]), use_abs=False, verbose=1):
    # assuming uniform sampling in x and t
    if (not isinstance(v_range, tuple)):
      sys.exit(f"{v_range} is not a tuple\n")
    if (len(v_range)!=4):
      sys.exit(f"incorrect velocity range {v_range}\n")
    if (verbose==1): print(f'apply fk filter {v_range}\n')
    dt = self.t[1] - self.t[0]
    x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
    dx = x_arr[1] - x_arr[0]
    f_arr = np.fft.fftfreq(len(self.t)*2, d=dt)
    k_arr = np.fft.fftfreq(len(x_arr)*2, d=dx)
    k_arr[abs(k_arr)<(1.0/dx/len(x_arr)*SMALL_VAL)] = 1.0/dx/len(x_arr)*SMALL_VAL
    K_arr, F_arr = np.meshgrid(k_arr, f_arr, indexing='ij')
    V_arr = - F_arr / K_arr
    if use_abs: V_arr = abs(V_arr)
    fac = get_taper(v_range, V_arr)
    waveforms_fft = np.fft.fft(np.fft.fft(self.waveforms, n=len(self.t)*2, axis=1), len(x_arr)*2, axis=0)
    waveforms_fft *= fac
    self.waveforms = np.real(np.fft.ifft(np.fft.ifft(waveforms_fft, axis=1), axis=0))[0:len(x_arr), 0:len(self.t)]

  def plot_velocity_analysis(self, *args, v_arr, h_arr, ax=None,
                             x_axis_trans=(lambda coord:coord[0]),
                             saturation=1.0, vmax=None, vmin=None, **kwargs):
    if ax is None: ax = plt.gca()
    nx, nt = self.waveforms.shape
    dt = self.t[1] - self.t[0]
    x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
    dx = x_arr[1] - x_arr[0]
    V, H = np.meshgrid(v_arr, h_arr, indexing='ij')
    nv = v_arr.shape[0]
    nh = h_arr.shape[0]
    Vr = np.reshape(V, newshape=(nv, nh, 1))
    Hr = np.reshape(H, newshape=(nv, nh, 1))
    X = np.tile(np.reshape(x_arr, newshape=(1,1,-1)), (nv, nh, 1))
    T = np.sqrt(X*X+4.0*Hr*Hr) / Vr
    interp = RegularGridInterpolator((x_arr, self.t), self.waveforms, bounds_error=False, fill_value=0.0)
    img = interp((X, T))
    img = np.sum(img, axis=-1) / nx
    img = abs(signal.hilbert(img, axis=-1))
    norm = abs(img).max() * saturation
    if vmax is None: vmax = norm
    if vmin is None: vmin = -norm
    return ax.pcolormesh(v_arr, h_arr, img.T, *args, vmin=vmin, vmax=vmax, **kwargs)
  def plot_spec_ax_waterfall(self, *args, ax=None, freq_range=None,
                              x_axis_trans=(lambda coord:coord[0]),
                              saturation=1.0, vmax=None, vmin=None,
                              **kwargs):
    if ax is None: ax = plt.gca()
    nx, nt = self.waveforms.shape
    dt = self.t[1] - self.t[0]
    x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
    f_arr = np.fft.fftfreq(len(self.t)*2, d=dt)
    waveforms_fft = np.fft.fft(self.waveforms, n=len(self.t)*2, axis=1) * dt
    if freq_range is None: freq_range = (f_arr[0], f_arr[len(self.t)-1])
    ind_f = np.logical_and(f_arr >= freq_range[0], f_arr <= freq_range[1])
    norm = abs(waveforms_fft[:,ind_f]).max()
    norm = norm * saturation
    if vmax is None: vmax = norm
    if vmin is None: vmin = -norm
    return  ax.pcolormesh(x_arr, f_arr[ind_f],
                  abs(waveforms_fft[:,ind_f]).T, *args,
                  vmin=vmin, vmax=vmax, **kwargs)
  
  def get_spec(self, freq_range=None):
    dt = self.t[1] - self.t[0]
    f_arr = np.fft.fftfreq(len(self.t)*2, d=dt)
    waveforms_fft = np.fft.fft(self.waveforms, n=len(self.t)*2, axis=1) * dt
    if freq_range is None: freq_range = (f_arr[0], f_arr[len(self.t)-1])
    ind_f = np.logical_and(f_arr >= freq_range[0], f_arr <= freq_range[1])
    sec_spec = self._zeros_like()
    sec_spec.t = f_arr[ind_f]
    sec_spec.waveforms = abs(waveforms_fft[:,ind_f])
    sec_spec.t_type = 'freq'
    return sec_spec

  def plot_fk_spec_ax_waterfall(self, *args, ax=None, freq_range=None,
                                 x_axis_trans=(lambda coord:coord[0]),
                                 k_range=None,
                                 saturation=1.0,
                                 label=None, **kwargs):
    if ax is None: ax = plt.gca()
    nx, nt = self.waveforms.shape
    dt = self.t[1] - self.t[0]
    x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
    dx = x_arr[1] - x_arr[0]
    f_arr = np.fft.fftfreq(len(self.t)*2, d=dt)
    k_arr = np.fft.fftfreq(len(x_arr)*2, d=dx) 
    waveforms_fft = np.fft.fft(np.fft.fft(self.waveforms, n=len(self.t)*2, axis=1), len(x_arr)*2, axis=0)
    if k_range is None: k_range=(k_arr[0], k_arr[len(x_arr)-1])
    if freq_range is None: freq_range = (f_arr[0], f_arr[len(self.t)-1])

    ind_f = np.logical_and(f_arr >= freq_range[0], f_arr <= freq_range[1])
    ind_k = np.logical_and(k_arr >= k_range[0], k_arr <= k_range[1])
    norm = abs(waveforms_fft[ind_k,:][:,ind_f]).max()
    norm = norm * saturation
    return  ax.pcolormesh(k_arr[ind_k], f_arr[ind_f],
                  abs(waveforms_fft[ind_k,:][:,ind_f]).T, *args,
                  vmin=0.0, vmax=norm, **kwargs)

  def reduce_time(self, reduce_by, t_range, x_axis_trans=(lambda coord:coord[0])):
    dt = self.t[1] - self.t[0]
    x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
    sec_reduce = self._zeros_like()
    sec_reduce.t = np.arange(t_range[0], t_range[1], dt)
    t_reduce_arr = reduce_by(x_arr)
    waveforms = [interp1d(self.t-t_reduce, w, bounds_error=False, 
                          fill_value=0.0)(sec_reduce.t) \
                 for t_reduce, w in zip(t_reduce_arr, self.waveforms)
                ]
    sec_reduce.waveforms = np.array(waveforms)
    return sec_reduce

  def nmo_correction(self, travel_time, xr_arr, h_arr, x_axis_trans=(lambda coord:coord[0])):
    x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
    sec_nmo = self._zeros_like()
    sec_nmo.t = np.copy(h_arr)
    sec_nmo.t_type = 'depth'
    nxr = xr_arr.shape[0]
    nh = h_arr.shape[0]
    Xr_arr, H_arr = np.meshgrid(xr_arr, h_arr, indexing='ij')
    #Xr_arr = Xr_arr.flatten()
    #H_arr = H_arr.flatten()
    T_arr, X_arr = travel_time(Xr_arr, H_arr)
    #T_arr = np.reshape(T_arr, newshape=(nxr, nh))
    #Xr_arr = np.reshape(Xr_arr, newshape=(nxr, nh))
    sec_nmo.coord_list = np.reshape(xr_arr, newshape=(nxr, 1)) 
    interp = RegularGridInterpolator((x_arr, self.t), self.waveforms, bounds_error=False, fill_value=0.0)
    sec_nmo.waveforms = interp((X_arr, T_arr))
    return sec_nmo

  def plot_fv_spec_ax_waterfall(self, *args, ax=None, freq_range=None,
                                 x_axis_trans=(lambda coord:coord[0]),
                                 v_arr=None,
                                 saturation=1.0, vmax=None, 
                                 label=None, **kwargs):
    if ax is None: ax = plt.gca()
    nx, nt = self.waveforms.shape
    dt = self.t[1] - self.t[0]
    x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
    dx = x_arr[1] - x_arr[0]
    f_arr = np.fft.fftfreq(len(self.t)*2, d=dt)
    k_arr = np.fft.fftfreq(len(x_arr)*2, d=dx)
    k_arr[abs(k_arr)<(1.0/dx/len(x_arr)*SMALL_VAL)] = 1.0/dx/len(x_arr)*SMALL_VAL
    K_arr, F_arr = np.meshgrid(k_arr, f_arr, indexing='ij')
    V_arr = - F_arr / K_arr 
    waveforms_fft = np.fft.fft(np.fft.fft(self.waveforms, n=len(self.t)*2, axis=1), len(x_arr)*2, axis=0)
    if freq_range is None: freq_range = (f_arr[0], f_arr[len(self.t)-1])
    if v_arr is None: v_arr=np.linspace(k_arr[0]/freq_range[1], k_arr[len(x_arr)-1]/freq_range[0], (k_arr[1]-k_arr[0])/(f_arr[1]-f_arr[0]))
    #V1_arr, F1_arr = np.meshgrid(v_arr, f_arr, indexing='ij')
    #waveforms_fv = griddata([V_arr.flatten(), F_arr.flatten()], waveforms_fft.flatten(), (V1_arr, F1_arr), method='linear')
    ind_f = np.logical_and(f_arr[:len(self.t)] >= freq_range[0], f_arr[:len(self.t)] <= freq_range[1])
    waveforms_fv = np.zeros(shape=(len(v_arr),len(f_arr[:len(self.t)])), dtype=complex)
    for ixf in range(len(f_arr[:len(self.t)])):
      interp = interp1d(V_arr[:,ixf], waveforms_fft[:,ixf], bounds_error=False, fill_value=0.0)
      waveforms_fv[:,ixf] = interp(v_arr)
    norm = abs(waveforms_fv[:,ind_f]).max()
    norm = norm * saturation
    if vmax is None: vmax = norm
    return  ax.pcolormesh(f_arr[:len(self.t)][ind_f], v_arr,
                  abs(waveforms_fv[:,ind_f]), *args,
                  vmin=0.0, vmax=vmax, **kwargs)

  def plot_fv_diff_spec_ax_waterfall(self, other, threshold=0.1, amp=True,
                                 *args, ax=None, freq_range=None,
                                 x_axis_trans=(lambda coord:coord[0]),
                                 v_arr=None,
                                 saturation=1.0, vmin=None, vmax=None,
                                 label=None, **kwargs):
    if ax is None: ax = plt.gca()
    nx, nt = self.waveforms.shape
    dt = self.t[1] - self.t[0]
    x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
    dx = x_arr[1] - x_arr[0]
    f_arr = np.fft.fftfreq(len(self.t)*2, d=dt)
    k_arr = np.fft.fftfreq(len(x_arr)*2, d=dx)
    k_arr[abs(k_arr)<(1.0/dx/len(x_arr)*SMALL_VAL)] = 1.0/dx/len(x_arr)*SMALL_VAL
    K_arr, F_arr = np.meshgrid(k_arr, f_arr, indexing='ij')
    V_arr = - F_arr / K_arr
    waveforms_fft = np.fft.fft(np.fft.fft(self.waveforms, n=len(self.t)*2, axis=1), len(x_arr)*2, axis=0)
    waveforms_other_fft = np.fft.fft(np.fft.fft(other.waveforms, n=len(self.t)*2, axis=1), len(x_arr)*2, axis=0)
    if freq_range is None: freq_range = (f_arr[0], f_arr[len(self.t)-1])
    if v_arr is None: v_arr=np.linspace(k_arr[0]/freq_range[1], k_arr[len(x_arr)-1]/freq_range[0], (k_arr[1]-k_arr[0])/(f_arr[1]-f_arr[0]))
    #V1_arr, F1_arr = np.meshgrid(v_arr, f_arr, indexing='ij')
    #waveforms_fv = griddata([V_arr.flatten(), F_arr.flatten()], waveforms_fft.flatten(), (V1_arr, F1_arr), method='linear')
    ind_f = np.logical_and(f_arr[:len(self.t)] >= freq_range[0], f_arr[:len(self.t)] <= freq_range[1])
    waveforms_fv = np.zeros(shape=(len(v_arr),len(f_arr[:len(self.t)])), dtype=complex)
    for ixf in range(len(f_arr[:len(self.t)])):
      interp = interp1d(V_arr[:,ixf], waveforms_fft[:,ixf], bounds_error=False, fill_value=0.0)
      waveforms_fv[:,ixf] = interp(v_arr)
    waveforms_other_fv = np.zeros(shape=(len(v_arr),len(f_arr[:len(self.t)])), dtype=complex)
    for ixf in range(len(f_arr[:len(self.t)])):
      interp = interp1d(V_arr[:,ixf], waveforms_other_fft[:,ixf], bounds_error=False, fill_value=0.0)
      waveforms_other_fv[:,ixf] = interp(v_arr)
    norm = abs(waveforms_fv[:,ind_f]).max()
    ind_divide = (abs(waveforms_fv) > norm * threshold)
    waveforms_diff_fv = np.zeros(shape=(len(v_arr),len(f_arr[:len(self.t)])), dtype=float)
    if amp: 
      waveforms_diff_fv[ind_divide] = abs(waveforms_other_fv[ind_divide] / waveforms_fv[ind_divide]) - 1.0
    else:
      waveforms_diff_fv[ind_divide] = np.angle(waveforms_other_fv[ind_divide] / waveforms_fv[ind_divide])
    norm = abs(waveforms_diff_fv).max()
    print(norm)
    norm = norm * saturation
    if vmax is None: vmax = norm
    if vmin is None: vmin = -norm
    return  ax.pcolormesh(f_arr[:len(self.t)][ind_f], v_arr,
                  waveforms_diff_fv[:,ind_f], *args,
                  vmin=vmin, vmax=vmax, **kwargs)


  def plot_waveforms_ax_fill(self, *args, ax=None, time_range=None, offset=None,
                          x_axis_trans=(lambda coord:coord[0]),
                          is_time_axis_x = False,
                          x_range = None,
                          fill=True, fill_color=('blue', 'red'),
                          norm_with=None, normalize=1.0, label=None, **kwargs):
    #fig = plt.figure(num=fig_num) # pull out the existed figure
    if ax is None: ax = plt.gca()
    if offset is None: offset = [0.0] * len(self.waveforms)
    if time_range is None: time_range = (self.t[0], self.t[-1])
    if norm_with is None: norm_with = self
    max_val = 0.0
    x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
    for i_wave in range(len(self.waveforms)):
      wave = norm_with.waveforms[i_wave]
      ind = np.logical_and((norm_with.t-offset[i_wave] >= time_range[0]), (norm_with.t-offset[i_wave] <= time_range[1]))
      if (np.amax(np.absolute(wave)) > max_val):
        max_val = np.amax(np.absolute(wave[ind]))
    for i_wave in range(len(self.waveforms)):
      x_axis_val = x_arr[i_wave]
      if (x_range is not None):
        if x_axis_val > x_range[1] or x_axis_val < x_range[0]: continue
      ind = np.logical_and((self.t-offset[i_wave] >= time_range[0]), (self.t-offset[i_wave] <= time_range[1]))
      wave_norm = self.waveforms[i_wave][ind] / max_val * normalize + x_axis_val
      t_ind = self.t[ind] - offset[i_wave]
      if ((i_wave == 0) and (label is not None)):
        if is_time_axis_x:
          ax.plot(t_ind, wave_norm, *args, label=label, **kwargs)
        else:
          ax.plot(wave_norm, t_ind, *args, label=label, **kwargs)
      else:
        if is_time_axis_x:
          ax.plot(t_ind, wave_norm, *args, **kwargs)
        else:
          ax.plot(wave_norm, t_ind, *args, **kwargs)
      if fill:
        #ax = plt.gca()
        zero = np.zeros_like(wave_norm) + x_axis_val
        if is_time_axis_x:
          if fill_color[0] is not None: ax.fill_between(t_ind, wave_norm, zero, where=wave_norm >= zero , facecolor=fill_color[0])
          if fill_color[1] is not None: ax.fill_between(t_ind, wave_norm, zero, where=wave_norm <= zero , facecolor=fill_color[1])
        else:
          if fill_color[0] is not None: ax.fill_betweenx(t_ind, wave_norm, zero, where=wave_norm >= zero , facecolor=fill_color[0])
          if fill_color[1] is not None: ax.fill_betweenx(t_ind, wave_norm, zero, where=wave_norm <= zero , facecolor=fill_color[1])

  def plot_waveforms_ax_waterfall(self, *args, ax=None, time_range=None,
                                 x_axis_trans=(lambda coord:coord[0]), 
                                 x_range=None,
                                 saturate_with=None, saturation=1.0,
                                 norm=None, vmin=None, vmax=None,
                                 label=None, **kwargs):
    if ax is None: ax = plt.gca()
    if time_range is None: time_range = (self.t[0], self.t[-1])
    if saturate_with is None: saturate_with = self

    waveforms_arr = self.waveforms
    waveforms_saturate_with_arr = saturate_with.waveforms
    x_arr = np.array(list(map(x_axis_trans, self.coord_list)))
    if x_range is None: x_range=(x_arr.min(), x_arr.max())

    ind_t = np.logical_and(self.t >= time_range[0], self.t <= time_range[1])
    ind_x = np.logical_and(x_arr >= x_range[0], x_arr <= x_range[1])
    if norm is None:
      norm = abs(waveforms_saturate_with_arr[ind_x,:][:,ind_t]).max()
      norm = norm * saturation
    if vmin is None: vmin = -norm
    if vmax is None: vmax = norm
    return  ax.pcolormesh(x_arr[ind_x], self.t[ind_t],
                  waveforms_arr[ind_x,:][:,ind_t].T, *args,
                  vmin=vmin, vmax=vmax, **kwargs)

