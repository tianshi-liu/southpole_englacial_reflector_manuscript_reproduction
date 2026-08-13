import pickle as pkl
import numpy as np
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import signal
from section_utils import WaveformSection, get_taper
with open("../south_pole_peg/stack_simple_medfilt15_1066.pkl", 'rb') as f:
#with open("../south_pole_peg/stack_simple_1066.pkl", 'rb') as f:
#with open("../south_pole_peg/stack_simple_trace_norm_960.pkl", 'rb') as f:
  stack = pkl.load(f)

dt = 0.001
#dx = 1.021
dx = 0.998
nt, nx = stack.shape
t_arr = np.arange(nt) * dt - 1.0
x_arr = (np.arange(nx) - 950) * dx

statics = 0.059466010538879205
t_arr -= statics

sec0 = WaveformSection()
sec0.from_numpy(stack.T, t_arr, np.reshape(x_arr, newshape=(-1,1)))
sec0.cut_in_x(1000.0, 7000.0)
sec0.cut_in_time(0.5, 6)
sec0.trace_normalize(normalize_by='rms')
sec0.taper((0.5, 0.7, 5.8, 6.0),axis='t')
sec0.taper((1000.0, 1200.0, 6800.0, 7000.0),axis='x')
sec0.filter((30.0, 80.0))
sec0.gaussian_filter((5,5))
sec0.filter_fk(v_range=(1500.0, 2000.0, None, None))
p = sec0.plot_velocity_analysis(ax=plt.gca(),
                                v_arr=np.linspace(1500.0, 4500.0, 200),
                                h_arr=np.linspace(1000.0, 3200.0, 200),
                                vmax=3e-2, vmin=0.0, cmap='Spectral_r', rasterized=True)

plt.show()
