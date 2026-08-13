import pickle as pkl
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import signal
from section_utils import WaveformSection, get_taper
from tt import tt_ses, tt_sbs

plt.rcParams.update({'font.size': 20, 'font.family': 'serif', 'font.serif': ['Nimbus Roman']})

#with open("../south_pole_peg/stack_simple_trace_norm_1066.pkl", 'rb') as f:
with open("../south_pole_peg/stack_simple_medfilt15_1066.pkl", 'rb') as f:
#with open("../south_pole_peg/stack_simple_1066.pkl", 'rb') as f:
#with open("../south_pole_peg/stack_simple_trace_norm_960.pkl", 'rb') as f:
  stack = pkl.load(f)

dt = 0.001
#dx = 1.021
dx = 0.998

H_ses = 2295.0
H_sbs = 2855.0
v_ses = 1915.0
v_sbs = 1936.0

qval = 30.0

nt, nx = stack.shape
statics = 0.059466010538879205
t_arr = np.arange(nt) * dt - 1.0 - statics
x_arr = (np.arange(nx) - 950) * dx
#x_arr = (9196 - np.arange(nx) + 950) * dx
#x_arr = x_arr[::-1]
#stack = stack[:,::-1]

#x_arr = -np.arange(nx) * dx
#stack = stack[:, ::-1]
#x_arr = x_arr[::-1]

fig = plt.figure(figsize=(12,5))
#fig, axes = plt.subplots(2,3, figsize=(12,5), width_ratios=(2,5,5), constrained_layout=True)

sec0 = WaveformSection()
sec0.from_numpy(stack.T, t_arr, np.reshape(x_arr, newshape=(-1,1)))
sec0.cut_in_x(3000.0, 7000.0)
sec0.cut_in_time(0.5, 6)
sec0.taper((0.5, 0.7, 5.5, 6.0),axis='t')
sec0.taper((3000.0, 3200.0, 6500.0, 7000.0),axis='x')
sec0.filter((30.0, 80.0))
sec0.cut_in_x(3200.0, 6500.0)

sec_ses = sec0.reduce_time(reduce_by=(lambda x:tt_ses(x, H=H_ses, vs_new=v_ses)), t_range=(-0.5, 0.5))
sec_sbs = sec0.reduce_time(reduce_by=(lambda x:tt_sbs(x, H=H_sbs, vs_new=v_sbs)), t_range=(-0.5, 0.5))

sec_ses.gaussian_filter((7,5))
sec_sbs.gaussian_filter((7,5))

sec_ses.filter_fk(v_range=(15000.0, 20000.0, None, None), use_abs=True)
sec_sbs.filter_fk(v_range=(15000.0, 20000.0, None, None), use_abs=True)

sec_ses.taper((-0.15, -0.1, 0.1, 0.15))
sec_sbs.taper((-0.15, -0.1, 0.1, 0.15))

sec_ses_spec = sec_ses.get_spec(freq_range=(30.0, 80.0))
sec_sbs_spec = sec_sbs.get_spec(freq_range=(30.0, 80.0))

freq = sec_ses_spec.t

X, F = np.meshgrid(sec_ses.coord_list[:,0], freq, indexing='ij')

dT = tt_sbs(X, H=H_sbs, vs_new=v_sbs) - tt_ses(X, H=H_ses, vs_new=v_ses)

spec_ses_correct = sec_ses_spec.waveforms * np.exp(-np.pi * F * dT / qval)

plt.plot(sec_ses_spec.t, np.mean(sec_ses_spec.waveforms, axis=0), 'r', linewidth=2, label='SeS')
plt.plot(sec_sbs_spec.t, np.mean(sec_sbs_spec.waveforms, axis=0), 'b', linewidth=2, label='SbS')
plt.plot(sec_ses_spec.t, np.mean(spec_ses_correct, axis=0), 'r--', linewidth=2, label='SeS att.')

plt.gca().set_xlabel("Frequency (Hz)")
plt.gca().set_ylabel("Amplitude spectrum")
plt.gca().set_yticks([1e-12, 2e-12, 3e-12])
plt.gca().legend()
plt.gca().set_ylim([0,3.5e-12])

plt.savefig("reduce_time_spec_curve_only.pdf", dpi=400, bbox_inches='tight')
