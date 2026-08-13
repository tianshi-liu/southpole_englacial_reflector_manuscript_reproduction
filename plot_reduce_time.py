import pickle as pkl
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import signal
from section_utils import WaveformSection, get_taper
from tt import tt_ses, tt_sbs

#with open("../south_pole_peg/stack_simple_trace_norm_1066.pkl", 'rb') as f:
with open("../south_pole_peg/stack_simple_medfilt15_1066.pkl", 'rb') as f:
#with open("../south_pole_peg/stack_simple_1066.pkl", 'rb') as f:
#with open("../south_pole_peg/stack_simple_trace_norm_960.pkl", 'rb') as f:
  stack = pkl.load(f)

dt = 0.001
#dx = 1.021
dx = 0.998
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


fig, axes = plt.subplots(1,2, figsize=(12,3), constrained_layout=True)

sec0 = WaveformSection()
sec0.from_numpy(stack.T, t_arr, np.reshape(x_arr, newshape=(-1,1)))
sec0.cut_in_x(3000.0, 7000.0)
sec0.cut_in_time(0.5, 6)
sec0.taper((0.5, 0.7, 5.5, 6.0),axis='t')
sec0.taper((3000.0, 3200.0, 6500.0, 7000.0),axis='x')
sec0.filter((30.0, 80.0))
sec0.cut_in_x(3200.0, 6500.0)

sec_ses = sec0.reduce_time(reduce_by=(lambda x:tt_ses(x, H=2295.0, vs_new=1915.0)), t_range=(-0.5, 0.5))
sec_sbs = sec0.reduce_time(reduce_by=(lambda x:tt_sbs(x, H=2855.0, vs_new=1936.0)), t_range=(-0.5, 0.5))

sec_ses.gaussian_filter((7,5))
sec_sbs.gaussian_filter((7,5))

sec_ses.filter_fk(v_range=(15000.0, 20000.0, None, None), use_abs=True)
sec_sbs.filter_fk(v_range=(15000.0, 20000.0, None, None), use_abs=True)

ax = axes[0]
p = sec_ses.plot_waveforms_ax_waterfall(ax=ax,vmin=-0.99e-10, vmax=0.99e-10, cmap='bwr', rasterized=True)
ax.set_xlim([3200.0, 6500.0])
ax.set_ylim([-0.5, 0.5])
ax.invert_yaxis()
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Time (s)")
#fig.colorbar(p, ax=ax, label='Strain rate (/s)')
ax.text(0.02, 0.98, '(a)', horizontalalignment='left',
                         verticalalignment='top',
                         transform=ax.transAxes,
                         backgroundcolor='white',
                         fontsize=12)

ax = axes[1]
p = sec_sbs.plot_waveforms_ax_waterfall(ax=ax,vmin=-0.99e-10, vmax=0.99e-10, cmap='bwr', rasterized=True)
ax.set_xlim([3200.0, 6500.0])
ax.set_ylim([-0.5, 0.5])
ax.invert_yaxis()
ax.set_xlabel("Distance (m)")
ax.yaxis.set_ticklabels([])
#ax.set_ylabel("Time (s)")
ax.text(0.02, 0.98, '(b)', horizontalalignment='left',
                         verticalalignment='top',
                         transform=ax.transAxes,
                         backgroundcolor='white',
                         fontsize=12)

fig.colorbar(p, ax=axes, orientation='vertical', fraction=0.5, ticks=[-8e-11, -4e-11, 0, 4e-11, 8e-11], label='Strain rate (/s)')
plt.savefig("reduce_time2.pdf", dpi=400, bbox_inches='tight')
