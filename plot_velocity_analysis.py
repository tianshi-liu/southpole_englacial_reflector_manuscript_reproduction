import pickle as pkl
import numpy as np
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import signal
from section_utils import WaveformSection, get_taper
from tt import tt_pbs, tt_ses, tt_sbs

plt.rcParams.update({'font.size': 20, 'font.family': 'serif', 'font.serif': ['Nimbus Roman']})
#with open("../south_pole_peg/stack_simple_trace_norm_1066.pkl", 'rb') as f:
with open("../south_pole_peg/stack_simple_medfilt15_1066.pkl", 'rb') as f:
#with open("../south_pole_peg/stack_simple_1066.pkl", 'rb') as f:
#with open("../south_pole_peg/stack_simple_trace_norm_960.pkl", 'rb') as f:
  stack = pkl.load(f)

def plot_box(ax, x_arr, y_arr, *args, width=0.1, **kwargs):
  ax.plot(np.concatenate([x_arr, x_arr[::-1], x_arr[0:1]]), np.concatenate([y_arr+width, y_arr[::-1]-width, y_arr[0:1]+width]), *args, **kwargs)

dt = 0.001
#dx = 1.021
dx = 0.998
nt, nx = stack.shape
t_arr = np.arange(nt) * dt - 1.0
x_arr = (np.arange(nx) - 950) * dx
statics = 0.059466010538879205
#x_arr = (9196 - np.arange(nx) + 950) * dx
#x_arr = x_arr[::-1]
#stack = stack[:,::-1]

#x_arr = -np.arange(nx) * dx
#stack = stack[:, ::-1]
#x_arr = x_arr[::-1]

fig, axes = plt.subplots(2,2, figsize=(15,10), constrained_layout=True)

ax = axes[0,0]
sec0 = WaveformSection()
sec0.from_numpy(stack.T, t_arr, np.reshape(x_arr, newshape=(-1,1)))
sec0.cut_in_x(1000.0, 7000.0)
sec0.cut_in_time(0.5, 6)
sec0.taper((0.5, 0.7, 5.8, 6.0),axis='t')
sec0.taper((1000.0, 1200.0, 6800.0, 7000.0),axis='x')
sec0.filter((30.0, 80.0))
sec0 = sec0.slice(None, None, 5)
p = sec0.plot_waveforms_ax_waterfall(ax=ax,vmin=-2e-9, vmax=2e-9, cmap='bwr', rasterized=True)
ax.set_xlim([1000.0, 7000.0])
ax.set_ylim([0.5, 6.0])
ax.invert_yaxis()
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Time (s)")
ax.set_yticks(ticks=(1,2,3,4,5,6))
cbar=fig.colorbar(p, ax=ax, label='Strain rate (/s)', ticks=[-2e-9, -1e-9, 0, 1e-9, 2e-9])
cbar.formatter.set_powerlimits((0,0))
cbar.formatter.set_useMathText(True)
ax.text(0.02, 0.98, '(a)', horizontalalignment='left', 
                         verticalalignment='top',
                         transform=ax.transAxes,
                         backgroundcolor='white')

#ax = axes[0,1]
sec0 = WaveformSection()
sec0.from_numpy(stack.T, t_arr, np.reshape(x_arr, newshape=(-1,1)))
sec0.cut_in_x(1000.0, 7000.0)
sec0.cut_in_time(0.5, 6)
sec0.taper((0.5, 0.7, 5.8, 6.0),axis='t')
sec0.taper((1000.0, 1200.0, 6800.0, 7000.0),axis='x')
sec0.gaussian_filter((7,7))
sec0.filter((30.0, 80.0))
sec0.filter_fk(v_range=(2000.0, 2500.0, None, None))
sec0 = sec0.slice(None, None, 5)
'''
p = sec0.plot_waveforms_ax_waterfall(ax=ax,vmin=-0.5e-10, vmax=0.5e-10, cmap='bwr', rasterized=True)

xx = np.linspace(2500.0, 7000.0, 100)
tt = tt_pbs(xx,H=2900.0)
ax.plot(xx, tt, 'k--', alpha=0.7)

xx = np.linspace(2500.0, 7000.0, 100)
tt = tt_ses(xx,H=2400.0)
ax.plot(xx, tt, 'g--', alpha=0.7)

xx = np.linspace(2500.0, 7000.0, 100)
tt = tt_sbs(xx,H=2900.0)
ax.plot(xx, tt, 'b--', alpha=0.7)

ax.set_xlim([1000.0, 7000.0])
ax.set_ylim([0.5, 6.0])
ax.invert_yaxis()
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Time (s)")
ax.set_yticks(ticks=(1,2,3,4,5,6))
cbar=fig.colorbar(p, ax=ax, label='Strain rate (/s)')
cbar.formatter.set_powerlimits((0,0))
cbar.formatter.set_useMathText(True)
ax.text(0.02, 0.98, '(b)', horizontalalignment='left',        
                         verticalalignment='top',                     
                         transform=ax.transAxes,
                         backgroundcolor='white',
                         fontsize=12)

ax.text(5000, 2.9, 'PbS', horizontalalignment='left',
                          verticalalignment='bottom',
                          fontsize=12)
ax.text(5000, 3.5, 'SeS', horizontalalignment='left',
                          verticalalignment='bottom',
                          fontsize=12)
ax.text(5000, 4.0, 'SbS', horizontalalignment='right',
                          verticalalignment='top',
                          fontsize=12)
'''
ax = axes[0,1]
p = sec0.plot_waveforms_ax_waterfall(ax=ax,vmin=-0.5e-10, vmax=0.5e-10, cmap='bwr', rasterized=True)

xx = np.linspace(2600.0, 5400.0, 100)
tt = tt_pbs(xx,H=2855.0)
#ax.plot(xx, tt+statics, 'k--', alpha=0.7)
plot_box(ax, xx, tt+statics, 'k--', width=0.1, alpha=0.7)

xx = np.linspace(2600.0, 5400.0, 100)
tt = tt_ses(xx,H=2295.0, vs_new=1915.0)
#ax.plot(xx, tt+statics, 'g--', alpha=0.7)
plot_box(ax, xx, tt+statics, 'g--', width=0.1, alpha=0.7)

xx = np.linspace(2600.0, 5400.0, 100)
tt = tt_sbs(xx,H=2855.0, vs_new=1936.0)
#ax.plot(xx, tt+statics, 'b--', alpha=0.7)
plot_box(ax, xx, tt+statics, 'b--', width=0.1, alpha=0.7)

ax.set_xlim([2500.0, 5500.0])
ax.set_ylim([2.0, 5.0])
ax.invert_yaxis()
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Time (s)")
ax.set_yticks(ticks=(2,3,4,5))
cbar=fig.colorbar(p, ax=ax, label='Strain rate (/s)')
cbar.formatter.set_powerlimits((0,0))
cbar.formatter.set_useMathText(True)
ax.text(0.02, 0.98, '(b)', horizontalalignment='left',        
                         verticalalignment='top',                     
                         transform=ax.transAxes,
                         backgroundcolor='white')

ax.text(5000, 2.9, 'PbS', horizontalalignment='left',
                          verticalalignment='bottom')
ax.text(5000, 3.5, 'SeS', horizontalalignment='left',
                          verticalalignment='bottom')
ax.text(5000, 4.0, 'SbS', horizontalalignment='right',
                          verticalalignment='top')

ax = axes[1,0]
sec0 = WaveformSection()
t_arr -= statics
sec0.from_numpy(stack.T, t_arr, np.reshape(x_arr, newshape=(-1,1)))
sec0.cut_in_x(1000.0, 7000.0)
sec0.cut_in_time(0.5, 6)
sec0.trace_normalize(normalize_by='rms')
sec0.taper((0.5, 0.7, 5.8, 6.0),axis='t')
sec0.taper((1000.0, 1200.0, 6800.0, 7000.0),axis='x')
sec0.filter((30.0, 80.0))
sec0.gaussian_filter((5,5))
sec0.filter_fk(v_range=(1500.0, 2000.0, None, None))
p = sec0.plot_velocity_analysis(ax=ax, 
                                v_arr=np.linspace(1500.0, 4500.0, 200),
                                h_arr=np.linspace(1000.0, 3200.0, 200),
                                vmax=3e-2, vmin=0.0, cmap='Spectral_r', rasterized=True)
#axes[1,1].set_xlim([1000.0, 7000.0])
#axes[1,0].set_ylim([2.0, 5.0])
ax.invert_yaxis()
ax.set_xlabel("Velocity (m/s)")
ax.set_ylabel("Depth (m)")
cbar=fig.colorbar(p, ax=ax, label='Intensity', ticks=[0.01, 0.02, 0.03])
cbar.formatter.set_powerlimits((0,0))
cbar.formatter.set_useMathText(True)
ax.text(0.02, 0.98, '(c)', horizontalalignment='left',
                         verticalalignment='top',
                         transform=ax.transAxes,
                         backgroundcolor='white')

ax.text(2700, 3000, 'PbS', horizontalalignment='left',
                          verticalalignment='bottom')
ax.text(1950, 2800, 'SbS', horizontalalignment='left',
                          verticalalignment='bottom')
ax.text(1950, 2300, 'SeS', horizontalalignment='left',
                          verticalalignment='bottom')

ax.plot([1500, 3000, 3000, 1500, 1500], [2000, 2000, 3200, 3200, 2000], 'orange', linestyle='--')

ax = axes[1,1]
sec0 = WaveformSection()
t_arr -= statics
sec0.from_numpy(stack.T, t_arr, np.reshape(x_arr, newshape=(-1,1)))
sec0.cut_in_x(1000.0, 7000.0)
sec0.cut_in_time(0.5, 6)
sec0.trace_normalize(normalize_by='rms')
sec0.taper((0.5, 0.7, 5.8, 6.0),axis='t')
sec0.taper((1000.0, 1200.0, 6800.0, 7000.0),axis='x')
sec0.filter((30.0, 80.0))
sec0.gaussian_filter((5,5))
sec0.filter_fk(v_range=(1500.0, 2000.0, None, None))
p = sec0.plot_velocity_analysis(ax=ax,
                                v_arr=np.linspace(1500.0, 3000.0, 200),
                                h_arr=np.linspace(2000.0, 3200.0, 200),
                                vmax=3e-2, vmin=0.0, cmap='Spectral_r', rasterized=True)
#axes[1,1].set_xlim([1000.0, 7000.0])
#axes[1,0].set_ylim([2.0, 5.0])
ax.invert_yaxis()
ax.set_xlabel("Velocity (m/s)")
ax.set_ylabel("Depth (m)")
cbar=fig.colorbar(p, ax=ax, label='Intensity', ticks=[0.01, 0.02, 0.03])
cbar.formatter.set_powerlimits((0,0))
cbar.formatter.set_useMathText(True)
ax.text(0.02, 0.98, '(d)', horizontalalignment='left',
                         verticalalignment='top',
                         transform=ax.transAxes,
                         backgroundcolor='white')

ax.text(2700, 3000, 'PbS', horizontalalignment='left',
                          verticalalignment='bottom')
ax.text(1950, 2800, 'SbS', horizontalalignment='left',
                          verticalalignment='bottom')
ax.text(1950, 2300, 'SeS', horizontalalignment='left',
                          verticalalignment='bottom')

#plt.show()
plt.savefig("section2.pdf", dpi=400, bbox_inches='tight')
