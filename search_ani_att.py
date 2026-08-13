import pickle as pkl
import numpy as np
import scipy
import matplotlib.pyplot as plt
from section_utils import WaveformSection
from compute_ref_coeff import compute_ss_coeff_ice_bed, compute_ss_coeff_ani
from scipy.interpolate import interp1d
import matplotlib.cm as cm
from tt import tt_sbs, tt_ses
#with open('spec.pkl', 'rb') as f:
#  sec0_spec = pkl.load(f)
#  sec1_spec = pkl.load(f)

plt.rcParams.update({'font.size': 20, 'font.family': 'serif', 'font.serif': ['Nimbus Roman']})

with open("../south_pole_peg/stack_simple_medfilt15_1066.pkl", 'rb') as f:
  stack = pkl.load(f)

dt = 0.001
dx = 0.998
nt, nx = stack.shape
statics = 0.059466010538879205
t_arr = np.arange(nt) * dt - 1.0 - statics
x_arr = (np.arange(nx) - 950) * dx

sec0 = WaveformSection()
sec0.from_numpy(stack.T, t_arr, np.reshape(x_arr, newshape=(-1,1)))

#sec0 = sec0.slice(1950, None, None)

sec0.cut_in_x(1000.0, 7500.0)
sec0.cut_in_time(0.5, 6)
sec0.taper((0.5, 0.7, 5.5, 6.0),axis='t')
sec0.taper((1000.0, 1200.0, 6500.0, 7000.0),axis='x')

sec0.filter((30.0, 80.0))

#H_sbs = 2950.0
#H_ses = 2400.0

#sec_sbs = sec0.reduce_time(reduce_by=(lambda x:tt_sbs(x, H=H_sbs)), t_range=(-0.5, 0.5))
#sec_ses = sec0.reduce_time(reduce_by=(lambda x:tt_ses(x, H=H_ses)), t_range=(-0.5, 0.5))
H_ses = 2295.0
H_sbs = 2855.0
v_ses = 1915.0
v_sbs = 1936.0
sec_ses = sec0.reduce_time(reduce_by=(lambda x:tt_ses(x, H=H_ses, vs_new=v_ses)), t_range=(-0.5, 0.5))
sec_sbs = sec0.reduce_time(reduce_by=(lambda x:tt_sbs(x, H=H_sbs, vs_new=v_sbs)), t_range=(-0.5, 0.5))

sec0 = sec_sbs
sec1 = sec_ses
sec0.gaussian_filter((7,5))
sec1.gaussian_filter((7,5))

sec0.filter_fk(v_range=(15000.0, 20000.0, None, None), use_abs=True)
sec1.filter_fk(v_range=(15000.0, 20000.0, None, None), use_abs=True)

sec0.taper((-0.15, -0.1, 0.1, 0.15))
sec1.taper((-0.15, -0.1, 0.1, 0.15))
sec0_spec = sec0.get_spec(freq_range=(30.0, 80.0))
sec0_spec.cut_in_x(3200.0, 6500.0)
sec1_spec = sec1.get_spec(freq_range=(30.0, 80.0))
sec1_spec.cut_in_x(3200.0, 6500.0)

freq = sec0_spec.t
x_arr = sec0_spec.coord_list[:,0]
spec_sbs = sec0_spec.waveforms
spec_ses = sec1_spec.waveforms

spec_sbs_avg = np.mean(spec_sbs, axis=0)

X, F = np.meshgrid(x_arr, freq, indexing='ij')

dT = tt_sbs(X, H=H_sbs, vs_new=v_sbs) - tt_ses(X, H=H_ses, vs_new=v_ses)

noa = 88+1
oa_arr = np.linspace(1.0, 89.0, noa)

misfit_arr = np.zeros(shape=(noa,noa), dtype=float)

best_q_arr = np.zeros(shape=(noa,noa), dtype=float)


q_arr = np.arange(10.0, 320.0, 0.2)
dist_arr = np.linspace(3000.0, 7000.0, 200)
inc_angle_ses = np.arctan2(dist_arr / 2.0, H_ses)
inc_angle_sbs = np.arctan2(dist_arr / 2.0, H_sbs)
oa1_best_arr = []
oa2_best_arr = []
f = open("optimal_val", 'w')
for i1 in range(noa):
  misfit_best = 1.0e20
  i2_best = None
  for i2 in range(noa):
    oa1 = oa_arr[i1]
    oa2 = oa_arr[i2]
    if (oa1 <= oa2):
      misfit_arr[i1, i2] = np.nan
      best_q_arr[i1, i2] = np.nan
      continue
    r_ses = [compute_ss_coeff_ani(inc_angle, oa1, oa2) for inc_angle in inc_angle_ses]
    r_ses_interp = interp1d(dist_arr, r_ses)
    r_ses_arr = r_ses_interp(X)
    r_sbs = [compute_ss_coeff_ice_bed(inc_angle) for inc_angle in inc_angle_sbs]
    r_sbs_interp = interp1d(dist_arr, r_sbs)
    r_sbs_arr = r_sbs_interp(X)
    r_sbs_ses_arr = r_sbs_arr / r_ses_arr

    misfit_min = 1.0e20
    best_q = 0.0
    for qval in q_arr:
      spec_sbs_avg_pred = np.mean(spec_ses * r_sbs_ses_arr * np.exp(-np.pi * F * dT / qval), axis=0)
      misfit = np.sqrt(np.mean((spec_sbs_avg_pred - spec_sbs_avg)**2)) / np.sqrt(np.mean(spec_sbs_avg**2))
      if misfit < misfit_min:
        misfit_min = misfit
        best_q = qval
    misfit_arr[i1, i2] = misfit_min
    best_q_arr[i1, i2] = best_q
    if (misfit_min <= misfit_best):
      i2_best = i2
      misfit_best = misfit_min
    print(f"oa1={oa1}, oa2={oa2}, min misfit={misfit_min}, best Q={best_q}")
  if (i2_best is not None): 
    f.write(f"{oa_arr[i1]} {oa_arr[i2_best]} {best_q_arr[i1, i2_best]} {misfit_arr[i1, i2_best]}\n")
    if (oa_arr[i2_best] > 1.0001):
      oa1_best_arr.append(oa_arr[i1])
      oa2_best_arr.append(oa_arr[i2_best])
f.close()

# Horgan result
xp = [32.0, 33.0, 34.8, 37.5, 42.0, 50.0, 60.0, 70.0, 80.0, 90.0]
yp = [0.0,  5.0,  10.0,  15.0,  20.0, 26.0, 30.0, 32.0, 30.0, 28.0]
interp1 = scipy.interpolate.make_interp_spline(xp, yp)
x1 = np.linspace(32.0, 90.0, 20)
y1 = interp1(x1)

xp = [55.0, 60.0, 67.5, 80.0, 90.0]
yp = [5.0, 10.0, 12.5, 10.0, 6.0]
interp2 = scipy.interpolate.make_interp_spline(xp, yp)
x2 = np.linspace(50.0, 90.0, 20)
y2 = interp2(x2)

cmap = cm.Spectral_r
cmap.set_bad(color='gray')

#plt.rcParams['text.usetex'] = True

fig, axes = plt.subplots(1,2, figsize=(14,5), constrained_layout=True)
ax = axes[0]
p=ax.pcolormesh(oa_arr, oa_arr, misfit_arr.T, cmap=cmap, vmax=0.3, rasterized=True)    
ax.set_xlabel(r"$\theta_a$ (deg)")
ax.set_ylabel(r"$\theta_b$ (deg)")
cbar=fig.colorbar(p, ax=ax, label=r"$\mathrm{min}_Q \Phi(\theta_a, \theta_b, Q)$")
#cbar.formatter.set_powerlimits((0,0))
ax.text(0.02, 0.98, '(a)', horizontalalignment='left',
                         verticalalignment='top',
                         transform=ax.transAxes,
                         backgroundcolor='white')
ax.plot(oa1_best_arr, oa2_best_arr, 'k--')
ax.plot(x1, y1, 'k-.')
ax.plot(x2, y2, 'k-.')
ax.set_xlim([0, 90])
ax.set_ylim([0, 90])

ax = axes[1]
p=ax.pcolormesh(oa_arr, oa_arr, best_q_arr.T, cmap=cmap, vmin=10.0, vmax=200.0, rasterized=True)
ax.set_xlabel(r"$\theta_a$ (deg)")
ax.set_ylabel(r"$\theta_b$ (deg)")
cbar=fig.colorbar(p, ax=ax, label=r"$\mathrm{argmin}_Q \Phi(\theta_a, \theta_b, Q)$")
ax.text(0.02, 0.98, '(b)', horizontalalignment='left',
                         verticalalignment='top',
                         transform=ax.transAxes,
                         backgroundcolor='white')
ax.plot(oa1_best_arr, oa2_best_arr, 'k--')
ax.plot(x1, y1, 'k-.')
ax.plot(x2, y2, 'k-.')
ax.set_xlim([0, 90])
ax.set_ylim([0, 90])
plt.savefig("ani2.pdf", dpi=400, bbox_inches='tight')
