import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pkl
from rtcoef import RTCOEF

plt.rcParams.update({'font.size': 20, 'font.family': 'serif', 'font.serif': ['Nimbus Roman']})

#dist_arr = (np.arange(32) + 1) * 0.25
#angle_arr = np.arctan2(dist_arr / 2.0, 3.0)
angle_arr = np.linspace(5.0, 85.0, 40)

oa_list = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
oa_list = [(20.0, 12.0), (30.0, 24.0), (40.0, 33.0), (50.0, 42.0), (60.0, 47.0), (70.0, 49.0), (80.0, 47.0), (89.0, 44.0)]
vp = 3840.0
vs = 1970.0
rho = 917.0
vp_sed = 1700.0
vs_sed = 200.0
rho_sed = 1800.0
H = 2400.0
Hb = 2950.0
PbP = []
PbS = []
SbS = []
for angle in angle_arr:
  hslow = np.sin(angle / 180.0 * np.pi) / vp
  rtcoeff = RTCOEF(vp, vs, rho, vp_sed, vs_sed, rho_sed, hslow)
  PbP.append(abs(rtcoeff[0]))
  PbS.append(abs(rtcoeff[1]))
  hslow = np.sin(angle / 180.0 * np.pi) / vs
  rtcoeff = RTCOEF(vp, vs, rho, vp_sed, vs_sed, rho_sed, hslow)
  SbS.append(abs(rtcoeff[5]))
PbP = np.array(PbP)
PbS = np.array(PbS)
SbS = np.array(SbS)

max_dist = 8000.0
f = open('rt_all.pkl', 'rb')
fig, axs = plt.subplots(1,3,figsize=(15,4),layout='constrained')
for oa1, oa2 in oa_list:
    kz_pp, kz_ps, kz_sp, kz_ss = pkl.load(f)
    r_pp, r_ps, r_sp, r_ss = pkl.load(f)
    #if oa1 <= 45: continue
    #if oa1 >= 55: continue
    #if oa2 <= 35: continue
    #if oa2 >= 45: continue
    #if oa2 > oa1: continue
    color = plt.cm.jet(oa1/90.0)

    k = kz_pp
    r = r_pp
    #noev = abs(np.imag(k) / np.real(k)) < 1e-5
    #ev = np.logical_not(noev)
    p_arr = np.sin(angle_arr / 180.0 * np.pi) / vp
    #p_arr = p_arr[noev]
    ax = axs[0]
    dist_arr = H*p_arr*vp / np.sqrt(1.0-(p_arr*vp)**2) + H*p_arr*vp / np.sqrt(1.0-(p_arr*vp)**2)
    ind = (dist_arr <= max_dist)
    ax.plot(dist_arr[ind], abs(r[ind]), '--', color=color)
    dist_arr = Hb*p_arr*vp / np.sqrt(1.0-(p_arr*vp)**2) + Hb*p_arr*vp / np.sqrt(1.0-(p_arr*vp)**2)
    ind = (dist_arr <= max_dist)
    ax.plot(dist_arr[ind], PbP[ind] / 5, '-', color='k')
    ax.set_ylim((0.0, 0.1))
    #axs[0,0].set_xlim((0.0, max_dist))
    ax.text(0.95,0.95,'PP', horizontalalignment='right', 
                            verticalalignment='top', 
                            transform=ax.transAxes,
                            backgroundcolor='white')
    ax.text(0.02, 0.98, '(a)', horizontalalignment='left',
                         verticalalignment='top',
                         transform=ax.transAxes,
                         backgroundcolor='white')
    ax.set_ylabel("Reflectivity")
    ax.set_xlabel("Distance (m)")

    k = kz_ps
    r = r_ps
    #noev = abs(np.imag(k) / np.real(k)) < 1e-5
    #ev = np.logical_not(noev)
    p_arr = np.sin(angle_arr / 180.0 * np.pi) / vp
    #p_arr = p_arr[noev]
    ax = axs[1]
    dist_arr = H*p_arr*vp / np.sqrt(1.0-(p_arr*vp)**2) + H*p_arr*vs / np.sqrt(1.0-(p_arr*vs)**2)
    ind = (dist_arr <= max_dist)
    ax.plot(dist_arr[ind], abs(r[ind]), '--', color=color)
    dist_arr = Hb*p_arr*vp / np.sqrt(1.0-(p_arr*vp)**2) + Hb*p_arr*vs / np.sqrt(1.0-(p_arr*vs)**2)
    ind = (dist_arr <= max_dist)
    ax.plot(dist_arr[ind], PbS[ind] / 5, '-', color='k')
    #axs[0,1].plot(dist_arr[ev], abs(r[ev]), '--', color=color)
    ax.set_ylim((0.0, 0.1))
    #axs[0,1].set_xlim((0.0, max_dist))
    ax.text(0.95,0.95,'PS', horizontalalignment='right', 
                            verticalalignment='top', 
                            transform=ax.transAxes,
                            backgroundcolor='white')
    ax.text(0.02, 0.98, '(b)', horizontalalignment='left',
                         verticalalignment='top',
                         transform=ax.transAxes,
                         backgroundcolor='white')
    ax.set_xlabel("Distance (m)")

    k = kz_ss
    r = r_ss
    #noev = abs(np.imag(k) / np.real(k)) < 1e-5
    #ev = np.logical_not(noev)
    p_arr = np.sin(angle_arr / 180.0 * np.pi) / vs
    #p_arr = p_arr[noev]
    ax = axs[2]
    dist_arr = H*p_arr*vs / np.sqrt(1.0-(p_arr*vs)**2) + H*p_arr*vs / np.sqrt(1.0-(p_arr*vs)**2)
    ind = (dist_arr <= max_dist)
    ax.plot(dist_arr[ind], abs(r[ind]), '--', color=color)
    dist_arr = Hb*p_arr*vs / np.sqrt(1.0-(p_arr*vs)**2) + Hb*p_arr*vs / np.sqrt(1.0-(p_arr*vs)**2)
    ind = (dist_arr <= max_dist)
    ax.plot(dist_arr[ind], SbS[ind] / 5, '-', color='k')
    #axs[1,1].plot(dist_arr[ev], abs(r[ev]), '--', color=color)
    ax.set_ylim((0.0, 0.1))
    #axs[1,1].set_xlim((0.0, max_dist))
    ax.text(0.95,0.95,'SS', horizontalalignment='right', 
                            verticalalignment='top', 
                            transform=ax.transAxes,
                            backgroundcolor='white')
    ax.text(0.02, 0.98, '(c)', horizontalalignment='left',
                         verticalalignment='top',
                         transform=ax.transAxes,
                         backgroundcolor='white')
    ax.set_xlabel("Distance (m)")

fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 90.0), cmap='jet'), ax=axs, label='bottom opening angle (deg)', orientation='vertical', fraction=0.5)
plt.savefig("rcoef_all2.pdf")
f.close()
