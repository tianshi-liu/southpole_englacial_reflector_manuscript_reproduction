import numpy as np
from ani_utils import get_wave_ani, get_wave_snell, get_traction_ani
from ice_aniso import get_rho, get_c
import matplotlib.pyplot as plt
import pickle as pkl

rho = get_rho()
#dist_arr = (np.arange(32) + 1) * 0.25
#angle_arr = np.arctan2(dist_arr / 2.0, 3.0) / np.pi * 180.0
angle_arr = np.linspace(5.0, 85.0, 40)

#oa_list = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
#oa_list = [(30.0, 18.75), (40.0, 30.0), (50.0, 37.5), (60.0, 41.25), (70.0, 42.5), (80.0, 40.0), (90.0, 38.75)]
oa_list = [(20.0, 12.0), (30.0, 24.0), (40.0, 33.0), (50.0, 42.0), (60.0, 47.0), (70.0, 49.0), (80.0, 47.0), (89.0, 44.0)]
f = open('rt_all.pkl', 'wb')
#for oa1 in oa_list:
#  for oa2 in oa_list:
for oa1, oa2 in oa_list:
    c1 = get_c(fabric='cone', open_angle=oa1)
    c2 = get_c(fabric='cone', open_angle=oa2)
    rt_all = []
    kz_all = []
    for inc in [0, 1]:
      ref_coeff_qp = np.zeros_like(angle_arr, dtype=complex)
      ref_coeff_qs = np.zeros_like(angle_arr, dtype=complex)
      kz_qp = np.zeros_like(angle_arr, dtype=complex)
      kz_qs = np.zeros_like(angle_arr, dtype=complex)
      for i, angle in enumerate(angle_arr):
        inc_angle = angle * np.pi / 180.0
        n_inc = np.array([np.sin(inc_angle), 0.0, -np.cos(inc_angle)])
        v, G = get_wave_ani(c1/rho, n_inc)
        v_inc = v[inc]
        g_inc = G[:,inc]
        k_inc = 1.0 / v_inc * n_inc
        g_inc = np.reshape(g_inc, newshape=(3,1))
        k_inc = np.reshape(k_inc, newshape=(3,1))
        #print(k_inc)
        #print(g_inc)
        q_inc = get_traction_ani(c1, g_inc, k_inc)
      
        p = k_inc[0,0]
      
        k_ref, G_ref = get_wave_snell(p, c1/rho, reflection=True)
        Q_ref = get_traction_ani(c1, G_ref, k_ref)
      
        #ref_coeff_qp[i] = abs(G_ref[2,0])
        #ref_coeff_qs[i] = abs(G_ref[2,1])
      
        #print(k_ref)
        #print(G_ref)
      
        k_tran, G_tran = get_wave_snell(p, c2/rho, reflection=False)
        Q_tran = get_traction_ani(c2, G_tran, k_tran)
      
        #print(k_tran)
        #print(G_tran)
      
        y = np.zeros(shape=(6,1), dtype=complex)
        A = np.zeros(shape=(6,6), dtype=complex)
      
        y[0:3, 0:1] = g_inc
        y[3:6, 0:1] = q_inc
      
        A[0:3, 0:3] = -G_ref
        A[3:6, 0:3] = -Q_ref
      
        A[0:3, 3:6] = G_tran
        A[3:6, 3:6] = Q_tran

        kz_qp[i] = k_ref[2,0]
        kz_qs[i] = k_ref[2,1]
        try:
          coef = np.linalg.solve(A, y)
          ref_coeff_qp[i] = coef[0,0]
          ref_coeff_qs[i] = coef[1,0]
        except Exception as e:
          print((oa1, oa2, angle))
          ref_coeff_qp[i] = np.nan
          ref_coeff_qs[i] = np.nan
      kz_all.append(kz_qp)
      kz_all.append(kz_qs)
      rt_all.append(ref_coeff_qp)
      rt_all.append(ref_coeff_qs)
        #print(coef)
    pkl.dump(kz_all, f)
    pkl.dump(rt_all, f)
    print((oa1, oa2))

f.close()
#plt.plot(angle_arr, ref_coeff_qp, 'b', label='qP')
#plt.plot(angle_arr, ref_coeff_qs, 'r', label='qSV')
#plt.legend()
#plt.show()
