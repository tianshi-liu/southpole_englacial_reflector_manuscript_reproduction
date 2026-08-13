import numpy as np
from rtcoef import RTCOEF
from ani_utils import get_wave_ani, get_wave_snell, get_traction_ani
from ice_aniso import get_rho, get_c

def compute_ss_coeff_ice_bed(incident_angle_rad, rho1=917.0, vp1=3871.8, vs1=1966.2, rho2=1800.0, vp2=1700.0, vs2=200.0):
  hslow = np.sin(incident_angle_rad) / vs1
  return np.absolute(RTCOEF(vp1, vs1, rho1, vp2, vs2, rho2, hslow)[5])

def compute_ss_coeff_ani(incident_angle_rad, oa1, oa2):
  rho = get_rho()
  c1 = get_c(fabric='cone', open_angle=oa1)
  c2 = get_c(fabric='cone', open_angle=oa2)
  n_inc = np.array([np.sin(incident_angle_rad), 0.0, -np.cos(incident_angle_rad)])
  v, G = get_wave_ani(c1/rho, n_inc)
  v_inc = v[1]
  g_inc = G[:,1]
  k_inc = 1.0 / v_inc * n_inc
  g_inc = np.reshape(g_inc, newshape=(3,1))
  k_inc = np.reshape(k_inc, newshape=(3,1))

  q_inc = get_traction_ani(c1, g_inc, k_inc)

  p = k_inc[0,0]
      
  k_ref, G_ref = get_wave_snell(p, c1/rho, reflection=True)
  Q_ref = get_traction_ani(c1, G_ref, k_ref)

  k_tran, G_tran = get_wave_snell(p, c2/rho, reflection=False)
  Q_tran = get_traction_ani(c2, G_tran, k_tran)

  y = np.zeros(shape=(6,1), dtype=complex)
  A = np.zeros(shape=(6,6), dtype=complex)

  y[0:3, 0:1] = g_inc
  y[3:6, 0:1] = q_inc

  A[0:3, 0:3] = -G_ref
  A[3:6, 0:3] = -Q_ref

  A[0:3, 3:6] = G_tran
  A[3:6, 3:6] = Q_tran

  coef = np.linalg.solve(A, y)

  return np.absolute(coef[1,0])
