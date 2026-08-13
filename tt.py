import numpy as np
vp = 3870.0
vs = 1940.0

def tt_pbs(x, H, output_reflection_point=False, vp_new=vp, vs_new=vs):
  #vp = 3870.0
  #vs = 1940.0
  #H = 3000.0
  k = vp_new / vs_new
  xi_arr = x / H
  r_arr = np.zeros_like(xi_arr)
  for i in range(len(xi_arr)):
    xi = xi_arr[i]
    if (xi >= 1e-8):
      coeffs = [k*k-1.0, -2.0*(k*k-1.0), (k*k-1.0)*(1.0+1.0/xi/xi), -2.0*k*k/xi/xi, k*k/xi/xi]
      roots = np.roots(coeffs)
      for root in roots:
        if (np.imag(root) <= 1.0e-8 and 0.5 <= np.real(root) <= 1.0):
          r = np.real(root)
    else:
      r = k/(k+1.0)
    r_arr[i] = r
  t = np.sqrt((r_arr*x)**2+H**2)/vp_new + np.sqrt(((1.0-r_arr)*x)**2+H**2)/vs_new
  if output_reflection_point:
    return t, r_arr*x
  else:
    return t

def tt_pbs_r(xr, H, vp_new=vp, vs_new=vs):
  #vp = 3870.0
  #vs = 1940.0
  st1 = xr / np.sqrt(xr**2+H**2)
  st2 = st1 / vp_new * vs_new
  x2 = H * st2 / np.sqrt(1.0-st2**2)
  return np.sqrt(H**2+xr**2)/vp_new+np.sqrt(H**2+x2**2)/vs_new, xr+x2


def tt_sbs(x, H, output_reflection_point=False, vp_new=vp, vs_new=vs):
  #vp = 3870.0
  #vs = 1920.0
  #H = 2950.0
  t = np.sqrt((0.5*x)**2+H**2) / vs_new * 2
  if output_reflection_point:
    return t, 0.5*x
  else:
    return t

def tt_sbs_r(xr, H, vp_new=vp, vs_new=vs):
  #vp = 3870.0
  #vs = 1940.0
  #H = 2950.0
  return np.sqrt(xr**2+H**2) / vs_new * 2, xr*2

def tt_pbp(x, H, output_reflection_point=False, vp_new=vp, vs_new=vs):
  #vp = 3870.0
  #vs = 1940.0
  #H = 3000.0
  t = np.sqrt((0.5*x)**2+H**2) / vp_new * 2
  if output_reflection_point:
    return t, 0.5*x
  else:
    return t

def tt_ses(x, H, output_reflection_point=False, vp_new=vp, vs_new=vs):
  #vp = 3870.0
  #vs = 1930.0
  #H = 2390.0
  t = np.sqrt((0.5*x)**2+H**2) / vs_new * 2
  if output_reflection_point:
    return t, 0.5*x
  else:
    return t

def tt_ses_r(xr, H, vp_new=vp, vs_new=vs):
  #vp = 3870.0
  #vs = 1940.0
  #H = 2950.0
  return np.sqrt(xr**2+H**2) / vs_new * 2, xr*2
