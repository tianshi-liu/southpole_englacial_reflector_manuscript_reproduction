import numpy as np
from ani_utils import ctensor_c21, get_wave_ani, get_wave_snell, convert_c21_to_ctensor

# return elastic tensors and density for different types of ice fibrics

A = 14.06e3
C = 15.24e3
L = 3.06e3
N = 3.455e3
F = 5.88e3
rho = 917.0

def ice_iso():
  la = (6*A+C-4*L+8*F-10*N)/15.0
  mu = (A+C+6*L-2*F+5*N)/15.0
  la2mu = la + 2.0 * mu
  return la2mu,   la,   la,    0.0,    0.0,    0.0, \
               la2mu,   la,    0.0,    0.0,    0.0, \
                     la2mu,    0.0,    0.0,    0.0, \
                                mu,    0.0,    0.0, \
                                        mu,    0.0, \
                                                mu   

def ice_cone(open_angle):
  ct = np.cos(open_angle * np.pi / 180.0) # angle goes from 0.0 (VTI) to 90.0 (ISO)

  X = 1.0 + ct + ct**2
  Y = ct**3 + ct**4

  C11 = 1.0/120.0*(    A        * (45.0 + 19.0 * X + 9.0 * Y) +
                   3.0*C        * (15.0 - 7.0  * X + 3.0 * Y) +
                   2.0*(2.0*L+F)* (15.0 +        X - 9.0 * Y)
                   )
  C33 = 1.0/15.0 *(
                       A        * (15.0 - 7.0  * X + 3.0 * Y) +
                   3.0*C        * (              X +       Y) +
                   2.0*(2.0*L+F)* (       2.0  * X - 3.0 * Y)
                   )
  C44 = 1.0/30.0 *((A+C-2.0*F)  * (       2.0  * X - 3.0 * Y) +
                   3.0*L        * (5.0  -        X + 4.0 * Y) +
                   5.0*N        * (3.0  -        X          )
                   )
  C66 = 1.0/120.0*((A+C-2.0*F)  * (15.0 - 7.0  * X + 3.0 * Y) +
                   12.0*L       * (5.0  -        X -       Y) +
                   40.0*N       *                X  
                   )
  C13 = 1.0/30.0 *(3.0*A        * (5.0  -        X -       Y) +
                   (C-4.0*L)    * (       2.0  * X - 3.0 * Y) -
                   10.0*N       * (3.0  -        X          ) +
                   F            * (15.0 +        X + 6.0 * Y)
                   )
  return  C11, C11-2.0*C66, C13,  0.0,  0.0,  0.0, \
               C11,         C13,  0.0,  0.0,  0.0, \
                            C33,  0.0,  0.0,  0.0, \
                                  C44,  0.0,  0.0, \
                                        C44,  0.0, \
                                              C66


def ice_pgirdle(open_angle):
  theta0 = open_angle * np.pi / 180.0
  S2 = np.sin(theta0*2.0) / theta0 / 2.0
  S4 = np.sin(theta0*4.0) / theta0 / 4.0
  C11 = A
  C22 = 1.0/8.0*(A*(3.0+4.0*S2+S4) + C*(3.0-4.0*S2+S4) + 2.0*(2.0*L+F)*(1.0-S4))
  C33 = 1.0/8.0*(A*(3.0-4.0*S2+S4) + C*(3.0+4.0*S2+S4) + 2.0*(2.0*L+F)*(1.0-S4))
  C44 = 1.0/8.0*((A+C-2.0*F)*(1.0-S4) + 4.0*L*(1.0+S4))
  C55 = 1.0/2.0*(L*(1.0+S2) + N*(1.0-S2))
  C66 = 1.0/2.0*(L*(1.0-S2) + N*(1.0+S2))
  C12 = 1.0/2.0*((A-2.0*N)*(1.0+S2) + F*(1.0-S2))
  C13 = 1.0/2.0*((A-2.0*N)*(1.0-S2) + F*(1.0+S2))
  C23 = 1.0/8.0*((A+C-4.0*L)*(1.0-S4) + 2.0*F*(3.0+S4))
  return C11, C12, C13, 0.0, 0.0, 0.0, \
              C22, C23, 0.0, 0.0, 0.0, \
                   C33, 0.0, 0.0, 0.0, \
                        C44, 0.0, 0.0, \
                             C55, 0.0, \
                                  C66

def ice_tgirdle(open_angle):
  theta0 = open_angle * np.pi / 180.0
  st = np.sin(theta0)
  st2 = st**2
  st4 = st**4
  C11 = 1.0/15.0  * (A             * (15.0 - 10.0*st2 + 3.0*st4) + \
                     3.0*C         *                        st4  + \
                     2.0*(2.0*L+F) * (        5.0*st2 - 3.0*st4)
                     )
  C33 = 1.0/120.0 * (A             * (45.0 + 10.0*st2 + 9.0*st4) + \
                     3.0*C         * (15.0 - 10.0*st2 + 3.0*st4) + \
                     2.0*(2.0*L+F) * (15.0 + 10.0*st2 - 9.0*st4)
                     )
  C44 = 1.0/120.0 * ((A+C-2.0*F)   * (15.0 - 10.0*st2 + 3.0*st4) + \
                     12.0*L        * ( 5.0            -     st4) + \
                     40.0*N        *              st2
                     )
  C55 = 1.0/30.0  * ((A+C-2.0*F)   * (        5.0*st2 - 3.0*st4) + \
                     3.0*L         * ( 5.0 -  5.0*st2 + 4.0*st4) + \
                     5.0*N         * ( 3.0 -      st2          )
                     )
  C12 = 1.0/30.0 *  (3.0*A         * ( 5.0            -     st4) + \
                     (C-4.0*L)     * (        5.0*st2 - 3.0*st4) - \
                     10.0*N        * ( 3.0 -      st2          ) + \
                     F             * (15.0 -  5.0*st2 + 6.0*st4)
                     )
  return C11,     C12,      C12,       0.0,    0.0,    0.0, \
                  C22, C22-2.0*C44,    0.0,    0.0,    0.0, \
                            C22,       0.0,    0.0,    0.0, \
                                       C44,    0.0,    0.0, \
                                               C55,    0.0, \
                                                       C55

def get_rho():
  return rho

def get_c(fabric='iso', open_angle=None):
  if fabric=='iso': return convert_c21_to_ctensor(ice_iso())
  elif fabric == 'cone': return convert_c21_to_ctensor(ice_cone(open_angle))
  elif fabric == 'pgirdle': return convert_c21_to_ctensor(ice_pgirdle(open_angle))
  elif fabric == 'tgirdle': return convert_c21_to_ctensor(ice_tgirdle(open_angle))
  else: return None

'''
#print(carr21)
la = (6*A+C-4*L+8*F-10*N)/15.0
mu = (A+C+6*L-2*F+5*N)/15.0
vs_iso = np.sqrt(mu/rho)
vp_iso = np.sqrt((la+2.0*mu)/rho)

c = np.zeros((3,3,3,3))
for i in range(3):
  for j in range(3):
    for p in range(3):
      for q in range(3):
        c[i,j,p,q] = ctensor_c21(carr21,i,j,p,q)

takeoff = 60.0 * np.pi / 180.0
#v, G = get_wave_ani(c/rho, np.array([np.sin(takeoff), 0.0, -np.cos(takeoff)]))
#print(v)
#print(G)
k, G = get_wave_snell(np.sin(takeoff)/vs_iso, c/rho, reflection=True)
print(k)
print(1.0/np.sqrt(np.sum(k*k,axis=0)))
print(G)
'''
