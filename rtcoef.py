import numpy as np

# RTCOEF calculates P/SV reflection/transmission coefficients
# for interface between two solid layers, based on equations 5.40
# (p. 144-145) of Aki and Richards (2nd edition).

#  Inputs:    vp1     =  P-wave velocity of layer 1 (top layer)
#  (real)     vs1     =  S-wave velocity of layer 1
#             den1    =  density of layer 1
#             vp2     =  P-wave velocity of layer 2 (bottom layer)
#             vs2     =  S-wave velocity of layer 2
#             den2    =  density of layer 2
#             hslow   =  horizontal slowness (ray parameter)
#  Returns:   rt(0)   =  down P to P up     (refl)
#  (complex)  rt(1)   =  down P to S up     (refl)
#             rt(2)   =  down P to P down   (tran)
#             rt(3)   =  down P to S down   (tran)
#             rt(4)   =  down S to P up     (refl)
#             rt(5)   =  down S to S up     (refl)
#             rt(6)   =  down S to P down   (tran)
#             rt(7)   =  down S to S down   (tran)
#             rt(8)   =    up P to P up     (tran)
#             rt(9)  =    up P to S up     (tran)
#             rt(10)  =    up P to P down   (refl)
#             rt(11)  =    up P to S down   (refl)
#             rt(12)  =    up S to P up     (tran)
#             rt(13)  =    up S to S up     (tran)
#             rt(14)  =    up S to P down   (refl)
#             rt(15)  =    up S to S down   (refl)

# NOTE:  All input variables are real.
#        All output variables are complex!
#        Coefficients are not energy normalized.

def RTCOEF(vp1, vs1, den1, vp2, vs2, den2, hslow):

   rt = np.zeros([16,], dtype=complex)

   alpha1 = complex(vp1, 0.)
   beta1 = complex(vs1, 0.)
   rho1 = complex(den1, 0.)
   alpha2 = complex(vp2, 0.)
   beta2 = complex(vs2, 0.)
   rho2 = complex(den2, 0.)
   p = complex(hslow, 0.)

   cone = complex(1., 0.)
   ctwo = complex(2., 0.)

   si1 = alpha1*p
   si2 = alpha2*p
   sj1 = beta1*p
   sj2 = beta2*p

   ci1 = np.sqrt(cone - si1**2)
   ci2 = np.sqrt(cone - si2**2)
   cj1 = np.sqrt(cone - sj1**2)
   cj2 = np.sqrt(cone - sj2**2)

   term1 = (cone - ctwo*beta2*beta2*p*p)
   term2 = (cone - ctwo*beta1*beta1*p*p)
   a = rho2*term1 - rho1*term2
   b = rho2*term1 + ctwo*rho1*beta1*beta1*p*p
   c = rho1*term2 + ctwo*rho2*beta2*beta2*p*p
   d = ctwo*(rho2*beta2*beta2 - rho1*beta1*beta1)
   E = b*ci1/alpha1 + c*ci2/alpha2
   F = b*cj1/beta1 + c*cj2/beta2
   G = a - d*ci1*cj2/(alpha1*beta2)
   H = a - d*ci2*cj1/(alpha2*beta1)
   DEN = E*F + G*H*p*p

   trm1 = b*ci1/alpha1 - c*ci2/alpha2
   trm2 = a + d*ci1*cj2/(alpha1*beta2)
   rt[0] = (trm1*F - trm2*H*p*p)/DEN              #refl down P to P up

   trm1 = a*b + c*d*ci2*cj2/(alpha2*beta2)
   rt[1] = (-ctwo*ci1*trm1*p)/(beta1*DEN)       #refl down P to S up

   rt[2] = ctwo*rho1*ci1*F/(alpha2*DEN)         #trans down P to P down

   rt[3] = ctwo*rho1*ci1*H*p/(beta2*DEN)        #trans down P to S down

   trm1 = a*b + c*d*ci2*cj2/(alpha2*beta2)
   rt[4] = (-ctwo*cj1*trm1*p)/(alpha1*DEN)      #refl down S to P up

   trm1 = b*cj1/beta1 - c*cj2/beta2
   trm2 = a + d*ci2*cj1/(alpha2*beta1)
   rt[5] = -(trm1*E - trm2*G*p*p)/DEN             #refl down S to S up

   rt[6] = -ctwo*rho1*cj1*G*p/(alpha2*DEN)      #trans down S to P down

   rt[7] = ctwo*rho1*cj1*E/(beta2*DEN)          #trans down S to S down


   trm1 = b*ci1/alpha1 - c*ci2/alpha2
   trm2 = a + d*ci2*cj1/(alpha2*beta1)
   rt[10] = -(trm1*F + trm2*G*p*p)/DEN            #refl up P to P down

   trm1 = a*c + b*d*ci1*cj1/(alpha1*beta1)
   rt[11] = (ctwo*ci2*trm1*p)/(beta2*DEN)       #refl up P to S down

   rt[8] = ctwo*rho2*ci2*F/(alpha1*DEN)         #trans up P to P up

   rt[9] = -ctwo*rho2*ci2*G*p/(beta1*DEN)      #trans up P to S up

   trm1 = a*c + b*d*ci1*cj1/(alpha1*beta1)
   rt[14] = (ctwo*cj2*trm1*p)/(alpha2*DEN)      #refl up S to P down

   trm1 = b*cj1/beta1 - c*cj2/beta2
   trm2 = a + d*ci1*cj2/(alpha1*beta2)
   rt[15] = (trm1*E + trm2*H*p*p)/DEN             #refl up S to S down

   rt[12] = ctwo*rho2*cj2*H*p/(alpha1*DEN)      #trans up S to P up

   rt[13] = ctwo*rho2*cj2*E/(beta1*DEN)         #trans up S to S up

   return rt
