import numpy as np
A = np.loadtxt('model_firn_varice_lithosed_att_layer')
v_rep = 1.97
t = 0.0
for i in range(len(A)-2):
  h = A[i,0]
  vs = A[i,1]
  t += h/vs - h/v_rep
  print (f"{h}, {vs}")

print(t*2)
