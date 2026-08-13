import numpy as np

SMALL_VAL = 1e-5

def voigt_index(i, j):
    """
    Maps tensor indices (i, j) to Voigt notation index.
    
    Parameters:
        i (int): Row index (0-based).
        j (int): Column index (0-based).
        
    Returns:
        int: Voigt notation index (0-based).
    """
    if i > j:  # Ensure the indices are in upper triangular order (i <= j)
        i, j = j, i

    mapping = {
        (0, 0): 0,  # sigma_11
        (1, 1): 1,  # sigma_22
        (2, 2): 2,  # sigma_33
        (1, 2): 3,  # sigma_23
        (0, 2): 4,  # sigma_13
        (0, 1): 5   # sigma_12
    }

    return mapping.get((i, j), None)

def get_wave_ani(a, n, plane=np.array([0.0,1.0,0.0])):
    """
    Params:
    =================================
    a: np.ndarray, shape(3,3,3,3)
        density normalized tensor: a = cijkl / rho
    n: np.ndarray, shape(3)
        slowness direction
    plane: np.ndarray, shape(3)
        normal to the P-SV plane in case of degeneracy

    Note here that n must be a real unit vector
    k := n/v is the slowness vector, and it satisfies
    (k.a.k-I)g = 0 => (n.a.n-(v^2)I)g = 0

    in case of degeneracy, g is defined to be parallel (SV) and 
    perpendicular (SH) to plane

    quasi-S waves are ordered such that v[1], G[:,1] represent qSV, v[2], G[:,2] represent qSH

    Returns:
    ========================
    v: np.ndarray shape(3)
        phase velocity, descending order
    G: np.ndarray shape(3,3)
        polarization vector, G[:,i] is polarization of i'th mode
    """
    # christoffel matrix
    gamma = np.einsum("ijkl,j,l-> ik",a,n,n) # aijkl n_j n_l

    # find eigenvalues
    eval,evec = np.linalg.eig(gamma)
    v = np.sqrt(np.real(eval))
    idx = np.argsort(abs(v))[::-1] # descending order
    evec = evec[:,idx]
    v = v[idx]
    if abs(v[1] - v[2]) < v[0] * SMALL_VAL: # degenerate case
        evec[:,1] = np.cross(evec[:,0], plane) # qSV
        evec[:,2] = np.cross(evec[:,1], evec[:,0]) # qSH
    if abs(np.dot(plane, evec[:,1])) > abs(np.dot(plane, evec[:,2])):
        v[[1,2]] = v[[2,1]]
        evec[:,[1,2]] = evec[:,[2,1]]
    return v, evec

def get_group_velocity(k, a, g):
    """
    Params:
    =================================
    a: np.ndarray, shape(3,3,3,3)
        density normalized tensor: a = cijkl / rho
    k: np.ndarray, shape(3)
        slowness vector, can be complex
    g: np.ndarray, shape(3)
        polarization vector, can be complex
    the three tensors satisfy the equation
    (k.a.k - I)g = 0
 
    Returns:
    U: np.ndarray, shape(3)
        group velocity vector, can be complex
    """
    return np.einsum("ijkl,l,j,k->i",a,k,g,g) # aijkl k_l g_j g_k

def get_wave_snell(p, a, reflection, plane=np.array([0.0,1.0,0.0])):
    """
    Params:
    =================================
    p: float
        ray parameter, horizontal slowness
    a: np.ndarray, shape(3,3,3,3)
        density normalized tensor: a = cijkl / rho
    reflection: bool
        is the reflection ray or not, in order to select the right eigenvalues
    plane: np.ndarray, shape(3)
        normal to the P-SV plane in case of degeneracy

    Returns:
    ========================
    k: np.ndarray shape(3,3)
        slowness vector, in form of (p, 0, nu), where nu can be complex
    G: np.ndarray shape(3,3)
        polarization vector, G[:,i] is polarization of i'th mode

    k and G satisfies
    (k.a.k - I) g = 0 => (nu^2 a[2,:,:,2] + nu*p*(a[0,:,:,2]+a[2,:,:,0]) + p^2 a[0,:,:,0] - I) g = 0
    in the most general case, the quadratic eigenvalue problem has 6 eigenvalues, in pairs of complex conjugates
    they are then selected in the following criteria (Cerveny 2001, pp. 50-51):
    (1) if nu is real, then Uz > 0 for reflection and Uz < 0 for transmission
    (2) if nu is complex, then Im(nu) > 0 for reflection and Im(nu) < 0 for transmission
    it is guaranteed that there are exactly 3 out of 6 eigenvalues selected for reflection and transmission

    quasi-S waves are ordered such that k[:,1], G[:,1] represent qSV, k[:,2], G[:,2] represent qSH
    """
    
    eval, evec = qep(a[2,:,:,2], p * (a[0,:,:,2] + a[2,:,:,0]), p * p * a[0,:,:,0] - np.eye(3))
    #print(eval)
    idx = np.argsort(np.real(eval*eval)) # ascending order, P wave would be the first one
    eval = eval[idx]
    evec = evec[:,idx]
    eval_valid = []
    evec_valid = []
    for i in range(6):
      nu = eval[i]
      g = evec[:,i]
      is_valid = False
      if (abs(np.imag(nu)) < SMALL_VAL):
        U = get_group_velocity(np.array([p, 0.0, nu]), a, g)
        if (reflection == (np.real(U[2]) > 0.0)): is_valid = True
      else:
        if (reflection == (np.imag(nu) > 0.0)): is_valid = True
      if is_valid:
        eval_valid.append(nu)
        evec_valid.append(g / np.sqrt(np.dot(g.conj(),g)))
    #print(evec_valid)
    if not (len(eval_valid) == 3): raise RuntimeError("wrong eigenvalue selection")
    if (abs(eval_valid[1] - eval_valid[2]) < abs(eval_valid[0]) * SMALL_VAL):
      norm = np.sqrt(abs(np.dot(plane, evec_valid[1]))**2 + abs(np.dot(plane, evec_valid[2]))**2)
      c1 = np.conjugate(np.dot(plane, evec_valid[1])) / norm
      c2 = np.conjugate(np.dot(plane, evec_valid[2])) / norm
      evec_valid[2], evec_valid[1] = c1*evec_valid[1]+c2*evec_valid[2], c2*evec_valid[1]-c1*evec_valid[2]
    evec_valid = np.array(evec_valid)
    evec_valid = evec_valid / np.sqrt(np.sum(abs(evec_valid)**2, axis=-1, keepdims=True)) # renormalize
    k = np.zeros(shape=(3,3), dtype=complex)
    G = np.zeros(shape=(3,3), dtype=complex)
    for i in range(3):
      k[:,i] = np.array([p, 0.0, eval_valid[i]])
      G[:,i] = evec_valid[i]
    if abs(np.dot(plane, G[:,1])) > abs(np.dot(plane, G[:,2])):
      k[:,[1,2]] = k[:,[2,1]]
      G[:,[1,2]] = G[:,[2,1]]
    return k, G

def qep(M, C, K):
    n = len(M)
    A = np.zeros((2*n, 2*n), dtype=float)
    A[:n,n:] = np.eye(n)
    A[n:,:n] = -np.matmul(np.linalg.inv(M), K)
    A[n:,n:] = -np.matmul(np.linalg.inv(M), C)
    eval, evec = np.linalg.eig(A)
    #print(eval)
    #print(evec)
    return eval, evec[:n,:]    

def get_traction_ani(c, G, k, n=np.array([0.0,0.0,1.0])):
    """
    Params:
    =================================
    c: np.ndarray, shape(3,3,3,3)
        elastic tensor: cijkl
    G: np.ndarray, shape(3,3)
        polarization vector, G[:,i] is polarization of i'th mode
    k: np.ndarray, shape(3,3)
        slowness vector, k[:,i] is slowness vector of i'th mode
    n: np.ndarray, shape(3)
        direction to take traction

    Returns:
    ========================
    Q: np.ndarray, shape(3,3)
        Q[:,i] is traction of i'th mode on surface of normal n
    """
    return np.einsum("ijkl,j,km,lm->im", c, n, k, G) # n . (C: kG)

def ctensor_c21(c21,i,j,k,l):
    m = voigt_index(i,j)
    n = voigt_index(k,l)
    if m > n:
        m,n = n,m
    idx = m * 6 + n - (m * (m + 1)) // 2

    return c21[idx]

def convert_c21_to_ctensor(carr21):
  c = np.zeros((3,3,3,3))
  for i in range(3):
    for j in range(3):
      for p in range(3):
        for q in range(3):
          c[i,j,p,q] = ctensor_c21(carr21,i,j,p,q)
  return c

def ctensor_rotate(c, R):
  return np.einsum("abcd,ia,jb,kc,ld->ijkl", c, R, R, R, R)
