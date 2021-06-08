import numpy as np



# Exact solution
# ue = lambda x,t: np.sin(x) * np.exp(-2*t)
ue = lambda x,t: np.sin(x)*(2*t-5)



# Diffusion, Convection Coefficients
mu = 8
gamma = 3



# Forcing term
# f = lambda x,t: (mu-2)*np.sin(x)*np.exp(-2*t) + gamma * np.cos(x)*np.exp(-2*t)
f = lambda x,t: (2+mu*(2*t-5))*np.sin(x) + gamma*(2*t-5)*np.cos(x)



# Boundary points
a = -1
b = 1



# Boundary conditions
# u_a = lambda a,t: np.sin(a)*np.exp(-2*t)         # Dirichlet boundary u(a,t)
u_a = lambda t: np.sin(a)*(2*t-5)
# u_xb = lambda b, t: np.cos(b)*np.exp(-2*t)     # Neumann boundary u_x(b,t)
# u_b = lambda b,t: np.sin(b)*np.exp(-2*t)         # Dirichlet boundary u(b,t)
u_b = lambda t: np.sin(b)*(2*t-5)
# ut = lambda x: np.sin(x)                         # Initial condition
ut = lambda x: -5*np.sin(x)
# u_x = lambda x,t: -5*np.cos(x)*np.exp(-2*t)
u_x = lambda x,t: -5*np.cos(x)*(2*t-5)




# Time interval
t0 = 0
T = 1


# Time points
n_t  = 150

# Time step
dt = (T-t0)/n_t

ts = np.linspace(t0,T,n_t+1)



# import numpy as np
from oned_fem_simulation import oned_mesh, oned_gauss,oned_shape 
from oned_fem_simulation import oned_bilinear, oned_linear
import scipy
from scipy.sparse import linalg
import matplotlib.pyplot as plt

# Generate the computational mesh
n_elements = 200  # specify number of elements

# Compute nodes and connectivity matrix
x, e_conn = oned_mesh(a,b,n_elements,'cubic')

n_nodes = len(x)   # number of nodes
n_dofs = len(e_conn[1,:])   # degrees of freedom per element

ut = ut(x)   # Initial solution
data = np.zeros(((n_dofs-1)*n_elements+1,len(ts)))
data[:,0] = ut    # solution snapshot at t = 0
for l,t in enumerate(ts): 
    if t == 0:
        continue
    
    # Index to keep track of equation numbers
    ide = np.zeros(n_nodes, dtype=int)  # initialize

    # Mark Dirichlet nodes by -1
    i_dir = [0,n_nodes-1]  # indices of dirichlet nodes
    ide[i_dir] = -1   # dirichlet nodes are marked by -1

    # Number remaining nodes from 0 to n_equations-1
    count = 0
    for i in range(n_nodes):
        if ide[i] == 0:
            ide[i] = count
            count = count + 1

    n_equations = count   # total number of equations


    # Initialize sparse stiffness matrix
    nnz = n_elements*n_dofs**2   # estimate the number of non-zero elements

    rows = np.zeros(nnz, dtype=int)   # row index
    cols = np.zeros(nnz, dtype=int)   # column index
    vals = np.zeros(nnz)              # matrix entries


    # Initialize the RHS
    c = np.zeros(n_equations)

    # Assembly
    r,w = oned_gauss(11) # Gauss rule accurate to degree (2n-1), n is the input of oned_gauss
    count = 0
    for i in range(n_elements):
        # local information
        i_loc = e_conn[i,:] # local node indices
        x_loc = x[i_loc]    # local nodes

        #compute shape function on element
        x_g,w_g,phi,phi_x,phi_xx = oned_shape(x_loc,r,w)

        # compute local stiffnes matrix
        M_loc = oned_bilinear(1,phi,phi,w_g) + \
                    dt*oned_bilinear(mu, phi_x,phi_x,w_g) + \
                        dt*oned_bilinear(gamma,phi_x,phi,w_g)

        # local RHS
        f_g = f(x_g,t)
        u_M_loc = data[i_loc,l-1]
        u_M_g = np.dot(phi,u_M_loc)
        c_loc = dt*oned_linear(f_g,phi,w_g) + oned_linear(u_M_g,phi,w_g)

        # global
        for j in range(n_dofs):
            # for each row
            j_test = i_loc[j]    # global node number
            j_eqn = ide[j_test]  # equation number


            if j_eqn >= 0 :
                #update RHS
                c[j_eqn] = c[j_eqn] + c_loc[j]

                for m in range(n_dofs):
                    # for each column
                    i_trial = i_loc[m]    #global node number
                    i_col = ide[i_trial]  # equation number

                    if i_col >= 0:
                        # interio node: fill column
                        rows[count] = j_eqn
                        cols[count] = i_col
                        vals[count] = M_loc[j,m]
                        count = count + 1
                    else:
                        # Dirichlet node: apply dirichlet condition
                        u_dir = ue(x_loc[m],t)
                        c[j_eqn] = c[j_eqn] - M_loc[j,m]*u_dir

    # Delete entries that weren't filled
    noz = len(rows) - (count + 1)    # read # of rows not affected by count
    rows = rows[:-noz]               # delete the last 'noz' entries
    cols = cols[:-noz]               # delete the last 'noz' entries
    vals_M = vals_M[:-noz]               # delete the last 'noz' entries
    vals_K = vals_K[:-noz]
    vals_W = vals_W[:-noz]

    # Assemble sparse stiffness matrix
    M = scipy.sparse.coo_matrix((vals_M,(rows,cols)), shape=(n_equations,n_equations)).tocsr()
    K = scipy.sparse.coo_matrix((vals_K,(rows,cols)), shape=(n_equations,n_equations)).tocsr()
    W = scipy.sparse.coo_matrix((vals_W,(rows,cols)), shape=(n_equations,n_equations)).tocsr()
    A = M + dt*(K + W)

    c = dt*c + M*ut[1:-1]                     # RHS for interior nodes
 
    # Compute finite element solution
    ua = np.zeros(n_nodes)
    ua[i_dir] = ue(x[i_dir],t)           # apply prescribed values at Dirichlet nodes
    ua[ide >=0] = linalg.spsolve(A,c)    # solve the system at interior nodes
    
    ut = ua
    data[:,l] = ut
    
    
    
    
    

    
    
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(x,ue(x,t),label='exact solution')
    ax1.plot(x,data[:,l],'r o',label='numerical solution')
    ax1.legend(loc='upper left')
    ax1.set_xlim([-1,1])
    ax1.set_ylabel('u(x)')
    ax1.set_xlabel('x')

    ax2.plot(x,np.abs(ua-ue(x,t)),'r--')
    ax2.set_xlim([-1,1])
    ax2.set_ylabel('|ua(x)-ue(x)|')
    ax2.set_xlabel('x')
    ax2.set_title('Error')
    plt.show()
    
