# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:24:00 2022

@author: teddy
"""
import arviz as az
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy import stats
import seaborn as sns
import matplotlib
import math
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import cm

np.random.seed(0)
def wave_tel(A_rt,r,rho,theta,phi,times,partitions,DivB_free):
    # A_rt (NxLxt_steps) array of measurements
    # r (NxLxt_steps) array of positions
    # rho (vector) defines k magnitude values [0,infinity] which we compute P_wk
    # theta (vector) defines k azimuthal angle values [0,2pi] which we compute P_wk
    # phi (vector) defines k polar angle values [0,pi] which we compute P_wk
    # times (vector) defines sample times
    # partitions (integer) defines the number of partitions of times to make when calculating ensamble average M(w)
    
    noise = 10**(-3)
    # create array of coordinates
    theta_mat, phi_mat, rho_mat = np.meshgrid(theta,phi,rho)
    theta_pts,phi_pts,rho_pts = np.size(theta),np.size(phi),np.size(rho)
    X = rho_mat*np.sin(phi_mat)*np.cos(theta_mat)
    Y = rho_mat*np.sin(phi_mat)*np.sin(theta_mat)
    Z = rho_mat*np.cos(phi_mat)
    
    k_set = np.zeros([3,np.size(X)])
    k_set[0,:] = X.flatten()
    k_set[1,:] = Y.flatten()
    k_set[2,:] = Z.flatten()

    kxs_ind = np.linspace(0,np.size(theta)-1,np.size(theta)).astype(int)
    kys_ind = np.linspace(0,np.size(phi)-1,np.size(phi)).astype(int)
    kzs_ind = np.linspace(0,np.size(rho)-1,np.size(rho)).astype(int)
    kx_ind,ky_ind,kz_ind = np.meshgrid(kxs_ind,kys_ind,kzs_ind)
    k_set_ind = np.array([kx_ind.flatten(),ky_ind.flatten(),kz_ind.flatten()])

    N = np.shape(r)[0]
    L = np.shape(r)[1]
    H_k = np.zeros([L*N,L,np.shape(k_set)[1]],dtype=np.complex_)
    for k_ind in range(np.shape(k_set)[1]):
        k = k_set[:,k_ind]
        for i in range(N):
            r_i = r[i,:,0]
            H_k[(3*i):(3*i+3),:,k_ind] = np.eye(L)*np.exp(np.dot(k,r_i)*1j)
    
    # calculate M_w using partitions of data
    t_set = int(np.size(time)/partitions)
    for part in range(partitions):
        # transform from time into frequency space
        A_wr = np.fft.fft(A_rt[:,:,t_set*part:t_set*(part+1)],axis=2)
        A_freq = np.fft.fftfreq(A_rt[:,:,t_set*part:t_set*(part+1)].shape[-1],d=dt)
        
        M_w_ = np.zeros([N*L,N*L,len(A_freq),partitions],dtype=np.complex_)
        w = np.zeros([len(A_freq)])
        for w_ind in range(len(A_freq)):
            w[w_ind] = A_freq[w_ind]
            A_wr_vector = np.reshape(A_wr[:,:,w_ind],(L*N,1))
            # calculate spacial correlation matrix
            M_w_[:,:,w_ind,part] = np.matmul(A_wr_vector,np.conjugate(np.transpose(A_wr_vector)))
    M_w = np.mean(M_w_,axis=3)
    
    #  compute power P_wk
    P_wk_ = np.zeros([len(A_freq),theta_pts,phi_pts,rho_pts])
    
    # majority of computation time is in this loop
    for w_ind in range(len(A_freq)):
        for k_ind in range(np.shape(k_set)[1]):
            M_w_temp = M_w[:,:,w_ind]
            H_k_temp = H_k[:,:,k_ind]
            
            H_mat_t = np.conjugate(np.transpose(H_k_temp))
            M_w_temp = M_w_temp + np.eye(N*L)*np.random.normal(0,noise*np.max(np.sqrt(np.real(M_w_temp)**2 + np.imag(M_w_temp)**2)),1)
            
            if DivB_free == True:
                C_mat = np.eye(L) + np.outer(k_set[:,k_ind],np.conjugate(k_set[:,k_ind]))/(la.norm(k_set[:,k_ind])**2)
            else:
                C_mat = np.eye(L)
                
            C_mat_t = np.conjugate(np.transpose(C_mat))
            
            temp = np.matmul(C_mat_t, H_mat_t)
            temp1 = np.matmul(temp,np.linalg.inv(M_w_temp))
            temp2 = np.matmul(temp1,H_k_temp)
            temp3 = np.matmul(temp2, C_mat)
            
            xt, yt, zt = k_set_ind[0,k_ind],k_set_ind[1,k_ind],k_set_ind[2,k_ind]
            P_wk_[w_ind,xt,yt,zt] = np.sqrt(np.real(np.trace(np.linalg.inv(temp3)))**2 + np.imag(np.trace(np.linalg.inv(temp3)))**2)
      
    return [P_wk_, A_freq, k_set]

def calc_RLEP(r): 
    # function calc_RLEP takes in array, each row of which represents a point in 3d space
    # find the number of satellites as the number of rows of r
    N = np.size(r[:,0])
    # find the mesocentre rb
    rb = np.mean(r, axis=0)
    # calculate the Volumetric Tensor R
    R = np.zeros([3,3])
    for i in range(N):
        R += np.outer(r[i,:]-rb, r[i,:]-rb)/N
    # find the eigenvalues of R as value in lambdas
    temp = la.eig(R)
    lambdas = temp[0]
    # find semiaxes of quasi-ellipsoid a,b,c
    # check if eigenvalues are real
    if any(np.imag(lambdas) != np.zeros(3)):
        raise ValueError('Eigenvalue has imaginary component')
    lambdas_real = np.real(lambdas)
    #print(lambdas_real)
    [c,b,a] = np.sqrt( np.sort(lambdas_real) )
    # calculate L,E,P
    L = 2*a
    E = 1 - b/a
    P = 1 - c/b
    return [R,L,E,P]

def sphere2cart(r, theta, phi):
    return [
         r * math.sin(phi) * math.cos(theta),
         r * math.sin(phi) * math.sin(theta),
         r * math.cos(phi)
    ]

# %% scan parameters

# resolution of wavevector scan
mag_res = 50                # resolution of k magnitudes
theta_res = 36              # angular resolution of k

# spacecraft positions
r = np.array([[1,0,-1/np.sqrt(2)],[-1,0,-1/np.sqrt(2)],[0,-1,1/np.sqrt(2)],[0,1,1/np.sqrt(2)]])

# time samples in timeseries
t_samples = 2**7    

# number of partions of t_samples to obtain average over
partitions = 2**2   

# time cadence of measurements
dt = .05     

# constraint that divergence of magnetic field be 0       
DivB_free = False

# %% derived parameters
r = r - np.mean(r,axis=0)
N = np.shape(r)[0]               # number of s/c
[R,L,E,P] = calc_RLEP(r)        # shape of s/c configuration

L = np.shape(r)[1]  # dim of data (3 for mag field B)
w_max = 1/(2*dt)
r = np.tile(np.expand_dims(r,axis=2),(1,1,t_samples))
time = dt*np.arange(t_samples)

# %% generate synthetic S/C measurements

# magnitude and directionof true wavevector
rho_ = 1
theta_ = 0.9*np.pi
phi_ = 0.4*np.pi

# phase, amplitude, and frequency of true wave
phi_0 = 0
coeff = [1]
omega = w_max*np.random.rand(1)    # in Hz


k_sphere = np.array([[ rho_, theta_, phi_ ]])    # r, theta, phi

k_field = np.array([[k_sphere[0][0]*np.sin(k_sphere[0][2])*np.cos(k_sphere[0][1]),
                    k_sphere[0][0]*np.sin(k_sphere[0][2])*np.sin(k_sphere[0][1]),
                    k_sphere[0][0]*np.cos(k_sphere[0][2])]])   


A_rt = np.zeros([N,L,t_samples]).astype(complex)
for ii in range(t_samples):
    t = time[ii]
    for sc in range(N):
        for j in range(np.size(coeff)):
            c=coeff[j]
            ww = omega[j]
            A_rt[sc,:,ii] = A_rt[sc,:,ii] + c*np.exp(1j*(ww*2*np.pi*t + np.dot(k_field[j,:],r[sc,:,ii]) + phi_0))
            #A_rt[sc,:,0] = A_rt[sc,:,ii] + c*np.sin(ww*2*np.pi*t + np.dot(k_field[j,:],r[sc,:,ii]) + phi_0)

# %% scan using Wave-telescope
rho = np.linspace(0,2,mag_res)
theta = np.linspace(0,2*np.pi,theta_res)
phi = np.linspace(0,np.pi,int(theta_res/2))

print('computing wave-telescope...')
[P_wk, w_set, k_set] = wave_tel(A_rt, r, rho, theta, phi, time, partitions, DivB_free)

# %% find peak in wave-telescope scan
w_ind = np.where(np.sum(P_wk,axis=(1,2,3)) == np.max(np.sum(P_wk,axis=(1,2,3))))[0][0]
w_found = w_set[w_ind]
print('omega true: %.2f' %omega)
print('omega found: %.3f +/- %.3f' %(w_found, np.median(np.diff(w_set)) ))
print('omega error: %.2f%% \n' %(100*np.abs(w_found - omega)/omega) )

P_k = P_wk[w_ind,:,:,:]
k_ind1 = np.where(P_k == np.max(P_k))[0][0]
k_ind2 = np.where(P_k == np.max(P_k))[1][0]
k_ind3 = np.where(P_k == np.max(P_k))[2][0]
k_found = np.array([rho[k_ind3],theta[k_ind1],phi[k_ind2]])

if k_ind3 == 0:
    rho_dt = (rho[k_ind3+1] - rho[k_ind3])
elif k_ind3 == mag_res-1:
    rho_dt = (rho[k_ind3] - rho[k_ind3-1])
else:
    rho_dt = (rho[k_ind3+1] - rho[k_ind3-1])/2
print('k true: [k_r, k_theta, k_phi] = [%.3f,%.3f,%.3f]' %(k_sphere[0][0],k_sphere[0][1],k_sphere[0][2]))
print('k found: [k_r, k_theta, k_phi] = [%.3f,%.3f,%.3f] +/- [%.3f,%.3f,%.3f]' %(k_found[0],k_found[1],k_found[2], rho_dt, 2*np.pi/theta_res, np.pi/theta_res))
print('k error: %.2f%% \n' %(100*la.norm(np.array([ rho_, theta_, phi_ ]) - k_found)/la.norm(np.array([ rho_, theta_, phi_ ]))) )

# %% heatmap figures
print('plotting results...')

cmap = plt.get_cmap('magma', 100)
tot_power = np.sum(P_wk,axis=(0,1,2,3))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# power over direction: integrated over magnitude of wavevector
Power_theta_phi = np.sum(P_k,axis=(2))/tot_power
XX, YY = np.meshgrid(phi, theta)

plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax1=plt.subplot(111)
plot1 = ax1.pcolormesh(XX,YY,Power_theta_phi,cmap=cmap, shading='auto', norm=matplotlib.colors.LogNorm())
cbar = plt.colorbar(plot1,ax=ax1, pad = .01, aspect=15,)
cbar.set_label(label=r'$P(k_{\theta},k_{\phi})/P_{tot}$', weight='bold', size=18)
#plt.xscale("log")
plt.xlabel(r'$k_\phi$',fontsize=15)
ylabel = plt.ylabel(r'$k_\theta$',fontsize=20)
ylabel.set_rotation(0)
#plt.savefig('figures/wavevector_direction.png', format='png',dpi = 600)

# %% make one combined figure

# power over theta: integrated over direction of wavevector
Power_theta = np.sum(P_wk,axis=(0,2,3))/tot_power

# power over phi: integrated over direction of wavevector
Power_phi = np.sum(P_wk,axis=(0,1,3))/tot_power

# power over k: integrated over direction of wavevector
Power_k = np.sum(P_wk,axis=(0,1,2))/tot_power

# power over frequency: integrated over wavevector
Power_w = np.sum(P_wk,axis=(1,2,3))/tot_power


fig, ax = plt.subplots(2, 2, sharey = False, sharex = False, figsize=(10,8))
ax[0,0].scatter(theta,Power_theta, c='b', s=20, alpha=0.7)
ax[0,0].plot([k_sphere[0][1],  k_sphere[0][1]], [np.min(Power_theta),np.max(Power_theta)], c='k', linewidth=2, alpha=0.5, linestyle = '--')
ax[0,0].set_xlabel(r'$k_\theta$',fontsize=18)
ax[0,0].set_title(r'$P(k_\theta)/P_{tot}$',fontsize=20)
ax[0,0].grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.4)
ax[0,0].minorticks_on()
ax[0,0].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
ax[0,0].set_xticks([0,np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax[0,0].set_xticklabels(['$0$','$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])

ax[0,1].scatter(phi, Power_phi, c='b', s=20, alpha=0.7)
ax[0,1].plot([k_sphere[0][2],  k_sphere[0][2]], [np.min(Power_phi),np.max(Power_phi)], c='k', linewidth=2, alpha=0.5, linestyle = '--')
ax[0,1].set_xlabel(r'$k_\phi$',fontsize=18)
ax[0,1].set_title(r'$P(k_\phi)/P_{tot}$',fontsize=20)
ax[0,1].grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.4)
ax[0,1].minorticks_on()
ax[0,1].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
ax[0,1].set_xticks([0,np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax[0,1].set_xticklabels(['$0$','$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$'])

ax[1,0].scatter(rho, Power_k, c='b', s=20, alpha=0.7)
ax[1,0].plot([k_sphere[0][0],  k_sphere[0][0]], [np.min(Power_k),np.max(Power_k)], c='k', linewidth=2, alpha=0.5, linestyle = '--')
#ax[1,0].set_xscale('log')
ax[1,0].set_xlabel(r'$k/L$',fontsize=18)
ax[1,0].set_title(r'$P(k)/P_{tot}$',fontsize=20)
ax[1,0].grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.4)
ax[1,0].minorticks_on()
ax[1,0].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

ax[1,1].scatter(w_set, Power_w, c='b', s=20, alpha=0.7)
ax[1,1].plot([omega[0],  omega[0]], [np.min(Power_w),np.max(Power_w)], c='k', linewidth=2, alpha=0.5, linestyle = '--')
ax[1,1].set_xlabel(r'$\omega$ Hz',fontsize=18)
ax[1,1].set_title(r'$P(\omega)/P_{tot}$',fontsize=20)
ax[1,1].grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.4)
ax[1,1].minorticks_on()
ax[1,1].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
ax[1,1].set_xlim(0,w_max)

plt.tight_layout()
plt.savefig('figures/WaveTelescope_Scan.png', format='png',dpi = 600)


# %% 3d plot of wavevector 

# color is Power_theta_phi
u = np.linspace(0, 2 * np.pi, theta_res)
v = np.linspace(0, np.pi, int(theta_res/2))

# create the sphere surface
x=1 * np.outer(np.cos(u), np.sin(v))
y=1 * np.outer(np.sin(u), np.sin(v))
z=1 * np.outer(np.ones(np.size(u)), np.cos(v))

Power_theta_phi = P_k[:,:,k_ind3]
myheatmap = Power_theta_phi / np.max(Power_theta_phi)

fig = plt.figure(figsize=(5,4.5))
ax = fig.add_subplot(1,1,1, projection='3d')
plot = ax.plot_surface(x, y, z, rstride=1, cstride=1, vmax = 1, vmin=0.5, facecolors=cm.hot(myheatmap), edgecolors='#000000', alpha=0.6)
#ax.plot([0,0,0], sphere2cart(2*rho_, theta_, phi_))

ax.view_init(elev=30, azim=135)
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)
ax.set_zlabel('z', fontsize=18)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
fig.suptitle(r'$P(k_{\theta}, k_{\phi})$ at $k=%.2f$ and $\omega=%.2f$ Hz' %(rho[k_ind3], w_set[w_ind]))
plt.tight_layout()
plt.savefig('figures/directional_scan.png', format='png',dpi = 600)

