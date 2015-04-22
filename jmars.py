from constants import *
import numpy as np
import scipy as sp
from scipy.optimize import fmin_l_bfgs_b
from numpy import linalg as LA
import numpy.matlib

def func(params, *args):
    global counter
    print counter
    counter += 1
    y = args[0]
    z = args[1]
    s = args[2]
    Nums = args[3]
    Numas = args[4]
    Numa = args[5]

    v_u = params[:(U*K)].reshape((U,K), order='F')
    b_u = params[(U*K):(U*K + U)].reshape((U,1), order='F')
    theta_u = params[(U*K + U):(U*K + U + U*A)].reshape((U,A), order='F')

    v_m = params[(U*K + U + U*A):(U*K + U + U*A + M*K)].reshape((M,K), order='F')
    b_m = params[(U*K + U + U*A + M*K):(U*K + U + U*A + M*K + M)].reshape((M,1), order='F')
    theta_m = params[(U*K + U + U*A + M*K + M):(U*K + U + U*A + M*K + M + M*A)].reshape((M,A), order='F')

    M_a = params[(U*K + U + U*A + M*K + M + M*A):].reshape((A,K), order='F')

    M_sum = np.diag(M_a.sum(0))

    r_hat =  np.dot(np.dot(v_u, M_sum), v_m.T) + b_o*np.ones((U,M)) + np.matlib.repmat(b_u,1,M) + np.matlib.repmat(b_m.T,U,1)

    loss1 = epsilon*np.square(rating_matrix - r_hat)
    loss2 = np.multiply(Nums[:,:,0], np.log(1 + np.exp(-1*(c*r_hat - b)))) + np.multiply(Nums[:,:,1], np.log(1 + np.exp((c*r_hat - b))))
    
    loss3 = np.zeros((U,M))
    for i in range(A):
        #print np.diag(M_a[i])
        ruma = np.dot(np.dot(v_u, np.diag(M_a[i])), v_m.T) + b_o*np.ones((U,M)) + np.matlib.repmat(b_u,1,M) + np.matlib.repmat(b_m.T,U,1)
        loss3 = loss3 + np.multiply(Numas[:,:,i,0], np.log(1 + np.exp(-1*(c*ruma - b)))) + np.multiply(Numas[:,:,i,1], np.log(1 + np.exp((c*ruma - b))))

    theta_uma = np.exp(np.tile(theta_u.reshape(U,1,A), (1,M,1)) + np.tile(theta_u.reshape(1,M,A), (U,1,1)))
    loss4 = theta_uma / (theta_uma.sum())
    loss4 = (np.multiply(Numa, np.log(loss4))).sum(2)

    loss = loss1 + loss2 + loss3 - loss4
    loss = np.multiply(loss, (rating_matrix > 0))
    total_loss = loss.sum()

    return total_loss

def fprime(params, *args):
    y = args[0]
    z = args[1]
    s = args[2]
    Nums = args[3]
    Numas = args[4]
    Numa = args[5]

    v_u = params[:(U*K)].reshape((U,K), order='F')
    b_u = params[(U*K):(U*K + U)].reshape((U,1), order='F')
    theta_u = params[(U*K + U):(U*K + U + U*A)].reshape((U,A), order='F')

    v_m = params[(U*K + U + U*A):(U*K + U + U*A + M*K)].reshape((M,K), order='F')
    b_m = params[(U*K + U + U*A + M*K):(U*K + U + U*A + M*K + M)].reshape((M,1), order='F')
    theta_m = params[(U*K + U + U*A + M*K + M):(U*K + U + U*A + M*K + M + M*A)].reshape((M,A), order='F')

    M_a = params[(U*K + U + U*A + M*K + M + M*A):].reshape((A,K), order='F')

    M_sum = np.diag(M_a.sum(0))

    #partial derivatives of ruma
    grad_ruma_vu = np.zeros((U,M,A,K))
    grad_ruma_vm = np.zeros((M,U,A,K))
    for i in range(A):
        grad_ruma_vu[:,:,i,:] = np.tile(np.multiply(np.matlib.repmat(M_a[i],M,1), v_m).reshape(M,1,K), (U,1,1,1))
        grad_ruma_vm[:,:,i,:] = np.tile(np.multiply(np.matlib.repmat(M_a[i],U,1), v_u).reshape(U,1,K), (M,1,1,1))

    grad_ruma_bu = 1
    grad_ruma_bm = 1
    grad_ruma_thetaua = 0
    grad_ruma_thetama = 0
    grad_ruma_mak =    np.tile(np.multiply(np.matlib.repmat(M_a[i],U,1), v_u).reshape(U,1,K), (M,1,1,1))




#params = [v_u, b_u, theta_u, v_m, b_m, theta_m, M_a]
args = (y,z,s,Nums,Numas,Numa)
#initial_values = np.array([v_u, b_u, theta_u, v_m, b_m, theta_m, M_a], dtype=object)
initial_values = numpy.concatenate((v_u.flatten('F'), b_u.flatten('F'), theta_u.flatten('F'), v_m.flatten('F'), b_m.flatten('F'), theta_m.flatten('F'), M_a.flatten('F')))

#print func(initial_values, *args)

x,f,d = fmin_l_bfgs_b(func, x0=initial_values, args=args, approx_grad=True, maxfun=1, maxiter=1)

print x
print f
print d









