#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:37:58 2019

@author: william
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#K = np.linspace(-10,10,1000)

a =  1
b =  4
c =  3
d = -10

j = 0
X_tuple = np.zeros(3)
while j < 10000:
    K = -500+0.1*j
    
    d =K
    
    coff = [a,b,c,d]
    x_ = np.roots(coff)
    '''
    x = x_[0]
    print ("should be zero: ",a*x**3+b*x**2+c*x+d)
    x = x_[1]
    print ("should be zero: ",a*x**3+b*x**2+c*x+d)
    x = x_[2]
    print ("should be zero: ",a*x**3+b*x**2+c*x+d)
    '''
    X_tuple = np.vstack((X_tuple,x_))    
    j=j+1
X_tuple = np.delete(X_tuple, 0, 0)

    


x1 = X_tuple[:,0]
x2 = X_tuple[:,1]
x3 = X_tuple[:,2]

x2_0 = np.interp(0,x2.real,x2.imag)
print ("x2_0",x2_0)
x1_0 = np.interp(0,x1.real,x1.imag)
print ("x1_0",x1_0)

plt.plot(x1.real,x1.imag,'g')
plt.plot(x2.real,x2.imag,'m')
plt.plot(x3.real,x3.imag,'r')
plt.vlines(0,-8,8, colors='k',linestyles = 'solid')

plt.savefig('Contril CW',dpi = 300)
plt.show()




