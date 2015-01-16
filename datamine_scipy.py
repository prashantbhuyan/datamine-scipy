__author__ = 'Prashant B. Bhuyan'

import timeit

setup = '''

import math
import sys
import numpy as np
import scipy
from scipy.optimize import curve_fit
import pandas as pd
import copy
import random

# This function converts elements of an input list to type float.
def convert(arr):
    floatConv = []
    for i in arr[1:]:
        floatConv.append(float(i))
    return floatConv

# def convert(x):
    # converted = []
    # for i in x:
        # converted.append(i)
    # return converted

# def targetFunc(x,m,b):
    # return m*x + b

def regress_man(arr1,arr2):

    # Error Handling.
    if len(arr1) != len(arr2):
        print("Cannot Operate on Different Size Arrays . . . ")
        return

    arr_length = len(arr1)

    # Store converted values to new arrays.
    floatList1 = []
    floatList2 = []

    # Initialize variables that will store sums.
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0

    # Populate float arrays by calling conversion function above.
    floatList1 = convert(arr1)
    floatList2 = convert(arr2)

    # Iterate over tuples and apply get the values needed for the determinant equation.
    for i,j in map(None,floatList1,floatList2):
        sum_x += i
        sum_y += j
        sum_xx += i**2
        sum_yy += j**2
        sum_xy += i*j

    # Compute the determinant.
    d = (arr_length*sum_xx - sum_x*sum_x)

    # print "Determinant Value: ", d
    # Compute the slope.
    m = (sum_xy * arr_length - sum_y * sum_x)/d

    # print "Slope Value: ", m

    # Compute the y - intercept.
    b = (sum_xx * sum_y - sum_x * sum_xy)/d

    # print "Y Intercept Value: ", b

    # In Slope Intercept Form . . .
    # print("")
    # print "In Form y = mx + b:"
    # print "y = " , m,"x +",b

    return m,b


def regress(x,y):


    return np.polyfit(x,y,1)


##### Non Linear Curve Fit (Gaussian) #####

# def func(x,l,m,n):
    # return l*np.exp(-(x-m)**2/(2*n**2))


# def nonlin():
#   x = np.linspace(0,5,500)
#   y = func(x,1,7,9)
#   z = y + .2 * np.random.normal(size = len(x))
#   popt,pcov = scipy.optimize.curve_fit(func,x, z)
#   return popt



brain = []
body = []
file = '/Users/MicrostrRes/Desktop/bb.csv'
try:
    f = pd.read_csv(file)


except:
    print "File Error . . . "
    return

df = pd.DataFrame(f)
brain = df['brain']
body = df['body']

# w/ numpy regression
print regress(body,brain)









'''

# print regress(body,brain)




n = 10000



print "Regression Function W/O Numpy (Slope, Intercept, Runtime (Seconds): "
t = timeit.Timer("x = copy.copy(brain); y = copy.copy(body); regress_man(y,x)", setup = setup)
print t.timeit(n), "seconds for", n, "loops."
print ""

print "Regression Function W/ Numpy (Slope, Intercept, Runtime (Seconds): "
t = timeit.Timer("x = copy.copy(brain); y = copy.copy(body); regress(x,y)",setup = setup)
print t.timeit(n), "seconds for", n, "loops."
print ""

