#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:41:43 2024

@author: UCLA
but toms code
"""

from scipy.constants import mu_0
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt


def field_on_axis(R, I, N, Z):
    """
    Calculate the magnetic field at Z on axis of a loop with radius R with current I, turns N.
    Args
    -----------
        R: Radius of loop in meters
        I: Current through loop in Amperes
        N: Number of turns in the loop
        Z: Distance from loop in meters

    Returns
    -----------
        B: Magnetic field at Z in Tesla
    """
    B = mu_0 * N * I * R**2 / (2 * (R**2 + Z**2)**(3/2))
    return B
    
def calc_coeff(W, C):
    return 1*C[0] + C[1]*(W/(1-W)) + C[2]*(np.log(1-W)/W + 1) + C[3] * np.log(1-W) + C[4] * W + C[5] * W**2
    
def field_off_axis(a, I, N, z, h):
    W = (4 * h**2 * a**2)/(h**2 + a**2 + z**2)**2
    AR_c = [3, 3.60126526462842428, 0.727021564606502388, -0.0522255035327797, -0.008695310845331860, -0.0014574683941872]
    AZ_c = [3, 3.6012649981533775, 0.7276276696916009, 0.0527599282256922, -0.00892318142802500, -0.0015500658123898]
    BZ_c = [1, 0.9003162495383443, -0.0304443563318930, -0.025946696478408, -0.003959589609185077, -0.0007177536438]
    AR = calc_coeff(W, AR_c)
    AZ = calc_coeff(W, AZ_c)
    BZ = calc_coeff(W, BZ_c)
    Br = (mu_0 * I * N * z * h * a**2) * AR / (4*(h**2 +a**2+z**2)**(5/2))
    Bz = (mu_0 * I * N) / ( 4 * np.sqrt(h**2 + a**2 + z**2)) * (((2 * a**2 * BZ)/(h**2 + a**2 + z**2))-(W*AZ/4))
    return Br, Bz

def calculate_field(mag_array, z_array, r_array):
    bfield = np.empty([len(mag_array), z_array.shape[0], r_array.shape[0], 2])
    for i, mag in enumerate(mag_array):
        for j, z in enumerate(z_array):
            for k, r in enumerate(r_array):
                bfield[i, j, k ] = field_off_axis(mag.radius, mag.power_supply, mag.turns, mag.position-z, r)            
    return np.sum(bfield, axis = 0)

def calculate_field_axis(mag_array, z_array):
    bfield = np.empty([len(mag_array), z_array.shape[0]])
    for i, mag in enumerate(mag_array):
        for j, z in enumerate(z_array):
            bfield[i, j] = field_on_axis(mag.radius, mag.power_supply, mag.turns, mag.position - z)
    return np.sum(bfield, axis = 0)


machine_z = np.linspace(-.5, 2.45, 601)
# machine_r = np.linspace(0.001, .38, 101)
machine_r = np.array([0.26])
Magnet = namedtuple('Magnet', ['radius', 'turns', 'position', 'power_supply'])

current0 = 41
current1 = 29.5
current2 = 75
current3 = 65

# 90G field
# current0 = 46.4
# current1 = 39.4
# current2 = 99.3
# current3 = 67

# 70G field
current0 = 41
current1 = 29.5
current2 = 75
current3 = 95

#gradient down, config 1
#current0 = 51
#current1 = 26
#current2 = 39
#current3 = 10

#magnetic mirror, config 2
#current0 = 57
#current1 = 10
#current2 = 90
#current3 = 0

#steep gradient/step function, config 3
#current0 = 25
#current1 = 38
#current2 = 0
#current3 = 0
labels = [('Constant 70G'), ('Magnetic Mirror'), ('Steep Gradient')]
fig, ax = plt.subplots(figsize = (8, 6))

current0 = [41, 57, 25]
current1 = [29.5, 10, 38]
current2 = [75, 90, 0]
current3 = [95, 0, 0]


for i in range(3):

    turns1 = 38
    turns2 = 15

    radius1 = .403
    radius2 = .403

    mag_pos = [(-.315-.21)/2,
            (-.21-.095)/2,
            (.085+.19)/2,
            (.205+.31)/2,
            (.47+.575)/2,
            (.59+.695)/2,
            (.89+.995)/2,
            (1.0+1.105)/2,
            (1.28+1.385)/2,
            (1.39+1.495)/2,
            (1.665+1.77)/2,
            (1.775+1.88)/2,
            (2.145+2.25)/2]

    # original magnetic configuration
    mag1 = Magnet(radius1, turns1, (-.315-.21)/2, current1[i])
    mag2 = Magnet(radius1, turns1, (-.21-.095)/2, current1[i])
    mag3 = Magnet(radius1, turns1, (.085+.19)/2, current1[i])
    mag4 = Magnet(radius1, turns1, (.205+.31)/2, current1[i])
    mag5 = Magnet(radius1, turns1, (.47+.575)/2, current1[i])
    mag6 = Magnet(radius1, turns1, (.59+.695)/2, current1[i])
    mag7 = Magnet(radius2, turns2, (.89+.995)/2, current2[i])
    mag8 = Magnet(radius2, turns2, (1.0+1.105)/2, current2[i])
    mag9 = Magnet(radius2, turns2, (1.28+1.385)/2, current2[i])
    mag10 = Magnet(radius2, turns2, (1.39+1.495)/2, current2[i])
    mag11 = Magnet(radius2, turns2, (1.665+1.77)/2, current2[i])
    mag12 = Magnet(radius2, turns2, (1.775+1.88)/2, current2[i])
    mag13 = Magnet(radius2, turns1, (2.145+2.25)/2, current3[i])

    mag1 = Magnet(radius1, turns1, (-.315-.21)/2, current0[i])
    mag2 = Magnet(radius1, turns1, (-.21-.095)/2, current0[i])
    mag3 = Magnet(radius1, turns1, (.085+.19)/2, current1[i])
    mag4 = Magnet(radius1, turns1, (.205+.31)/2, current1[i])
    mag5 = Magnet(radius1, turns1, (.47+.575)/2, current1[i])
    mag6 = Magnet(radius1, turns1, (.59+.695)/2, current1[i])
    mag7 = Magnet(radius2, turns2, (.89+.995)/2, current2[i])
    mag8 = Magnet(radius2, turns2, (1.0+1.105)/2, current2[i])
    mag9 = Magnet(radius2, turns2, (1.28+1.385)/2, current2[i])
    mag10 = Magnet(radius2, turns2, (1.39+1.495)/2, current2[i])
    mag11 = Magnet(radius2, turns2, (1.665+1.77)/2, current2[i])
    mag12 = Magnet(radius2, turns2, (1.775+1.88)/2, current2[i])
    mag13 = Magnet(radius2, turns1, (2.145+2.25)/2, current3[i])

    mags = [mag1, mag2, mag3, mag4, mag5, mag6, mag7, mag8, mag9, mag10, mag11, mag12, mag13]



    plt.setp(ax.spines.values(), linewidth=2)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    ax.tick_params(which='minor', top = True, direction = 'in', length = 6, width = 2, right = True, labelsize = 18)
    ax.tick_params(which='major', top = True, direction = 'in', length = 12, width = 2, right = True, labelsize = 18)

    Bz_off_axis = calculate_field(mags, machine_z, machine_r)*10000
    Bz_on_axis = calculate_field_axis(mags, machine_z)*10000

    #plt.plot(machine_z, Bz_off_axis[:,0,1])
    plt.plot(machine_z, Bz_on_axis, linewidth = 4, label = labels[i])
    plt.vlines(mag_pos, 0, 105, color = 'Black')
    #plt.hlines(70,0,2)
    plt.vlines(0.41, 0, 105, color='red', linestyle='dashed', linewidth = 3)
    plt.vlines(0.81, 0, 105, color='red', linestyle='dashed', linewidth = 3)
    plt.ylabel('YBo (Gauss)', fontsize = 25)
    #plt.legend(loc = 'upper right', fontsize = 25)
    plt.xlabel('Machine Z (m)', fontsize = 25)
    plt.legend(loc = 'upper right', fontsize = 20)




plt.savefig('Ball.pdf')
plt.show()


