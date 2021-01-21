import numpy as np

def sag_from_length_and_radius(length, r):
    return r - np.sqrt(r**2 - (length/2)**2)

def radius_from_length_and_sag(length, sag):
    return (sag**2+(length/2)**2)/(2*sag)

def arc_angle_from_length_and_radius(length, r):
    sag = sag_from_length_and_radius(length, r)
    return arc_angle_from_length_radius_and_sag(length,r,sag)

def arc_angle_from_length_radius_and_sag(total_length,r,sag):
    return 2*np.arctan(0.5*total_length/(r-sag))    

def length_from_radius_and_arc_angle(r, theta):
    return np.sqrt(1-np.cos(theta/2)**2)*2*r