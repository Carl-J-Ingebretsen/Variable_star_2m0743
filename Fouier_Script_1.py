''' This script runs the fourier analysis of pulsating star light curve data. It was written
for ASTR 302 in april of 2021. Authors: Aiden Nakleh, Carl Ingebretsen, Cyrus Worley, Patrick Flint'''

import numpy as np 
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d 
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt 
import pandas as pd 

def main():
    '''main function'''
    star_data = import_star_data() #import the data from a csv file
    display_graph(star_data) #Display the light curve
    no_baseline_data, time = subtract_a_polynomial(star_data) #Subtract a baseline off the data
    intr_data, new_time = interpolate_data(no_baseline_data, time)#Interpolate the data
    transformed_data, xf = fourier_analyze_data(intr_data, new_time)#Do the fourier transform
    peaks = find_data_peaks(transformed_data, xf) #Find the maximums of the peaks
    #characterize_peaks(peaks,transformed_data,xf)
    test_fit_curve(peaks,transformed_data,xf)

def import_star_data():
    '''import the data'''
    file_name=input("enter a file name: ")
    data=pd.read_csv(file_name)
    data=pd.DataFrame(data, columns=['J.D.-2400000','rel_flux_T1'])#Be sure of which columns.
    #One time column and one flux column.
    data=data.to_numpy()
    #print(data)
    return data

def subtract_a_polynomial(data):
    '''remove the outer shape'''
    time=[]
    flux=[]
    for point in data:
        time.append(point[0])
        flux.append(point[1])
    curve=np.polyfit(time,flux,deg=2)
    #print(time)

    a = curve[0]
    b = curve[1]
    c = curve[2]
    polynomial_values = []
    fitted_flux = []
    for i in range(len(time)):
        num = a*(time[i])**2 + b*(time[i]) + c
        polynomial_values.append(num)
        fitted_flux.append(flux[i]-num)
       
    #print(flux)
    #print(fitted_flux)
    plt.plot(time,fitted_flux)
    plt.xlabel('JD-2400000')
    plt.ylabel('relative flux')
    plt.show()
    return fitted_flux, time


def interpolate_data(flux_data, time):
    '''perform an interpolation'''
    new_time = np.linspace(time[0], time[-1], num=933, endpoint=True)
    interpolated_flux = interp1d(new_time, flux_data, kind='cubic')
    #print(interpolated_flux)
    #Plot of the new data.
    #plt.plot(new_time, interpolated_flux(new_time), marker = '.', linestyle='solid', color = 'blue', markersize = 0.5)
    #plt.show()
    return interpolated_flux, new_time

def fourier_analyze_data(flux_data, time):
    '''Fourier analyze the data'''
    transform_data = fft(flux_data(time))
    transform_data = abs(transform_data)
    #xf = fftfreq(933, 0.0003351502164150588) #step in number of days 28.957 in seconds
    xf = fftfreq(933, 28.957)
    plt.plot(xf, transform_data)
    plt.show()
    return transform_data ,xf

def find_data_peaks(flux_data, frequencies):
    '''Find the peaks at wich there is power in the fourier transform.'''
    peaks, peak_data = find_peaks(x=flux_data, threshold=0.04)#Check threshold to get all 5 peaks. 4 peaks at 0.05.
    #print(peaks)
    for i in peaks:
        print('a frequency of pulsation is', frequencies[i])
        print('the corresponding period in seconds is', (1/frequencies[i]))
        
    return peaks

def save_data_to_file():
    '''save the necessary data.'''

def display_graph(data):
    '''display a graph of a light curve.'''
    time = []
    flux = []

    for point in data:
        time.append(point[0])
        flux.append(point[1])

    plt.plot(time, flux, marker = '.', linestyle='solid', color = 'blue', markersize = 0.5)
    plt.xlabel('JD-2400000')
    plt.ylabel('relative flux')
    plt.show()

def display_graph_2(data):
    '''another way to display a graph'''
    plt.plot(data[0],data[1])
    plt.xlabel('JD-2400000')
    plt.ylabel('relative flux')
    plt.show()

#def guassian_model(x, sigma, mu):
    #'''Make a guassian function to use as a model.'''
    #return (1/(sigma*(2*np.pi)**0.5))*np.exp(-0.5*(x-mu/sigma)**2)

def fit_the_model(xx, yy):
    '''try to fit the model to get central frequency and to get an uncertainty.'''
    param, covarients = curve_fit(gaussian, xx, yy, method='lm')
    print(param)
    return param, covarients

def characterize_peaks(peaks,flux_data,frequencies):
    '''Slice and fit the peaks.'''
    peaks=peaks[3:]
    for i in peaks:
        print(i)
        
        t_1=flux_data[flux_data[(i-3)]:flux_data[(i+3)]]
        xf_1=xf[frequencies[(i-3)]:frequencies[(i+3)]]
        param, cov = fit_the_model(xf_1, t_1)
        print(param, cov)
        plt.plot(frequencies, flux_data)
        plt.plot(xf_1, guassian_model(xf_1, param[0], param[1]))
        plt.show()

def test_fit_curve(peaks, flux_data, frequencies):
    '''test function'''
    print(peaks[5])
    flux_1=flux_data[742:750]
    freq_1=frequencies[742:750]
    param, cov = fit_the_model(freq_1, flux_1)
    print(param, cov)
    #plt.plot(frequencies, flux_data)
    plt.plot(freq_1, gaussian(freq_1, param[0], param[1], param[2]))
    plt.show()

def gaussian(x,amp,cen,wid):
    return amp * np.exp(-(x-cen)**2 / wid)



main()