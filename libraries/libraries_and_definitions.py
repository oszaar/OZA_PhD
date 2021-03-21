############################################# import libraries ###############################
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import (LinearSegmentedColormap,ListedColormap)
import matplotlib.ticker as mtick
from matplotlib.patches import Ellipse
from matplotlib.text import OffsetFrom
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
import matplotlib.ticker 
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator,LogLocator)
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.collections import LineCollection
from matplotlib import cm
import matplotlib.font_manager as mpfm


############################################ 
import pandas as pd
import glob
import os
from os import listdir
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy.polynomial.polynomial as poly

########################################### for image composition
import svgutils.compose as sc
from IPython.display import SVG
import matplotlib.image as imgs 
from PIL import Image 

### scypy libraries
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import optimize
from scipy.interpolate import *
from scipy.signal import savgol_filter
from scipy import integrate
import scipy.interpolate as si
# from scipy.interpolate import BSpline
import scipy.interpolate as interpolate
from scipy import constants as cnts

### lmfit libraries, used for curve and peak fitting 
import lmfit 
import asteval
from lmfit import *
from lmfit import models
from lmfit.model import load_model
from lmfit.model import save_modelresult
from lmfit.models import (ExpressionModel, GaussianModel, ExponentialModel,PowerLawModel,
                          StepModel,LinearModel) 

###################### other libraries
from brukeropusreader import read_file  #opus_data = read_file('opus_file.0')
import ntpath
import argparse
import sys
import statsmodels as sm
from patsy import dmatrices
from BaselineRemoval import BaselineRemoval

# pio.renderers.default ='firefox'

##################### Some handy scientific constants 
kbe = cnts.physical_constants['Boltzmann constant in eV/K'] #'in eV k^-1'
teu = kbe[0]*1E6 # natural unit of thermopower
pi =  cnts.pi
e = cnts.physical_constants['elementary charge'] # in C

############################################# plotting style ###################################

plt.style.use('seaborn-paper')
plt.rc('font', family='serif')
# plt.rcParams['figure.dpi']= 100

nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8.2,
        "font.size": 8.2,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'ytick.minor.left': True,
        'xtick.minor.bottom': True,
        'ytick.direction': 'in',
        'xtick.direction': 'in',
}
plt.rcParams.update(nice_fonts)

'''
--------------------
This creates a set of definitions for fancy annotations
--------------------
'''


########### annotations and labels 
def annotate(ax, x, y, text, fc="#ff7777", y0=0):      ##### label inside a frame with a guide dotted guide to the eye
    y = y - 0.5
    ax.annotate(
        " "+text+" ", xy=(x,y), xycoords='data',
        xytext=(0, 12), textcoords='offset points',
        color="white",
#         size="x-small",
        va="center", ha="center", weight="bold",
        bbox=dict(boxstyle="round", fc=fc, ec="none"),
        arrowprops = dict(arrowstyle="wedge,tail_width=1.",
                          fc=fc, ec="none", patchA=None))
    plt.plot([x,x], [y,y0], color="black", linestyle=":", linewidth=.75)
    
    
def annotate0(ax, x, y, text, fc="#ff7777", y0=0):      ##### label inside a frame with a guide dotted guide to the eye
    y = y - 0.5
    ax.annotate(
        " "+text+" ", xy=(x,y), xycoords='data',
        xytext=(0, 12), textcoords='offset points',
        color="white",
#         size="x-small",
        va="center", ha="center", weight="bold",
        bbox=dict(boxstyle="round", fc=fc, ec="none"),
        arrowprops = dict(arrowstyle="wedge,tail_width=1.",
                          fc=fc, ec="none", patchA=None),rotate=180)
    plt.plot([x,x], [y,y0], color="black", linestyle=":", linewidth=.75)
    
    
def annotate2(ax, x, y, text, fc="#ff7777"):      ##### label inside a frame with a guide dotted guide to the eye
    y = y - 0.5
    ax.annotate(
        " "+text+" ", xy=(x,y), xycoords='axes fraction',
        xytext=(0, 12), textcoords='offset points',
        color="k",
#         size="x-small",
        va="center", ha="center", weight="bold",
        bbox=dict(boxstyle="round", fc=fc, ec="dimgray", alpha=0.8),fontsize=10.5
#         arrowprops = dict(arrowstyle="wedge,tail_width=1.",
#                           fc=fc, ec="none", patchA=None)
    )
#     plt.plot([x,x], [y,y0], color="black", linestyle=":", linewidth=.75)


'''
definitions for plotting with seaborn

'''
def plot_absorbance_1(data, labels):
    # figure frame and subplots
    fig = plt.figure(figsize=(5, 1.8))
    ax = fig.add_subplot(1, 1, 1,)
    # Plot the bars
    ax.plot(data, lw=1, ls='-')
    ax.set_ylim(0, 1)
    ax.set_xlim(385,2500)
    # Apply labels to the bars so you know which is which
    ax.legend(labels=labels, loc='upper right', bbox_to_anchor=(1, 1))

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Absorbance")
    ax.set_xlim(385, 1110)
    ax.set_ylim(0, 0.9)
    ax.xaxis.set_minor_locator(MultipleLocator(125))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    return fig, ax


def plot_absorbance_2(data,labels):
    # Create the figure and axis objects I'll be plotting on fig, ax = plt.subplots()
    fig = plt.figure(figsize=(5,1.8))
    ax = fig.add_subplot(1, 1, 1,)# 
    # Plot the bars
    ax.plot(data, lw=1, ls='-')

    # Set a reasonable y-axis limit
    ax.set_ylim(0, 1)
    ax.set_xlim(385,2500)
    
    # Apply labels to the bars so you know which is which
    ax.legend(labels=labels,loc='upper right',bbox_to_anchor=(1, 1))
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Absorbance")
    ax.set_xlim(385,2500)
    ax.set_ylim(0,0.9);
    ax.xaxis.set_minor_locator(MultipleLocator(125))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    return fig, ax







    
    
'''
this section creates a function for the power law functions 

'''

def ulaw_S_Glaudell(x):
    return teu*((x)/1)**(-1/4)  #Glaudell universal power law for thermoelectrics

def ulaw_PF_Glaudell(x):
    return teu*((x)/0.5)**(1/2)  #Glaudell universal power law for thermoelectrics

def maxzt(sigma, zt, kl): #sigma in siemens per cm and kappa in wmk
    T=300
    kl=0.35
    sigma=sigma*100
    Lorenz=2.44E-8
    ke=sigma*Lorenz*T
    kappa=(ke+kl)
    return np.sqrt((kappa*zt)/(sigma*300))*1000000  # contour lines for max ZT in a log log scale of S vs sigma 