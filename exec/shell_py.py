# system
import os
import sys
import glob
# data
import xarray as xr
import xroms
import numpy as np
# plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import code
from tqdm import tqdm
import time

ROMSPY_PATH = os.environ['ROMSPY_ROOT']
sys.path.append(ROMSPY_PATH)
import nc_read_write as nc
import calculate
calc = calculate.Diagnostics()
import plot_utils as pu
import tools as rt

project_path = os.getcwd()

all_ds = nc.load(project_path)
grd_ds = nc.load(project_path,'grd')

