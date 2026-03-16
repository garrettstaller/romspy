import os
# User Inputs  ![ 

# soon-to-be CL Prompts
project_path = os.getcwd()
variables=['temp']
plot_type = 'sliced'
plot_param = 'top'
vtype = 'his' # the type of ROMS output (his, avg, rst, depths, etc!)

# output
snapshot_time  = -1    # -1 for final timestep
load_animation = True  # overridden for hovmoller plots (no time to animate over...)
output_name = 'test'

# specifics
grd_path  = os.getcwd()
kelp_path = None
plot_kelp = False # only takes effect for surface plots and cross-sections that intersect kelp
plot_sponge = True
verbose = True   # print status in function calls
# directory names to exclude when looking for files to load
exclude = ['input_data','short']

# Plotting
# data rep.
# fill in variables that you want specfic ranges for!
# if you chose to place None and None, then ranges are based on min and max
data_cranges_overwrite = {'temp':  [10.5, 14.5],
                          'vort':  [-3,3],
                          'u':     [-.15, .15],
                          'cross': [-.15, .15],
                          'v':     [-.30, .30],
                          'along': [-.15, .15],
                          'n2':    [1e-4,1e-8],
                          'eke':   [0,1e-2],
                          'cross_hf':[-.03,.03]
                                               }
cmap = None
# parameters
fs_header = 14
fs_title = 12
fs_text  = 12
fs_label = 10
fs_tick  = 10
# axes
xlab  = None
ylab  = None
# ticks
ntick = None
# contours  (labels for bathymetry if horizontal plot!)
contour_labels = True

#              !]

# Initialization ![
print('Loading Modules')
# system
import sys
import glob
sys.path.append('/home/gstaller/repos/roms-py/')
# data
import xarray as xr
import xroms
import numpy as np
# plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
# misc
import code
from tqdm import tqdm
import time as clock
# roms-py
ROMSPY_PATH = os.environ['ROMSPY_ROOT']
sys.path.append(ROMSPY_PATH)
import nc_read_write as nc
import calculate
calc = calculate.Diagnostics()
import plot_utils as pu
import tools as rt

# init values
read_inputs = {}
read_inputs[plot_type] = plot_param
read_inputs['verbose'] = verbose
read_inputs['to_numpy'] = False
read_inputs['interpolate'] = True

load_inputs = {}
load_inputs['filetype'] = vtype

#                !]

# File Parser  ![

# Get all simulation types for project:
simulation_directories = []
for entry in os.scandir(project_path):
  if (entry.is_dir() and entry.name not in exclude):  # we want to ignore directories in projects that are used for other files
    simulation_directories.append(entry.name)
# sort them (these are essentially the base identifiers of diff. simulations)
file_types = np.sort(simulation_directories)

# if there are data directories we are in a parent directory, we are then comparing projects
# OR
# we are in a specific directory and are plotting multiple vars
if ((len(file_types)!=0) or (len(file_types)==0 and len(variables)>1)):
  comp_param = True
  if len(file_types)!=0:
    call_inputs = True
  else:
    print(f'Working in {project_path}')
    call_inputs = False
# Otherwise just plot one variable in one plot
else:
  comp_param  = False
  call_inputs = False
  print(f'Working in {project_path}')

# select parameters
if call_inputs:
  # give user a list of data directories to select from for plotting
  for i,val in enumerate(file_types):
    print(f'{i}: {val}')
  print('Select indices of parameters to compare', '\n', 'ENTER q WHEN READY')
  n = 1
  parameters = {}
  # multiple parameters in x and y
  if (len(variables)==1):
    while True:
      # get user input
      user_input = input(f"Param {n} (space separated): ")
      if user_input == 'q':
        break
      else:
        user_input = np.fromstring(user_input,dtype=int,sep=' ')
      if ((n==2) and (len(user_input)!=len(parameters[f'param{n-1}']))):
        raise SystemExit("Selected number of indices MUST be consistent. Exiting.")
      parameters[f'param{n}'] = user_input
      n += 1 
  # multi variables (coloumns) per parameter (y)
  else:
    while True:
      # get user input
      user_input = input(f"Param {n} (pick one simulation per input): ")
      if user_input == 'q':
        break
      else:
        user_input = np.fromstring(user_input,dtype=int,sep=' ')
      if (len(user_input)!=1):
        raise SystemExit("When doing multi varibles, pick one simulation per input. Exiting.")
      parameters[f'param{n}'] = user_input
      n += 1
  # get number of rows and coloumns based on user input of comparisons
  if len(variables)==1:
    default_x = len(parameters['param1'])
    default_y = len(parameters)
    #if default_x>=default_y:
    ncols = default_x
    nrows = default_y
    #else:
     # ncols = default_y
     # nrows = default_x
    # and get single variable for loading
    var = variables[0]
  elif len(variables)!=1:
    nrows = len(parameters)
    ncols = len(variables)
# we are not comparing params, check that requested files are accessible
else:
  if glob.glob(project_path+f'/*{vtype}*.nc'):
    print(f'{vtype} files found...')
    # set number of rows to simply number of vars
    ncols = len(variables)
    nrows = 1
  else:
    raise ValueError('Could not find .{vtype} files in {project_path}') 


#              !]

# Loading Data  ![
if plot_kelp and (plot_type=='sliced' or plot_type=='cross'):
  try:
    kelp_ds = nc.load(kelp_path,filetype='mag')
    b2d = nc.read(kelp_ds,'algae_b2d',to_numpy=True)[::,::,0]
    mask_kelp = np.zeros(b2d.shape)
    mask_kelp[b2d>0] = 1.
  except:
    raise ValueError(f'Error loading in kelp file at: {kelp_path}')

if plot_sponge and (plot_type=='sliced' or (plot_type=='cross' and plot_param[1] is None)):
  check_sponge=True
else:
  check_sponge=False
    

print('Loading in grid dataset')
grd_ds = nc.load(grd_path,filetype='grd')
# mask 
mask = nc.read(grd_ds,'mask_rho',to_numpy=True)[0]
h    = nc.read(grd_ds,'h',to_numpy=True)[0]
pm   = nc.read(grd_ds,'pm',to_numpy=True)[0]

print('Loading in all datasets')
all_sets = {}
n = 1
if (call_inputs):
  # Single Variable Loading - for each row and coloumn position, get new file
  if len(variables)==1:
    for i in range(nrows):
      row_indices = parameters[f'param{i+1}']
      files_in_row = file_types[row_indices]
      for j in range(ncols):
        # now we can select our dataset
        ds = nc.load(project_path+'/'+files_in_row[j],**load_inputs)
        all_sets[n] = {'dataset': ds,
                       'simname': files_in_row[j]}
        n += 1
  # Multi Variable Loading - for each row get files
  elif len(variables)>1:
     for i in range(nrows):
      row_indices = parameters[f'param{i+1}']
      files_in_row = file_types[row_indices][0]
      # now we can select our dataset
      ds = nc.load(project_path+'/'+files_in_row,**load_inputs)
      all_sets[n] = {'dataset': ds,
                     'simname': files_in_row}
      n += 1
else:
  ds = nc.load(project_path,**load_inputs)
  all_sets[n] = {'dataset': ds,
                 'simname': project_path.split('/')[-1]}

all_data = {}
n = 1
# pick a dataset from those lazily loaded in above
for s in range(len(all_sets)):
  ds = all_sets[s+1]['dataset']

  # also get depths data for each dataset (simulation) if cross-section plot
  if plot_type=='cross' or (plot_type=='hov'):
      depths_path = project_path+'/'+all_sets[s+1]['simname']+'/*depths.nc' if call_inputs else project_path+'/*depths.nc'
      print(f'Grabbing depth data in {depths_path}')
      zr, zw = rt.get_depths(depths_path,snapshot_time,plot_type,plot_param)
      #zr = rt.get_depths(depths_path,snapshot_time,plot_type,plot_param)

  # check for any other speicifcs (kelp locations, masks, sponges etc.)
  if check_sponge:
    try:
      sponge_loc=(ds.attrs['sponge_size'])/(np.mean(pm)*1000)  # HARDLOCKED: physical dimensions conversion [km]
    except:
      print(f"Sponge attribute not found in {all_sets[s+1]['simname']}")
      check_sponge=False

  # for each dataset, run through requested variables
  print(f'Processing variables in {all_sets[s+1]['simname']}')
  for var in variables:
    # simply read in using ncread with specific plotting options depending on user input
    data = nc.read(ds,var,**read_inputs)
    if plot_type=='cross' or (plot_type=='hov'):
      # check vertical dim to grab correct depth array:
      if 's_rho' in ds[var].dims:
        depths = zr
      elif 's_w' in ds[var].dims:
        depths = zw
      elif 's_win' in ds[var].dims:
        depths = zw[1:-1]
      else:
        depths = zr[0]
      all_data[n] = {'timesteps': data,
                     'depths': depths,
                     'simname': all_sets[s+1]['simname'],
                     'var': var,
                     'sponge': None if check_sponge==False else sponge_loc}
    else:
      all_data[n] = {'timesteps': data,
                     'simname': all_sets[s+1]['simname'],
                     'var': var,
                     'sponge': None if check_sponge==False else sponge_loc}

    n += 1
  
#               !]

# Plotting Data  ![
print('\n','Beginning Plotting Sequence')

# get info on netcdf dimensions for plotting
nx,ny,nz,time = nc.info(ds)  # simply use last dataset

# arrays to plot against
X,Y = pu.get_arrays(nx,ny,nz,time,grd_ds,plot_type,plot_param)  # arrays to plot (X by Y)
x,y,z,t = pu.get_arrays(nx,ny,nz,time,grd_ds)                   # all respecitve dimension arrays

# based on dimension of plot generate a figure with proper aspect ratios
fig, axes = pu.get_fig(nrows,ncols,X,Y)

plt.ion()

# initial image (saved as quadmesh)
ims = []

for i, ax in enumerate(axes):

  # get data and sim info for plot
  if plot_type!='hov':
    z = all_data[i+1]['timesteps'].isel(time=snapshot_time)
  else:
    z = all_data[i+1]['timesteps']

  # get varname
  vname = all_data[i+1]['var']

  # name subplots after simulation they come from
  subtitle = f'Sim: {all_data[i+1]['simname']}'

  # Plot data based on plot type and set titles based on that.
  if plot_type=='sliced':
    im = ax.pcolormesh(X,Y,z)
    header = f'Horizontal Slice of {plot_param}' 
    xlab = xlab if xlab is not None else 'Distance [km]'
    ylab = ylab if ylab is not None else 'Distance [km]'
    print('Created Horizontally Sliced Plot')
  elif plot_type=='cross':
    depths = all_data[i+1]['depths']
    #im = ax.pcolormesh(X,depths[0:nz:,:],z)
    im = ax.pcolormesh(X,depths,z)
    header = f"Y={y[plot_param[0]]}" if plot_param[0] is not None else f"X={x[plot_param[1]]}"
    xlab = xlab if xlab is not None else f"{'Cross-Shore' if plot_param[0] is not None else 'Along-Shore'} Distance [km]"
    ylab = ylab if ylab is not None else 'Depth [m]'
    print('Created Cross-Section Plot')
  elif plot_type=='hov':
    depths= all_data[i+1]['depths']
    if plot_param[0] is None:
      im = ax.pcolormesh(X,depths,z.T)
      header = f'Depth Profile with Time at X={x[plot_param[2]]} and Y={y[plot_param[1]]}'
    elif plot_param[1] is None:
      im = ax.pcolormesh(X,Y,z.T)
      header=f'Y with Time at X={x[plot_param[2]]} and Depth={0 if plot_param[0]=='top' else depths[plot_param[0]]}'
    else:
      im = ax.pcolormesh(X,Y,z)
      header=f'X with Time at Y={y[plot_param[1]]} and Depth={0 if plot_param[0]=='top' else depths[plot_param[0]]}'
    print('Created Hovmoller Plot')

  # Add titles, labels and colorbars
  pu.set_axes(ax,xlab,ylab,
              title=subtitle,
              title_fontsize=fs_title,label_fontsize=fs_label,total_ticks=ntick,tick_fontsize=fs_tick)
  # check no cmin or cmax values for var
  if vname in data_cranges_overwrite.keys():
    cmin,cmax = data_cranges_overwrite[vname][0], data_cranges_overwrite[vname][1]
  else:
    cmin,cmax = None, None
  cbar=pu.set_colorbar(im,ax,vname,z,
                       cmin=cmin,cmax=cmax,cmap=cmap,cfnt=fs_tick)

  # Masks and Contours
  if plot_type=='sliced':
    # land mask
    pu.add_mask(ax,X,Y,mask)
    # bathymetry contours
    pu.add_cons(ax,X,Y,h,limit=[10,125],ncontours=7,clabels=contour_labels)
    # use contours to outline kelp forest
    if plot_kelp:
      pu.add_cons(ax,X,Y,mask_kelp,limit=[0,1],ncontours=1,lw=1.0,cb='darkgreen')
  elif plot_type=='cross':
    # hacky depths mask
    ax.fill_between(X,depths[0],y2=np.min(depths),color='gainsboro')
    # cross sections of kelp forests if applicable
    if plot_kelp:
      # based on cross-section type search along slice for kelp using kelp mask
      if plot_param[1] is None:
        mask_slice = plot_param[0]
        kelp_cross_indices = np.where(mask_kelp[mask_slice]!=0)[0]   # slice along Y
      else:
        mask_slice = plot_param[1]
        kelp_cross_indices = np.where(mask_kelp[:,mask_slice]!=0)[0] # slice along x
      # plot a vertical line to delineate kelp forest bounds
      ax.axvline(X[kelp_cross_indices[0]],alpha=.5,color='darkgreen')
      ax.axvline(X[kelp_cross_indices[-1]],alpha=.5,color='darkgreen')
      # fill any gaps in between
      gaps = np.where(np.diff(kelp_cross_indices) > 1)[0]
      for ind in gaps:
       ax.axvline(X[kelp_cross_indices[ind]],alpha=.5,color='darkgreen') 
       ax.axvline(X[kelp_cross_indices[ind+1]],alpha=.5,color='darkgreen') 

#  elif plot_type=='hov' and plot_param[0] is None:
#    ax.axvline(10,color='magenta')

  # extra (misc.)
  if plot_sponge and check_sponge:
    ax.axvline(all_data[i+1]['sponge'],color='magenta',linestyle='--')

  # append to net figure
  ims.append(im)

# add title with time of sim
if plot_type!='hov':
  title = fig.suptitle(f'{header} | Day: {((time[snapshot_time])/(86400)):.2f}',fontsize=fs_header)
else:
  title = fig.suptitle(f'{header}',fontsize=fs_header)

plt.show(block=False)

#               !]
      
if load_animation and plot_type!='hov':
# Loading Animation  ![

  # give user time to make edits
  print("👀 Adjust the figure as needed (use fig.canvas.draw_idle() to update), then use 'ctrl+d' to start rendering animation...", \
        '\n','\n','Opening Interactive Shell:')
  code.interact(local=locals())
  print("Continuing...", '\n')
  
  # must import new backend for use of ffmpeg
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import matplotlib.animation as animation
  
  writer = animation.FFMpegWriter(fps=20)
  
  # Get all data for animation loop
  print('Writing to disk')
  for k in all_data:
      if verbose:
        start_time=clock.time()
      all_data[k]['timesteps_values'] = all_data[k]['timesteps'].values
      if verbose:
        print(f'Loaded {all_data[k]['var']} in {clock.time()-start_time:.2f} sec.')
  
  with writer.saving(fig, f'{output_name}.mp4', dpi=150):
    for t in tqdm(range(len(time)),desc='Rendering Animation'):
      for k, im in enumerate(ims):
        ax = im.axes
        z_vals = all_data[k+1]['timesteps_values']
        im.set_array(z_vals[t])
        title.set_text(f'{header} | Day: {t/24:.2f}')
      writer.grab_frame()


#                    !]


