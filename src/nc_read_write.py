'''
nc_read_write.py - Garrett S. [Version 1 - January 28, 2026]

Purpose:
Python analog to ROMS nc_read_write module. Frequently called functions for
obtaining netcdf data. 

Description:
When properly imported into a python script, users may call for data of a 
variable that is first checked for dimensionality and then availibility.

Architecture:
  1. Loading
     - get_grd 
     - load
  2. Reading
     - read
     - slicer
  3. Coordinate Transforms
     - get_var
     - u2rho(2d, 3d or 4d)
     - v2rho(2d, 3d or 4d)

'''
import os
import glob
import xarray as xr
import numpy as np
ROMSPY_PATH = os.environ['ROMSPY_ROOT']
sys.path.append(ROMSPY_PATH)
import calculate 

# ---- 1. Loading in NetCDFs    ---- #

# get_grd  ![
def get_grd(path,info=True):

  '''
  All plotting scripts and most diagnostics need information on the grd. Given a path to a parent directory 
  this will retrieve the grd used in simulations OR simply direct to a specific grd file.
  Returns the raw dataset and a dictionary of neccessary info (mask, pm, pn, f) - This can be turned off...
  '''

  # simply import grid dataset
  if 'grd.nc' in path:
    grd_ds = xr.open_mfdataset(path)
    if info:
      # arrays
      mask_rho = read(grd_ds,'mask_rho',to_numpy=True)
      pm = read(grd_ds,'pm',to_numpy=True)
      pn = read(grd_ds,'pn',to_numpy=True)
      f  = read(grd_ds,'f',to_numpy=True)
      # add all to dict for return
      grd_details_dict = {'mask': mask_rho, 'pm': pm, 'pn': pn, 'f': f}
      return grd_ds, grd_details_dict
    else:
      return grd_ds

  # if the user simply provides simulation directory, recursively search through each simulation
  # directory to find a grd file!
  else:
    simulation_directories = []
    for entry in os.scandir(path):
      if entry.is_dir():
        simulation_directories.append(entry.name)
    # sort them (these are essentially the base identifiers of diff. simulations)
    simulations = np.sort(simulation_directories)
    
    grd_not_found = True
    n = 1
    while grd_not_found:
      try:
        grd_path = path+'/'+simulations[n]+'/'+'*grd*.nc'
        grd_ds = xr.open_mfdataset(grd_path)
        grd_not_found = False        
        if info:
          # arrays
          mask_rho = read(grd_ds,'mask_rho',to_numpy=True)
          pm = read(grd_ds,'pm',to_numpy=True)
          pn = read(grd_ds,'pn',to_numpy=True)
          f  = read(grd_ds,'f',to_numpy=True)
          # add all to dict for return
          grd_details_dict = {'mask': mask_rho, 'pm': pm, 'pn': pn, 'f': f}
          return grd_ds, grd_details_dict
        else:
          return grd_ds
      except:
        # if we have exhausted all simulation folders and no grd file is found, raise error
        if n==len(simulations)-1:
          raise ValueError('The provided project folder does not have a grid file, or the one provided has an invalid location')
      n += 1


#          !]

# load  ![
def load(project_path,    # path to project folder holding simulations
         filetype='his',  # filetype as str: his, avg, or rst
         verbose=False):  # turns on/off print statements

  '''
  Given a path to a directory of simiulation folders, user may select which simulation(s) and specific file type 
  to have accessible in script using xarray datasets. Returns a dictionary with dataset metadata, such that all
  information is available per netcdf but data is not loaded in saving memory and runtime. 
  '''

  if isinstance(filetype,list):
    if len(filetype)==1:
      filetype = filetype[0]
    else:
      for f,file in enumerate(filetype):
        if glob.glob(project_path+f'/*{file}*.nc'):
          file_path = project_path+f'/*{file}*.nc*'
          ds = xr.open_mfdataset(file_path,concat_dim='time',combine='nested')
          if f==0:
            base_ds = ds
          else:
            base_ds = xr.merge([base_ds,ds])
          #else:
          #  base_ds = xr.merge([base_ds,read_var])
        return base_ds

  if glob.glob(project_path+f'/*{filetype}*.nc'):
    file_path = project_path+f'/*{filetype}*.nc*'
    ds = xr.open_mfdataset(file_path,concat_dim='time',combine='nested')
    return ds

  else:
    print(f'Could not find {filetype} in {project_path}')
    # Get all simulation types for project:
    simulation_directories = []
    for entry in os.scandir(project_path):
      if entry.is_dir():
        simulation_directories.append(entry.name)
    # sort them (these are essentially the base identifiers of diff. simulations)
    simulations = np.sort(simulation_directories)
    
    for i,val in enumerate(simulations):
      print(f'{i}: {val}')
    
    # select parameters
    print('Select indices of simulations to load', '\n', 'PRESS ENTER WHEN READY')
    
    parameters = {}
    
    # get user input
    user_input = input("Seperate each input with a space: ")
    user_input = np.fromstring(user_input,dtype=int,sep=' ')
    sim_indices = user_input
  
    # if only one sim just return datatset
    if len(sim_indices)==1:
      sim_files_path = project_path+'/'+simulations[sim_indices[0]]+'/'+f'*{filetype}*.nc'
      ds = xr.open_mfdataset(sim_files_path, concat_dim='time',combine='nested')

      return ds

    # otherwise load dictionary of all our datatsets
    else:
      # load a dictionary of with each dataset lazily loaded in
      datasets = {}
      n = 1
      for i in sim_indices:
        sim = simulations[i]  # grabs specifc sim per specific indices
        if verbose:
          print(f'Loading in: {sim}')
        sim_files_path = project_path+'/'+sim+'/'+f'*{filetype}*.nc'
        ds = xr.open_mfdataset(sim_files_path, concat_dim='time',combine='nested')  # parallel and flattens outputs all into one set
        # add to dictionary
        datasets[n] = {'sim': sim,
                       'dataset':ds}
        n += 1

      return datasets

#       !]

# ---- 2. Reading in Datsets    ---- #

# info  ![
def info(ds):
  # horizontal points
  nx,ny = ds.sizes['xi_rho'],ds.sizes['eta_rho']
  # vertical
  try:
    nz=ds.sizes['s_rho']
  except:
    if 's_w' in ds.sizes:
      nz=ds.sizes['s_w']
    elif 's_win' in ds.sizes:
      nz=ds.sizes['s_win']
  # time
  try:
    nt=ds['ocean_time'].values
  except:
    print('ocean_time not found, returning array')
    nt=np.arange(0,ds.sizes['time'])
  
  return nx,ny,nz,nt

#       !]

# read  ![
def read(ds,var,                           # ds - netCDF dataset, var - string of requested variable
         sliced=None,cross=None, hov=None, # plotting optimizations for slicing before writing data to disk
                                           # see below for specific inputs
         verbose=True,                     # if true, will print statements for profiling
         interpolate=False,                # if true, will automatically interpolate all data to rho points
                                           # it is natively set to false to prevent error propogation in derivatives
         to_numpy=False):                  # only should be true if you want to write all to disk

  '''
  A python version of the ROMS nc_read_write module's ncread function, to streamline the input of data
  given a specific dataset (using xarray) and variable of interest. Additionally, data used specifically
  for visualization may be inputted with premptive slicing that ensures only what is necessary is written
  to disk - this is very helpful with large ROMS outputs.
  '''

  # check if var is present
  if var in ds.variables:
    var_ds = ds[var]

    # verbose print
    if verbose:
      print(f'Reading in {var}')

    # Vertical Slicer:
    # used for both surface plots and hovmoller vertically slicing
    # inputs (surface plot)   --> sliced = 'top' or some sigma level
    #  ...   (hovmoller plot) --> hov = ('top',y,x)
    if (sliced is not None or hov is not None) and ('s_win' in var_ds.dims or 's_rho' in var_ds.dims or 's_w' in var_ds.dims):
       slice_arr = [None,-1,None,None]
       var_ds = slicer(var_ds,slice_arr)
#      if sliced == 'top' or hov[0]=='top':
#        vslice = var_ds.sizes['s_rho'] - 1
#        var_ds = var_ds.isel(s_rho=vslice)
#      else:
#        if sliced is not None:
#          vslice = sliced
#          var_ds = var_ds.isel(s_rho=vslice)
#        elif hov[0] is not None:
#          vslice = hov[0]
#          var_ds = var_ds.isel(s_rho=vslice)

    # Cross-section Plot:
    # inputs: cross = (y, x) where if y is an int, x is None (or vice versa)
    # slices array along a specified horizontal dimension for vertical cross section view
    # only works for variables with depth dimension...
    if (cross is not None) and ('s_win' in var_ds.dims or 's_rho' in var_ds.dims or 's_w' in var_ds.dims) and len(var_ds.dims)>=3:
      if cross[0] is None and cross[1] is None:
        raise ValueError('Please pick an x or y to slice along')
      if cross[0] is None:
        if 'xi_u' in var_ds.dims:
          var_ds = var_ds.isel(xi_u=cross[1]-1)
        else:
          var_ds = var_ds.isel(xi_rho=cross[1])
      elif cross[1] is None:
        if 'eta_v' in var_ds.dims:
          var_ds = var_ds.isel(eta_v=cross[0]-1)
        else:
          var_ds = var_ds.isel(eta_rho=cross[0])
      else:
        raise ValueError('At least x or y must be None')

    # Hovmoller Plot:
    # inputs: hov = (z, y, x) where type is int or None. Z may also be 'top' for surface.
    # Slices array vertically (see sliced option) if needed then along specified x/y
    # OR does not slice vertically but slices along x and y
    # only works for variables with time dimension...
    if (hov is not None) and ('time' in var_ds.dims) and (len(var_ds.dims)>2):
      # data with depth and pre-sliced vertical OR no depth data
      if (hov[0] is not None) or not ('s_rho' in var_ds.dims or 's_w' in var_ds.dims or 's_win' in var_ds.dims):
        if hov[1] is None:
          if 'xi_u' in var_ds.dims:
            var_ds = var_ds.isel(xi_u=hov[2]-1)
          else:
            var_ds = var_ds.isel(xi_rho=hov[2])
        elif hov[2] is None:
          if 'eta_v' in var_ds.dims:
            var_ds = var_ds.isel(eta_v=hov[1]-1)
          else:
            var_ds = var_ds.isel(eta_rho=hov[1])
        else:
          raise ValueError('At least one x or y must be a value to plot with time')
      # if indeed data with depth and hov[0] is None, slice both x and y
      else:
        if 'xi_u' in var_ds.dims:
          var_ds = var_ds.isel(xi_u=hov[2]+1)
        else:
          var_ds = var_ds.isel(xi_rho=hov[2])
        if 'eta_v' in var_ds.dims:
          var_ds = var_ds.isel(eta_v=hov[1]+1)
        else:
          var_ds = var_ds.isel(eta_rho=hov[1])
       
    if interpolate:
      data = get_var(var_ds)
    else:
      data = var_ds

    # return data once grabbed for either plot or general use
    if to_numpy:
      return data.to_numpy()
    else:
      return data

  # if variable is not included in dataset, check if it can be computed
  else:
    # the Diagnostics class is a class of functions with recorded requisites 
    # for easy calculation on the fly with informed error messages
    calc = calculate.Diagnostics()

    # if this has a function to support its calculation, get necessary variables
    if var in calc.registry:
      needed_variables = calc.registry[var]['requires'] 
      if verbose:
        print(f'Computing {calc.registry[var]['description']}')
      
      # fills inputs with necessary varibles to later plug into compute function
      # for the requested diagnostic variable
      inputs = {}
      for name in needed_variables:
        var_ds = ds[name].to_numpy()
        inputs[name] = var_ds
      # will specify if it will be returned as a dataset or numpy array
      inputs['to_numpy'] = to_numpy

      return calc.compute(var,**inputs) 

    # IF variable if not in netCDF or a part of diagnostics
    else:
      raise ValueError('The requested variable is not in netCDF nor included in \
                        available calculations in calculate.py')

#       !]

# slicer![
def slicer(var_ds,
           slice_arr):

  # build a dictionary to slice data
  isel_dict = {}
  # get all possible dimensions
  dims = var_ds.dims

  # erase dimensions that do not exist for this variable
  slice_copy = np.array(slice_arr)
  if not any(dim.startswith('time') for dim in dims):
    slice_copy[0] = -99
  if not any(dim.startswith('s_')   for dim in dims):
    slice_copy[1] = -99
  if not any(dim.startswith('eta')  for dim in dims):
    slice_copy[2] = -99
  if not any(dim.startswith('xi')   for dim in dims):
    slice_copy[3] = -99
  # if dimension was missing we erase it from the array of indices
  slice_copy = slice_copy[slice_copy!=-99]

  # fill dictionary based on provided dimensions
  for d,dim in enumerate(dims):
    if slice_copy[d] is None:
      continue
    isel_dict[dim] = slice_copy[d]

  return var_ds.isel(**isel_dict)
#       !]

# ---- 3. Coordinate Transforms ---- #

#  Variable Interpolation  ![
def get_var(var_ds):
  if 'xi_u' in var_ds.dims:
    var_ds = u2rho(var_ds)
  elif 'eta_v' in var_ds.dims:
    var_ds = v2rho(var_ds)
  return var_ds

#  U --> RHO  ![
def u2rho(var_u):
  if isinstance(var_u,xr.DataArray):
    var_rho = 0.5 * (var_u + var_u.shift(xi_u=1))
    var_rho = var_rho.isel(xi_u=slice(1,None))
    var_rho = var_rho.rename(xi_u='xi_rho')
    var_rho = var_rho.pad(xi_rho=(1,1),mode='edge')
  elif isinstance(var_u,np.ndarray):  
    if np.ndim(var_u)<=2:
      var_rho = u2rho_2d(var_u)
    elif np.ndim(var_u)==3:
      var_rho = u2rho_3d(var_u)
    elif np.ndim(var_u)==4:
      var_rho = u2rho_4d(var_u)
  return var_rho

def u2rho_2d(var_u):
  [Mp,L]=var_u.shape
  Lp=L+1
  Lm=L-1
  var_rho=np.zeros((Mp,Lp))
  var_rho[:,1:L]=0.5*(var_u[:,0:Lm]+var_u[:,1:L])
  var_rho[:,0]=var_rho[:,1]
  var_rho[:,Lp-1]=var_rho[:,L-1]
  return var_rho

def u2rho_3d(var_u):
  [N,Mp,L]=var_u.shape
  Lp=L+1
  Lm=L-1
  var_rho=np.zeros((N,Mp,Lp))
  var_rho[:,:,1:L]=0.5*(var_u[:,:,0:Lm]+var_u[:,:,1:L])
  var_rho[:,:,0]=var_rho[:,:,1]
  var_rho[:,:,Lp-1]=var_rho[:,:,L-1]
  return var_rho

def u2rho_4d(var_u):
  [K,N,Mp,L]=var_u.shape
  Lp=L+1
  Lm=L-1
  var_rho=np.zeros((K,N,Mp,Lp))
  var_rho[:,:,:,1:L]=0.5*(var_u[:,:,:,0:Lm]+var_u[:,:,:,1:L])
  var_rho[:,:,:,0]=var_rho[:,:,:,1]
  var_rho[:,:,:,Lp-1]=var_rho[:,:,:,L-1]
  return var_rho

#  !]

#  V --> RHO  ![
def v2rho(var_v):
  if isinstance(var_v,xr.DataArray):
    var_rho = 0.5 * (var_v + var_v.shift(eta_v=1))
    var_rho = var_rho.isel(eta_v=slice(1,None))
    var_rho = var_rho.rename(eta_v='eta_rho')
    var_rho = var_rho.pad(eta_rho=(1,1),mode='edge')
  elif isinstance(var_v,np.ndarray):
    if np.ndim(var_v)<=2:
      var_rho = v2rho_2d(var_v)
    elif np.ndim(var_v)==3:
      var_rho = v2rho_3d(var_v)
    elif np.ndim(var_v)==4:
      var_rho - v2rho_4d(var_v)
  return var_rho

def v2rho_2d(var_v):
  [M,Lp]=var_v.shape
  Mp=M+1
  Mm=M-1
  var_rho=np.zeros((Mp,Lp))
  var_rho[1:M,:]=0.5*(var_v[0:Mm,:]+var_v[1:M,:])
  var_rho[0,:]=var_rho[1,:]
  var_rho[Mp-1,:]=var_rho[M-1,:]
  return var_rho

def v2rho_3d(var_v):
  [N,M,Lp]=var_v.shape
  Mp=M+1
  Mm=M-1
  var_rho=np.zeros((N,Mp,Lp))
  var_rho[:,1:M,:]=0.5*(var_v[:,0:Mm,:]+var_v[:,1:M,:])
  var_rho[:,0,:]=var_rho[:,1,:]
  var_rho[:,Mp-1,:]=var_rho[:,M-1,:]
  return var_rho

def v2rho_4d(var_v):
  [K,N,M,Lp]=var_v.shape
  Mp=M+1
  Mm=M-1
  var_rho=np.zeros((K,N,Mp,Lp))
  var_rho[:,:,1:M,:]=0.5*(var_v[:,:,0:Mm,:]+var_v[:,:,1:M,:])
  var_rho[:,:,0,:]=var_rho[:,:,1,:]
  var_rho[:,:,Mp-1,:]=var_rho[:,:,M-1,:]
  return var_rho

#  !]

#  !]

















