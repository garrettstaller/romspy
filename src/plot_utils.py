'''
plot_functions.py - Garrett S,

Purppse:
Streamlining the plotting process for future scripts. 

'''
# User Edits to apply to ALL scripts:
# Fontsizes:
default_fonts = { # set below
                 'fs_label': 14,
                 'fs_tick':  12,
                 'fs_title': 16,
                 'fs_text':  13,
                 }

# Other specifics
defaults = { # set below
            'nticks':5,              # number of axes ticks
            'colorbar': 'viridis',   # cmap filler
            'mask'    : 'lightgray', # color of masks
            'contours': 'darkseagreen',   # color of contours
            'ncontours': 7,
             # Different scaling of the width of plots based on 
             # extreme aspect ratios
            'base_width': 5,         # no issue
            'base_width_long': 8,    # wide plots (cross-sections/hovmollers)
            'base_width_short': 3}   # narrow plots (hovmollers in y or long alongshore)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmocean
import numpy as np
ROMSPY_PATH = os.environ['ROMSPY_ROOT']
sys.path.append(ROMSPY_PATH)
import nc_read_write as nc

# get_arrays    ![
def get_arrays(nx,ny,nz,nt,  # length (indices) of x and y dimensions
               grd_ds,       # grd_info dictionary from nc.get_grd() for pm and pn
               # Optional
               ptype=None,   # plot type (cross or sliced or hov)
               param=None):  # parameter for plot type

  # error handling
  if (ptype and not param) or (param and not ptype):
    raise ValueError('Must include plot type and slicing')

  # from grd_info
  pm = grd_ds['pm'].isel(eta_rho=1,xi_rho=1).values
  pn = grd_ds['pn'].isel(eta_rho=1,xi_rho=1).values

  # get x and y arrays
  x = np.arange(0,nx)*(1/pm)
  y = np.arange(0,ny)*(1/pn)
  if np.max(x)>=2.5e2:  # scale these to km's if
    x = x/1e3           # values exceed 250m
  if np.max(y)>=2.5e2:
    y = y/1e3
  # depth or z array
  z = np.arange(0,nz)
  # time dimensions
  if np.max(nt)>1e3:
    time = nt/86400
  else:
    time = nt/24

  # return all four arrays if no specific plot dimensions are requested
  if not (ptype and param):
    return x,y,z,time
  # otherwise check each plot
  else:
    # first conditionals get X and Y dimensions for plotting
    if ptype=='sliced':
      X,Y = x,y
    elif ptype=='cross':
      if param[0] is None:
        [yy,zz] = np.meshgrid(y,z)
        X,Y = yy[0],z
      else:
        [xx,zz] = np.meshgrid(x,z)
        X,Y = xx[0],z
    elif ptype=='hov':
      if param[0] is None:
        X,Y = time,z
      elif param[1] is None:
        X,Y = time,y
      elif param[2] is None:
        X,Y = x,time
    # we return the X and Y
    return X,Y
  
#             !]

# get_fig       ![
def get_fig(nrows,ncols,        # how many subplots in y and x
            x,y,                # arrays we plot against 
            base_width = None): # Allow user to adjust how many inches
                                # the figure is scaled to in x/y

  # get subplot aspect ratio (based on model)
  ratio = len(y)/len(x)
  
  # aspect ratio defined above may be too extreme (depths versus cross-shore)
  # this tunes the ratio to still appear 'physical' while still inerpretable
  # the overwrites here are also in the default dictionary

  # too wide
  if ratio<(.4):
    ratio = .4
    bw = defaults['base_width_long']
    tuning = 1.0  # additional tuning
  # too narrow
  elif ratio>(1.77):
    ratio = 2.5
    bw = defaults['base_width_short']
    tuning = .60  # additional tuning
  # no issues with ratio
  else:
    bw = base_width if base_width is not None else defaults['base_width']
    tuning = .60

  # get complementary height based on our subplot aspect ratio and physical dimensions
  bh = ratio * bw

  # then get final fig size by scaling by rows and columns
  fig_height = tuning*nrows*bh  # fig height is scaled additionally with tuning
  fig_width  = ncols*bw

  # plot result and apply aspect ratio to each subplot
  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), constrained_layout=True)
  if (nrows*ncols)==1:
    axes = np.array([axes])
  axes = axes.flatten()
  for ax in axes:
    ax.set_box_aspect(ratio)
  
  return fig, axes

#          !]

# set_axes      ![
def set_axes(ax,                   # current axis in plot
             xlab,ylab,            # respective labels
             title=None,           # user can specify title
             title_fontsize=None,  # title fontsize switch
             label_fontsize=None,  # label fontsize switch
             total_ticks=None,     # number of ticks switch
             tick_fontsize=None):  # size of ticks switch
 
  from matplotlib.ticker import MaxNLocator

  fs_title = title_fontsize if title_fontsize is not None else default_fonts['fs_title']
  fs_label = label_fontsize if label_fontsize is not None else default_fonts['fs_label']
  fs_tick  = tick_fontsize  if tick_fontsize  is not None else default_fonts['fs_tick']
  nticks   = total_ticks    if total_ticks    is not None else defaults['nticks']

  ax.set_xlabel(xlab,fontsize=fs_label)
  ax.set_ylabel(ylab,fontsize=fs_label)
  if title is not None:
    ax.set_title(title,fontsize=fs_title)

  ax.xaxis.set_major_locator(MaxNLocator(nbins=nticks))
  ax.yaxis.set_major_locator(MaxNLocator(nbins=nticks))
  ax.tick_params(axis='both',labelsize=fs_tick)

#           !]

# set_colorbar  ![ 
def set_colorbar(im,ax,       # image and axis
                 var,         # var name (str)
                 data,        # raw data
                 # Optional Inputs #
                 log =False,  # turns on logarithmic scale for colorbar
                 cmin=None,   # ranges
                 cmax=None,   #  ...
                 cmap=None,
                 cfnt=None):

  # check for name and necessary details
  if var in VARIABLE_REGISTRY:
    clab = VARIABLE_REGISTRY[var]['name']   # full name with units
    ctype = VARIABLE_REGISTRY[var]['type']  # diverging colorbar or not
    cstyle = VARIABLE_REGISTRY[var]['cmap'] # optimal cmap
    cscale = log if log else VARIABLE_REGISTRY[var]['log']  # logarithmic scale
  else:
    clab = var
    ctype = False  # just plot from min to max vals (no diverging colorbar specification)
    cstyle = defaults['colorbar']  # default colorbar...
    cscale = log if log else False

  if not isinstance(data,np.ndarray): # convert to array if not already
    data = data.values
  
  # set bounds
  if ctype:
    # this is for divergent colorbars
    crange = np.max([np.abs(np.min(data)), np.max(data)])  # take the greatest value
    min_val = cmin if cmin is not None else -1*crange
    max_val = cmax if cmax is not None else crange
  # if we do not have divergence or ctype is not set (variable not in regsitry)
  else:
    min_val = cmin if cmin is not None else np.nanmin(data[data!=0])  # Non-zero minima
    max_val = cmax if cmax is not None else np.nanmax(data[data!=0])

  if cscale:
    im.set_norm(LogNorm(vmin=min_val,vmax=max_val))
  else:
    im.set_clim(min_val,max_val)

  # set colormap style
  chosen_cmap = cmap if cmap is not None else cstyle
  im.set_cmap(chosen_cmap)
  
  # create cbar object
  cbar = ax.figure.colorbar(im,ax=ax,shrink=.70)

  # label and tick sizes
  fs_label = cfnt if cfnt is not None else default_fonts['fs_label']
  fs_tick = cfnt if cfnt is not None else default_fonts['fs_label']

  # add them to colorbar object
  cbar.set_label(clab,fontsize=fs_label)
  cbar.ax.tick_params(labelsize=fs_tick)  

  return cbar  # must return this for later edits...

#           !]  

# add_cons      ![
def add_cons(ax,             # current axis (subplot) to plot against
             x,y,            # respective x/y axes
             h,              # array with values to contour
             limit=None,     # user specify range of depths, e.g. limit=[10,100]
             ncontours=None, # number of contours 
             clabels=None,   # provide depth values on contours
             fontsize=None,  # how big contour labels are
             lw=0.75,        # set linewidth (edit here if you insist)
             cb=None):       # colors of contour

  # make sure dimensions line up
  # u points
#  if x.shape[0] > h.shape[1]:
#    raise ValueError('Cannot plot u-point data on rho-point mask')

  # set number of contours
  n = ncontours if ncontours is not None else defaults['ncontours']

  # make sure contour range was given properly
  if limit is not None:
    if (not isinstance(limit,np.ndarray) and len(limit)!=2): 
      raise ValueError('limit must be an array of length two')
    limit = np.sort(limit)  # ensure increasing depth
    contour_values = np.linspace(limit[0],limit[1],n)
  # if not input then simply plot from min to max depths
  else:
    contour_values = np.linspace(np.min(h),np.max(h),n)
    print(f'contour vals: {contour_values}')
    
  # set color and fontsizes if labels are requested
  contour_color = cb if cb is not None else defaults['contours']

  # make object
  ch = ax.contour(x,y,h,contour_values,colors=contour_color,linewidths=lw)

  # add additional features
  if clabels:
    fs_contour = fontsize if fontsize is not None else default_fonts['fs_tick']
    ch.clabel(fontsize=fs_contour,fmt='%1.f')
 
  return ch
  
#                !]

# add_mask      ![
def add_mask(ax,          # axis to plot on
             x,y,         # resp. axes
             mask,        # mask of 1's (not masked) and 0's (mask)
             color=None): # color of mask

  # check that mask is 0 to 1
  if (np.min(mask)!=0 or np.max(mask)!=1):
    raise ValueError('Masked array must be 0 to 1')

  lvls = [-0.5,.5,1.5]  # fixed levels to only fill values less than 1

  # color inputs
  mask_color = color if color is not None else defaults['mask']

  # plot mask
  ax.contourf(x,y,mask,levels=lvls,colors=[mask_color,'none'])

#               !]

# Colorbar Registry #
VARIABLE_REGISTRY = {
                    # ---- Core ROMS Variables ---- #
                    # Temperature
                    "temp": {"name": "Temperature [$^\\circ$C]",
                             "type": False,
                             "log" : False,
                             "cmap": 'jet'},
                    # Free-Surface Height
                    "zeta": {"name": "Sea Surface Height [m]",
                             "type": True,
                             "log" : False,
                             "cmap": "RdBu_r"},
                    # U-Comp. Barotropic Vel.
                    "ubar": {"name": "Barotropic U Vel. [ms$^{-1}$]",
                             "type": True,
                             "log" : False,
                             "cmap": "RdBu_r"},
                    # V-Comp. Barotropic Vel.
                    "vbar": {"name": "Barotropic V Vel. [ms$^{-1}$]",
                             "type": True,
                             "log" : False,
                             "cmap": "RdBu_r"},
                    # U-Comp. Vel.
                    "u":    {"name": "U Velocity [ms$^{-1}$]",
                             "type": True,
                             "log" : False,
                             "cmap": "RdBu_r"},
                    # V-Comp. Vel.
                    "v":    {"name": "V Velocity [ms$^{-1}$]",
                             "type": True,
                             "log" : False,
                             "cmap": "RdBu_r"},

                    # ---- Grid File Variables  ---- #

                    # ---- Diagnostic Variables ---- #
                    # Vorticity
                    "vort": {"name": "Depth-Averaged $\\frac{\\zeta}{f}$",
                             "type": True,
                             "log" : False,
                             "cmap": "RdBu_r"},
                    # Cross-Shore Velocity
                    "cross": {"name": "Cross Shore Velocity [ms$^{-1}$]",
                             "type": True,
                             "log" : False,
                             "cmap": "RdBu_r"},
                    # Along-Shore Velocity
                    "along": {"name": "Along Shore Velocity [ms$^{-1}$]",
                             "type": True,
                             "log" : False,
                             "cmap": "RdBu_r"},
                    # Stratification (n2)
                    "n2": {"name": "Stratification,N$^2$ [s$^{-1}$]",
                           "type": True,
                           "log" : True,
                           "cmap": "viridis"},
                    # Cross-Shore Heat Flux
                    "cross_hf": {"name": "U'$_{cross}$'T' [ms$^{-1}$$^\\circ$C]",
                                 "type": True,
                                 "log" : False,
                                 "cmap": "RdBu_r"}
                                                    }
