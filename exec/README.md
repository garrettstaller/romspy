# shell_py
When run in a directory with ROMS outputs, 'his' files and/or grd's, it will load these lazily into
an interactive Python environment as xarray dataset. Can also be run in a parent directory of multiple
subdirectories with different simulations to immeditaely begin plotting from CL. 

# depth_slicer
**Must run before plotting cross sections or Hovmollers of depth profiles**
When run in a directory with _his and _grd files from ROMS, will read in the free-surface height and grd properties
to calculate depths at s_rho and s_w vertical coordinates.

# roms_plotter
***Work in progress***
This is tool utilizes Xarray lazy loading to rapidly make plots of multiple variables and/or simulations. The user can edit 
which type of plot, the variables to plot and other specific within the script. 

The types of plots include:
- 'sliced' where the plot_param is either 'top' (for surface) or some integer for a zslice (if avaialable) or sigma level.
- 'cross' where plot_param is either [Y,NONE] or [NONE,X] for cross-sections of X or Y, respectively. 
- 'hov' (Hovmoller) where plot_param is either ['top',Y,NONE], ['top',NONE,X], or [NONE,Y,X]
    - whatever dimension [Z,Y,X] is NONE will be the one plotted against time
** For cross-section and Hovmollers, the depths at each sigma coordiante is needed and thus you may need to run depth_slicer**

To plot across multiple simulations, the user may call this tool from the CL in the parent directory of their simulation directories. 
Assuming there are no netCDF files in this parent, and the exclude variable in roms_plotter rules out specific directories to check, the user can 
input which simulations to load in for plotting! 

This also comes with animation capabilities by reading netcdfs to disk, then quickly plotting onto the figure object made in the viewing portion.
Such that normal workflow includes setting variables (adjusting their ranges), loading in for a specific time value (default of -1 for final output of sim),
and then viewing the plot. If approved, 'ctrl+d' in the interactive viewer will begin animation process.
