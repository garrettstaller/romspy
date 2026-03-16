# Import Modules  ![
import os
import sys
# data
import xarray as xr
# ROMS PY
ROMSPY_PATH = os.environ['ROMSPY_ROOT']
sys.path.append(ROMSPY_PATH)
import nc_read_write as nc
import tools

#                 !]

# Load in data for zslicing
project_path = os.getcwd()
print(f'Getting files in {project_path} for depths')
ds = nc.load(project_path)

# acquire grid dataset
grd_ds = nc.get_grd(project_path+f'/*grd.nc',info=False)  # append sim name to path and specify grd

# Now call tools module to load in depths
zr, zw = tools.get_zrzw_tind(ds,grd_ds)

output_name = f'{os.path.split(os.getcwd())[1]}_depths.nc'

print(f'Saving as {output_name}')
zr_ds = xr.DataArray(zr,
                     dims=('time','s_rho','eta_rho','xi_rho'),
                     name='z_r')
zw_ds = xr.DataArray(zw,
                     dims=('time','s_w','eta_rho','xi_rho'),
                     name='z_w')

depths_ds = xr.merge([zr_ds,zw_ds])

depths_ds.to_netcdf(output_name)
