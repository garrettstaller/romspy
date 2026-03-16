'''
Offline Diagnostics for ROMS Outputs - Garrett Staller

Organization:
 - General Class Architecture
 - Common outputs
 - Project specific 

'''
import numpy as np
import xarray as xr
ROMSPY_PATH = os.environ['ROMSPY_ROOT']
sys.path.append(ROMSPY_PATH)
import nc_read_write as nc

# Constants  ![
# changes made here will apply to all functions below
# and the scripts that utilize them

define_constants = { # set below
                    'thermal-coeff': 2e-4,
                    'gravity': 9.81
                                 } 

#            !]

# Overarching Class Architecture  ![
class DiagnosticError(Exception):
    pass

class Diagnostics:
    """
    Diagnostic calculator with self-registering diagnostics.
    """

    registry = {}

    @classmethod
    def register(cls, name, *, requires, description, dimensions):
        def decorator(func):
            cls.registry[name] = {
                "func": func,
                "requires": requires,
                "description": description,
                "dimensions": dimensions
            }
            return func
        return decorator

    def compute(self, name, **kwargs):
        if name not in self.registry:
            raise DiagnosticError(
                f"Diagnostic '{name}' not implemented.\n"
                f"Available diagnostics: {list(self.registry)}"
            )

        entry = self.registry[name]
        missing = set(entry["requires"]) - kwargs.keys()
        if missing:
            raise DiagnosticError(
                f"Diagnostic '{name}' missing inputs: {missing}"
            )

        return entry["func"](**kwargs)

    def available(self):
        return {
            k: v["description"] for k, v in self.registry.items()
        }

#                                 !]

# Compute Functions ![

def compute_var(var,  # variable name as string
                ds,   # dataset to calculate diagnostics
                verbose=True,
                to_numpy=True):
  # set class
  calc = Diagnostics

  # if this has a function to support its calculation, get necessary variables
  if var in calc.registry:
    needed_variables = calc.registry[var]['requires']
    if verbose:
      print(f'Computing {calc.registry[var]['description']}')

    # fills inputs with necessary varibles to later plug into compute function
    # for the requested diagnostic variable
    inputs = {}
    for name in needed_variables:
      var_ds = nc.read(ds,name,verbose=verbose,to_numpy=to_numpy)
      inputs[name] = var_ds
    # will specify if it will be returned as a dataset or numpy array
    inputs['to_numpy'] = to_numpy
    print(inputs)

    return calc.compute(var,**inputs)


#                   !]

# ----- Computable Variables ----- #
# The below computables are also available to ncread by registry...

# Vorticity  ![
@Diagnostics.register(
    name="vort",
    requires=["ubar","vbar"],
    description="Vertical Relative Vorticity",
    dimensions=['time','eta_rho','xi_rho'])

def vort(ubar,vbar,to_numpy=False):
  
  # Perform calculation
  # centered-difference
  dvdx = np.gradient(vbar,axis=2)*.02
  dudy = np.gradient(ubar,axis=1)*.02
  
  # get on common grid and solve
  dudy = nc.u2rho_3d(dudy)
  dvdx = nc.v2rho_3d(dvdx)
  xi = dvdx - dudy
  xi = xi/(8.22e-5)

  # add to a dataset
  if not to_numpy:
    xi = xr.DataArray(xi,
                      dims=('time','eta_rho','xi_rho'),
                      name='vort',
                      attrs={})

  return xi

#            !]

# Divergence  ![
@Diagnostics.register(
    name="hdiv",
    requires=["ubar","vbar"],
    description="Horizontal divergence",
    dimensions=['time','eta_rho','xi_rho'])

def hdiv(ubar,vbar):
  dudx = np.gradient(ubar,axis=1)*.02
  dvdy = np.gradient(vbar,axis=0)*.02
  hdiv = dudx + dvdy
  return hdiv

#             !]

# Stratification  ![
@Diagnostics.register(
    name="n2",
    requires=["temperature","rho depths"],
    description="Brunt-Vaisailla Frequnecy",
    dimensions=['time','s_win','eta_rho','xi_rho'])

def stratification(temp,depths):

  # Constants
  g = define_constants['gravity']
  alpha = define_constants['thermal-coeff'] 

  # calculate dT/dZ
  dt = np.empty(temp.shape)
  dz = np.empty(temp.shape)
  
  # currently assumes temp is 4d (okay if run with raw temp output)
  # NOTE: change in future so calculation from pre-slice array is option
  nz = len(temp[1])
  for t in range(nz-1):
    dt[:,t,:,:] = temp[:,t+1,:,:] - temp[:,t,:,:]
    dz[:,t,:,:] = depths[:,t+1,:,:] - depths[:,t,:,:]
  
  n2 = g*alpha*(dt/dz)

  # nan out negative stratification and 0
  n2[n2<=0] = np.nan

  return n2

#                 !]

# Average Kinetic Energy  ![
@Diagnostics.register(
    name="avg_ke",
    requires=["ubar","vbar"],
    description="Integreated Kinteic Energy",
    dimensions=['time'])

def avg_ke(ubar,vbar):

  # square values then convert them to rho points for summing
  ubar_sq_rho = nc.u2rho(ubar**2)
  vbar_sq_rho = nc.v2rho(vbar**2)

  ke = .5*(ubar_sq_rho + vbar_sq_rho) 

  avg_ke = np.mean(ke,axis=(1,2))

  return avg_ke
#                         !]

