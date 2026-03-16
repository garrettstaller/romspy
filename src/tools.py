import numpy as np
import xarray as xr
ROMSPY_PATH = os.environ['ROMSPY_ROOT']
sys.path.append(ROMSPY_PATH)
import nc_read_write as nc

# ---- ROMS Depths ---- #

# Functions to calculate depth values at resepctive sigma levels 
# get_zrzw_tind --> compute_zslice --> set_depth (for both rho and w levels) --> stretching (respective stretch functions)
# compute_zslice   ![
def compute_zslice(t, zeta, h, Vtrans, Vstret, theta_s, theta_b, hc, N):
    zr = set_depth(
        Vtrans, Vstret, theta_s, theta_b, hc,
        N, 1, h, zeta[t]
    ).T

    zw = set_depth(
        Vtrans, Vstret, theta_s, theta_b, hc,
        N, 5, h, zeta[t]
    ).T

    return t, zr, zw
#                  !]

# get_zrzw_tind    ![
def get_zrzw_tind(ds, grd_ds, parallel=False):

    nt = len(ds.time)
    N  = ds.sizes['s_rho']

    Ly_all, Lx_all = nc.read(grd_ds, 'pm', to_numpy=True).shape

    Vtrans = 2
    Vstret = 4
    hc = ds.attrs['hc']
    theta_b = ds.attrs['theta_b']
    theta_s = ds.attrs['theta_s']

    h = nc.read(grd_ds, 'h', to_numpy=True)
    zeta = nc.read(ds, 'zeta', to_numpy=True)

    z_r = np.empty((nt, N,   Ly_all, Lx_all), dtype=np.float32)
    z_w = np.empty((nt, N+1, Ly_all, Lx_all), dtype=np.float32)

    if not parallel:
        from tqdm import tqdm
        for t in tqdm(range(nt),desc='Zslice in progress'):
          z_r_t = set_depth(
                  Vtrans, Vstret, theta_s, theta_b, hc,
                  N, 1, h, zeta[t])

          z_w_t = set_depth(
                  Vtrans, Vstret, theta_s, theta_b, hc,
                  N, 5, h, zeta[t])

          z_r[t] = np.transpose(z_r_t, (2, 0, 1))
          z_w[t] = np.transpose(z_w_t, (2, 0, 1))

    else:
        from joblib import Parallel, delayed
        from tqdm import tqdm

        results = Parallel(n_jobs=10, backend="threading")(
            delayed(compute_zslice)(
                t, zeta, h, Vtrans, Vstret, theta_s, theta_b, hc, N
            )
            for t in range(nt)        )

        for t, zr, zw in results:
            z_r[t] = np.swapaxes(zr,1,2)
            z_w[t] = np.swapaxes(zw,1,2)

    return z_r, z_w
#                  !]

# set_depth        ![
def set_depth( Vtr, Vstr, thts, thtb, hc, N, igrid, h, zeta ):
    Np      = N+1
    Lp,Mp   = np.shape(h)
    L       = Lp-1
    M       = Mp-1
    if (igrid==5):
        z   = np.empty((Lp,Mp,Np))
    else:
        z   = np.empty((Lp,Mp,N))

    hmin    = np.min(h)
    hmax    = np.max(h)

    if (igrid == 5):
        kgrid=1
    else:
        kgrid=0

    s,C = stretching(Vstr, thts, thtb, hc, N, kgrid);
    #-----------------------------------------------------------------------
    #  Average bathymetry and free-surface at requested C-grid type.
    #-----------------------------------------------------------------------

    if (igrid==1):
        hr    = h
        zetar = zeta
    elif (igrid==2):
        hp    = 0.25*(h[0:L,0:M]+h[1:Lp,0:M]+h[0:L,1:Mp]+h[1:Lp,1:Mp])
        zetap = 0.25*(zeta[0:L,0:M]+zeta[1:Lp,0:M]+zeta[0:L,1:Mp]+zeta[1:Lp,1:Mp])
    elif (igrid==3):
        hu    = 0.5*(h[0:L,0:Mp]+h[1:Lp,0:Mp])
        zetau = 0.5*(zeta[0:L,0:Mp]+zeta[1:Lp,0:Mp])
    elif (igrid==4):
        hv    = 0.5*(h[0:Lp,0:M]+h[0:Lp,1:Mp])
        zetav = 0.5*(zeta[0:Lp,0:M]+zeta[0:Lp,1:Mp])
    elif (igrid==5):
        hr    = h
        zetar = zeta

    #----------------------------------------------------------------------
    # Compute depths (m) at requested C-grid location.
    #----------------------------------------------------------------------
    if (Vtr == 1):
        if (igrid==1):
            for k in range (0,N):
                z0 = (s[k]-C[k])*hc + C[k]*hr
                z[:,:,k] = z0 + zetar*(1.0 + z0/hr)
        elif (igrid==2):
            for k in range (0,N):
                z0 = (s[k]-C[k])*hc + C[k]*hp
                z[:,:,k] = z0 + zetap*(1.0 + z0/hp)
        elif (igrid==3):
            for k in range (0,N):
                z0 = (s[k]-C[k])*hc + C[k]*hu
                z[:,:,k] = z0 + zetau*(1.0 + z0/hu)
        elif (igrid==4):
            for k in range (0,N):
                z0 = (s[k]-C[k])*hc + C[k]*hv
                z[:,:,k] = z0 + zetav*(1.0 + z0/hv)
        elif (igrid==5):
            z[:,:,0] = -hr
            for k in range (0,Np):
                z0 = (s[k]-C[k])*hc + C[k]*hr
                z[:,:,k] = z0 + zetar*(1.0 + z0/hr)
    elif (Vtr==2):
        if (igrid==1):
            for k in range (0,N):
                z0 = (hc*s[k]+C[k]*hr)/(hc+hr)
                z[:,:,k] = zetar+(zeta+hr)*z0
        elif (igrid==2):
            for k in range (0,N):
                z0 = (hc*s[k]+C[k]*hp)/(hc+hp)
                z[:,:,k] = zetap+(zetap+hp)*z0
        elif (igrid==3):
            for k in range (0,N):
                z0 = (hc*s[k]+C[k]*hu)/(hc+hu)
                z[:,:,k] = zetau+(zetau+hu)*z0
        elif (igrid==4):
            for k in range (0,N):
                z0 = (hc*s[k]+C[k]*hv)/(hc+hv)
                z[:,:,k] = zetav+(zetav+hv)*z0
        elif (igrid==5):
            for k in range (0,Np):
                z0 = (hc*s[k]+C[k]*hr)/(hc+hr)
                z[:,:,k] = zetar+(zetar+hr)*z0

    return z
#                  !]

# stretching       ![
def stretching(Vstr, thts, thtb, hc, N, kgrid):
    s=[]
    C=[]

    Np=N+1

    #-----------------------------------------------------------------
    # Compute ROMS S-coordinates vertical stretching function
    #-----------------------------------------------------------------

    # Original vertical stretching function (Song and Haidvogel, 1994).
    if (Vstr == 1):
        ds = 1.0/N

        if (kgrid == 1):
            Nlev = Np
            lev  = np.linspace(0.0,N,Np)
            s    = (lev-N)*ds
        else:
            Nlev = N
            lev  = np.linspace(1.0,N,Np)-0.5
            s    = (lev-N)*ds

        if (thts > 0):
            Ptheta = np.sinh(thts*s)/np.sinh(thts)
            Rtheta = np.tanh(thts*(s+0.5))/(2.0*np.tanh(0.5*thts))-0.5
            C      = (1.0-thtb)*Ptheta+thtb*Rtheta
        else:
            C=s

    # A. Shchepetkin (UCLA-ROMS, 2005) vertical stretching function.
    if (Vstr==2):
        alfa = 1.0
        beta = 1.0
        ds   = 1.0/N

        if (kgrid == 1):
            Nlev = Np
            lev  = np.linspace(0.0,N,Np)
            s    = (lev-N)*ds
        else:
            Nlev = N
            lev  = np.linspace(1.0,N,Np)-0.5
            s    = (lev-N)*ds

        if (thts > 0):
            Csur = (1.0-np.cosh(thts*s))/(np.cosh(thts)-1.0)
            if (thtb > 0):
                Cbot   = -1.0+np.sinh(thtb*(s+1.0))/np.sinh(thtb)
                weigth = (s+1.0)**alfa*(1.0+(alfa/beta)*(1.0-(s+1.0)**beta))
                C      = weigth*Csur+(1.0-weigth)*Cbot
            else:
                C=Csur
        else:
            C=s

    # R. Geyer BBL vertical stretching function.
    if (Vstr==3):
        ds   = 1.0/N

        if (kgrid == 1):
            Nlev = Np
            lev  = np.linspace(0.0,N,Np)
            s    = (lev-N)*ds
        else:
            Nlev = N
            lev  = np.linspace(1.0,N,Np)-0.5
            s    = (lev-N)*ds

        if (thts > 0):
            exp_s = thts   # surface stretching exponent
            exp_b = thtb   # bottom  stretching exponent
            alpha = 3      # scale factor for all hyperbolic functions
            Cbot  = np.log(np.cosh(alpha*(s+1.0)**exp_b))/np.log(np.cosh(alpha))-1.0
            Csur  = -np.log(cosh(alpha*abs(s)**exp_s))/log(cosh(alpha))
            weight= (1-np.tanh( alpha*(s+0.5)))/2.0
            C     = weight*Cbot+(1.0-weight)*Csur
        else:
            C=s

    # A. Shchepetkin (UCLA-ROMS, 2010) double vertical stretching function
    # with bottom refinement
    if (Vstr == 4):
        ds   = 1.0/N
        if (kgrid == 1):
            Nlev = Np
            lev  = np.linspace(0.0,N,Np)
            s    = (lev-N)*ds
        else:
            Nlev = N
            #lev  = np.linspace(1.0,N,Np)-0.5
            lev  = np.linspace(1.0,N,N)-0.5
            s    = (lev-N)*ds

        if (thts > 0):
            Csur = (1.0-np.cosh(thts*s))/(np.cosh(thts)-1.0)
        else:
            Csur = -s**2

        if (thtb > 0):
            Cbot = (np.exp(thtb*Csur)-1.0)/(1.0-np.exp(-thtb))
            C    = Cbot
        else:
            C    = Csur

    return (s,C)
#                  !]

# call in plotting or diagnostic script to load in correct depth values
# get_depths       ![
def get_depths(path,            # location of depths file (should be in same directory as data)
               time_index=None, # slices specific time, otherwise all data is loaded
               ptype=None,      # for pre-emptive slicing to save space
               param=None):     # slicing directions

  read_inputs = {}
  # error handling
  if (ptype and not param) or (param and not ptype):
    raise ValueError('Must include plot type and slicing')
  else:
   # we only need depths for cross sections or hovmoller of depth profiles...
   if ptype=='cross' or ptype=='hov':
     # we have correctly inputted slicing preferences - save these to dictionary 
     read_inputs[ptype] = param

  # get depths dataset
  depths_ds = xr.open_mfdataset(path,concat_dim='time',combine='nested')

  # read in data, pre-emptive slicing included if necessary
  zr_ds = nc.read(depths_ds,'z_r',**read_inputs) if (ptype and param) else nc.read(depths_ds,'z_r')
  zw_ds = nc.read(depths_ds,'z_w',**read_inputs) if (ptype and param) else nc.read(depths_ds,'z_w')

  # slice in time if needed 
  zr = zr_ds.isel(time=time_index).values if time_index is not None else zr_ds.values
  zw = zw_ds.isel(time=time_index).values if time_index is not None else zw_ds.values

  return zr,zw

#                  !]

# ---- ROMS Tools Misc. ---- #

