########################################
####    CMIP5 - Climate Analysis    ####
########################################
####     Behzad Asadieh, Ph.D.      ####
####  University of Pennsylvania    ####
####    basadieh@sas.upenn.edu      ####
####     github.com/behzadasd       ####
########################################

from Behzadlib import func_latlon_regrid, func_regrid, func_oceanlandmask, func_gridcell_area, func_plotmap_contourf, func_plot_laggedmaps
########################################
import numpy as np
import xarray as xr
import numpy.ma as ma
from netCDF4 import MFDataset, Dataset, num2date, date2num, date2index
import os
import matplotlib
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm, maskoceans
from scipy.interpolate import griddata
import math
import copy
########################################
dir_pwd = os.getcwd() # Gets the current directory (and in which the code is placed)

### Regrdridding calculations ###
# creating new coordinate grid, same which was used in interpolation in data processing code
lat_n_regrid, lon_n_regrid = 180, 360 # Number of Lat and Lon elements in the regridded data
lon_min_regrid, lon_max_regrid = 0, 360 # Min and Max value of Lon in the regridded data
lat_min_regrid, lat_max_regrid = -90, 90 # Min and Max value of Lat in the regridded data

# This function for creating new Lat-Lon fields is saved in Behzadlib code in this directory - imported at the begenning
Lat_regrid_1D, Lon_regrid_1D, Lat_bound_regrid, Lon_bound_regrid = func_latlon_regrid(lat_n_regrid, lon_n_regrid, lat_min_regrid, lat_max_regrid, lon_min_regrid, lon_max_regrid)
Lon_regrid_2D, Lat_regrid_2D = np.meshgrid(Lon_regrid_1D, Lat_regrid_1D)

# Land/Ocean mask - The function is saved in Behzadlib code in this directory - imported at the begenning
Ocean_Land_mask = func_oceanlandmask(Lat_regrid_2D, Lon_regrid_2D) # 1= ocean, 0= land

####################################################################
GCM_Names = ['GFDL-ESM2M', 'GFDL-ESM2G', 'IPSL-CM5A-MR', 'IPSL-CM5A-LR', 'MIROC-ESM', 'MIROC-ESM-CHEM', 'CESM1-BGC', 'CMCC-CESM', 'CanESM2', 'GISS-E2-H-CC', 'GISS-E2-R-CC', 'MPI-ESM-MR', 'MPI-ESM-LR', 'NorESM1-ME']
GCM = 'GFDL-ESM2G'

lat_t='lat'
lon_t='lon'
time_t='time'

##################################################################################################
###  Calculating Wind Curls using wind stress in latitudinal and longitudinal directions  ########
##################################################################################################
dir_data_in1 = ('/data2/scratch/cabre/CMIP5/CMIP5_models/atmosphere_physics/') # Directory to raed raw data from
dir_data_in2=(dir_data_in1+ GCM + '/historical/mo/')

year_start=1901
year_end=2000

Var_name='tauu' # The variable name to be read from .nc files
dset = xr.open_mfdataset(dir_data_in2+Var_name+'*12.nc')
Data_all = dset[Var_name]

Lat_orig = dset[lat_t]
Lat_orig = Lat_orig.values
Lon_orig = dset[lon_t]
Lon_orig = Lon_orig.values     

Data_monthly = Data_all[ (Data_all.time.dt.year >= year_start ) & (Data_all.time.dt.year <= year_end)]
Data_monthly = Data_monthly.values # Converts to a Numpy array
Data_rannual = Data_monthly.reshape( np.int(Data_monthly.shape[0]/12) ,12,Data_monthly.shape[1],Data_monthly.shape[2])
Data_rannual = np.nanmean(Data_rannual,axis=1)
Tau_X = func_regrid(Data_rannual, Lat_orig, Lon_orig, Lat_regrid_2D, Lon_regrid_2D)

Var_name='tauv' # The variable name to be read from .nc files
dset = xr.open_mfdataset(dir_data_in2+Var_name+'*12.nc')
Data_all = dset[Var_name] 
Data_monthly = Data_all[ (Data_all.time.dt.year >= year_start ) & (Data_all.time.dt.year <= year_end)]
Data_monthly = Data_monthly.values # Converts to a Numpy array
Data_rannual = Data_monthly.reshape( np.int(Data_monthly.shape[0]/12) ,12,Data_monthly.shape[1],Data_monthly.shape[2])
Data_rannual = np.nanmean(Data_rannual,axis=1)
Tau_Y = func_regrid(Data_rannual, Lat_orig, Lon_orig, Lat_regrid_2D, Lon_regrid_2D)

# Wind_Curl = ( D_Tau_Y / D_X ) - ( D_Tau_X / D_Y ) # D_X = (Lon_1 - Lon_2) * COS(Lat)
Wind_Curl = np.zeros(( Tau_X.shape[0], Tau_X.shape[1], Tau_X.shape[2]))  
for tt in range (0,Tau_X.shape[0]):  
    for ii in range (1,Tau_X.shape[1]-1):
        for jj in range (1,Tau_X.shape[2]-1): # Wind_Curl = ( D_Tau_Y / D_X ) - ( D_Tau_X / D_Y ) # D_X = (Lon_1 - Lon_2) * COS(Lat)
            
            Wind_Curl[tt,ii,jj] = (  ( Tau_Y[tt, ii,jj+1] - Tau_Y[tt, ii,jj-1] ) /  np.absolute(  ( ( Lon_regrid_2D[ii,jj+1] -  Lon_regrid_2D[ii,jj-1] ) * 111321  *  math.cos(math.radians( ( Lat_regrid_2D[ii,jj])))   )  )     )   -   (  ( Tau_X[tt, ii+1,jj] - Tau_X[tt, ii-1,jj] ) / np.absolute( ( ( Lat_regrid_2D[ii+1,jj] -  Lat_regrid_2D[ii-1,jj] ) * 111321 ) )  )

        Wind_Curl[tt,ii,0] = (  ( Tau_Y[tt, ii,1] - Tau_Y[tt, ii,-1] ) /  np.absolute(  ( ( Lon_regrid_2D[ii,1] -  Lon_regrid_2D[ii,-1] ) * 111321  *  math.cos(math.radians( ( Lat_regrid_2D[ii,0])))   )  )     )   -   (  ( Tau_X[tt, ii+1,0] - Tau_X[tt, ii-1,0] ) / np.absolute( ( ( Lat_regrid_2D[ii+1,0] -  Lat_regrid_2D[ii-1,0] ) * 111321 ) )  )
        Wind_Curl[tt,ii,-1] = (  ( Tau_Y[tt, ii,0] - Tau_Y[tt, ii,-2] ) /  np.absolute(  ( ( Lon_regrid_2D[ii,0] -  Lon_regrid_2D[ii,-2] ) * 111321  *  math.cos(math.radians( ( Lat_regrid_2D[ii,-1])))   )  )     )   -   (  ( Tau_X[tt, ii+1,-1] - Tau_X[tt, ii-1,-1] ) / np.absolute( ( ( Lat_regrid_2D[ii+1,-1] -  Lat_regrid_2D[ii-1,-1] ) * 111321 ) )  )

# Wind_Crul / f # f = coriolis parameter = 2Wsin(LAT) , W = 7.292E-5 rad/s
Wind_Curl_f = np.zeros(( Tau_X.shape[0], Tau_X.shape[1], Tau_X.shape[2])) 
for tt in range (0,Tau_X.shape[0]):   
    for ii in range (1,Tau_X.shape[1]-1):
        if np.absolute( Lat_regrid_2D[ii,0] ) >= 5: # Only calulate for Lats > 5N and Lats < 5S, to avoid infinit numbers in equator where f is zero
            for jj in range (1,Tau_X.shape[2]-1): # Wind_Curl = ( D_Tau_Y / D_X ) - ( D_Tau_X / D_Y ) # D_X = (Lon_1 - Lon_2) * COS(Lat)
            
                Wind_Curl_f[tt,ii,jj] = (  ( ( Tau_Y[tt, ii,jj+1] - Tau_Y[tt, ii,jj-1] ) / ( 2*7.292E-5 *  math.sin(math.radians( ( Lat_regrid_2D[ii,jj]))) ) ) /  np.absolute(  ( ( Lon_regrid_2D[ii,jj+1] -  Lon_regrid_2D[ii,jj-1] ) * 111321  *  math.cos(math.radians( ( Lat_regrid_2D[ii,jj])))   )  )     )   -   (  ( ( Tau_X[tt, ii+1,jj] - Tau_X[tt, ii-1,jj] ) / ( 2*7.292E-5 *  math.sin(math.radians( ( Lat_regrid_2D[ii,jj]))) ) ) / np.absolute( ( ( Lat_regrid_2D[ii+1,jj] -  Lat_regrid_2D[ii-1,jj] ) * 111321 ) )  )

            Wind_Curl_f[tt,ii,0] = (  ( ( Tau_Y[tt, ii,1] - Tau_Y[tt, ii,-1] ) / ( 2*7.292E-5 *  math.sin(math.radians( ( Lat_regrid_2D[ii,0]))) ) ) /  np.absolute(  ( ( Lon_regrid_2D[ii,1] -  Lon_regrid_2D[ii,-1] ) * 111321  *  math.cos(math.radians( ( Lat_regrid_2D[ii,0])))   )  )     )   -   (  ( ( Tau_X[tt, ii+1,jj] - Tau_X[tt, ii-1,0] ) / ( 2*7.292E-5 *  math.sin(math.radians( ( Lat_regrid_2D[ii,0]))) ) ) / np.absolute( ( ( Lat_regrid_2D[ii+1,0] -  Lat_regrid_2D[ii-1,0] ) * 111321 ) )  )
            Wind_Curl_f[tt,ii,-1] = (  ( ( Tau_Y[tt, ii,0] - Tau_Y[tt, ii,-2] ) / ( 2*7.292E-5 *  math.sin(math.radians( ( Lat_regrid_2D[ii,-1]))) ) ) /  np.absolute(  ( ( Lon_regrid_2D[ii,0] -  Lon_regrid_2D[ii,-2] ) * 111321  *  math.cos(math.radians( ( Lat_regrid_2D[ii,-1])))   )  )     )   -   (  ( ( Tau_X[tt, ii+1,-1] - Tau_X[tt, ii-1,-1] ) / ( 2*7.292E-5 *  math.sin(math.radians( ( Lat_regrid_2D[ii,-1]))) ) ) / np.absolute( ( ( Lat_regrid_2D[ii+1,-1] -  Lat_regrid_2D[ii-1,-1] ) * 111321 ) )  )

###############################################################################
Plot_Var = np.nanmean(Wind_Curl,axis=0) * 1E7
Plot_Var[ Ocean_Land_mask==0 ]=np.nan # masking over land, so grid cells that fall on land area (value=0) will be deleted
Plot_Var2 = np.nanmean(Tau_X,axis=0)
Plot_Var2 [ Ocean_Land_mask==0 ]=np.nan # masking over land, so grid cells that fall on land area (value=0) will be deleted

cmap_limit=np.nanmax(np.abs( np.nanpercentile(Plot_Var, 99)))
Plot_range=np.linspace(-cmap_limit,cmap_limit,27)
Plot_unit='(1E-7 N/m3)'; Plot_title= 'Wind Curl (1E-7 N/m3) - (contour lines = Tau_x) - '+str(year_start)+'-'+str(year_end)+' - '+str(GCM)

fig, m = func_plotmap_contourf(Plot_Var, Lon_regrid_2D, Lat_regrid_2D, Plot_range, Plot_title, Plot_unit, plt.cm.seismic, 'cyl', 210., 80., -80., '-')
im2=m.contour(Lon_regrid_2D, Lat_regrid_2D,Plot_Var2, 20, latlon=True, colors='k')
plt.clabel(im2, fontsize=8, inline=1)
plt.show()
fig.savefig(dir_pwd+'/'+'Fig_Wind_Curl_'+str(GCM)+'.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

###############################################################################
Lat_regrid_1D_4, Lon_regrid_1D_4, Lat_bound_regrid_4, Lon_bound_regrid_4 = func_latlon_regrid(45, 90, lat_min_regrid, lat_max_regrid, lon_min_regrid, lon_max_regrid)
Lon_regrid_2D_4, Lat_regrid_2D_4 = np.meshgrid(Lon_regrid_1D_4, Lat_regrid_1D_4)
Tau_X_4 = func_regrid(np.nanmean(Tau_X,axis=0), Lat_regrid_2D, Lon_regrid_2D, Lat_regrid_2D_4, Lon_regrid_2D_4)
Tau_Y_4 = func_regrid(np.nanmean(Tau_Y,axis=0), Lat_regrid_2D, Lon_regrid_2D, Lat_regrid_2D_4, Lon_regrid_2D_4)

Plot_Var_f = np.nanmean(Wind_Curl_f,axis=0) * 1E3

cmap_limit=np.nanmax(np.abs( np.nanpercentile(Plot_Var_f, 99)))
Plot_range=np.linspace(-cmap_limit,cmap_limit,27)
Plot_unit='(1E-3 N.S/m3.rad)'; Plot_title= 'Curl of (Wind/f) (Ekman upwelling) (1E-3 N.S/m3.rad) - '+str(year_start)+'-'+str(year_end)+' - '+str(GCM)+'\n(Arrows: wind direction) (contour line: Curl(wind/f)=0)'

fig, m = func_plotmap_contourf(Plot_Var_f, Lon_regrid_2D, Lat_regrid_2D, Plot_range, Plot_title, Plot_unit, plt.cm.seismic, 'cyl', 210., 80., -80., '-')
im2=m.quiver(Lon_regrid_2D_4, Lat_regrid_2D_4, Tau_X_4, Tau_Y_4, latlon=True, pivot='middle')
plt.show()
im3=m.contour(Lon_regrid_2D[25:50,:], Lat_regrid_2D[25:50,:],Plot_Var_f[25:50,:], levels = [0], latlon=True, colors='darkgreen')
plt.clabel(im3, fontsize=8, inline=1)
fig.savefig(dir_pwd+'/'+'Fig_Wind_Curl_f_WQuiver_'+str(GCM)+'.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


###############################################################
#### Average Arctic Sea Ice Concentration for each month  #####
###############################################################
dir_data_in1 = ('/data2/scratch/cabre/CMIP5/CMIP5_models/ocean_physics/') # Directory to raed raw data from
dir_data_in2=(dir_data_in1+ GCM + '/historical/mo/')

year_start=1991
year_end=2000

Var_name='sic' # The variable name to be read from .nc files
dset = xr.open_mfdataset(dir_data_in2+Var_name+'*12.nc')
Data_all = dset[Var_name]

Lat_orig = dset[lat_t]
Lat_orig = Lat_orig.values
Lon_orig = dset[lon_t]
Lon_orig = Lon_orig.values  

Data_monthly = Data_all[ (Data_all.time.dt.year >= year_start ) & (Data_all.time.dt.year <= year_end)]
Data_monthly = Data_monthly.values # Converts to a Numpy array
Data_monthly_ave = Data_monthly.reshape( np.int(Data_monthly.shape[0]/12) ,12,Data_monthly.shape[1],Data_monthly.shape[2])
Data_monthly_ave = np.nanmean(Data_monthly_ave,axis=0)
SIC_monthly = func_regrid(Data_monthly_ave, Lat_orig, Lon_orig, Lat_regrid_2D, Lon_regrid_2D)
SIC_monthly [SIC_monthly==0] = np.nan # To mask the ice-free ocean in the map

Time_months = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec');
n_r=3; n_c=4 ; n_t=12
fig=plt.figure()
for ii in range(0,n_t):
    ax = fig.add_subplot(n_r,n_c,ii+1)
    Plot_Var=SIC_monthly[ii,:,:]
    m = Basemap( projection='npstere',lon_0=0, boundinglat=30)
    if ii==0 or ii==n_c or ii==n_c*2 or ii==n_c*3 or ii==n_c*4 or ii==n_c*5 or ii==n_c*6 or ii==n_c*7 or ii==n_c*8:
        m.drawmeridians(np.arange(0,360,30), labels=[1,0,0,0], linewidth=1, color='k', fontsize=12) # labels = [left,right,top,bottom] # Longitudes
    elif ii == (n_c*(n_r-1)): # Adds longitude ranges only to the last subplots that appear at the bottom of plot
        m.drawmeridians(np.arange(0,360,30), labels=[1,0,0,1], linewidth=1, color='k', fontsize=12) # labels = [left,right,top,bottom] # Longitudes      
    elif ii >= n_t-n_c: # Adds longitude ranges only to the last subplots that appear at the bottom of plot
        m.drawmeridians(np.arange(0,360,30), labels=[0,0,0,1], linewidth=1, color='k', fontsize=12) # labels = [left,right,top,bottom] # Longitudes      
    else:
        m.drawmeridians(np.arange(0,360,30), labels=[0,0,0,0], linewidth=1, color='k', fontsize=12) # labels = [left,right,top,bottom] # Longitudes
    m.drawparallels(np.arange(-90,90,20), labels=[1,0,0,0], linewidth=1, color='k', fontsize=12) # labels = [left,right,top,bottom] # Latitutes  
    m.drawcoastlines(linewidth=1.0, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
    m.fillcontinents(color='0.8')
    im=m.contourf(Lon_regrid_2D, Lat_regrid_2D, Plot_Var,np.linspace(0,100,51) ,latlon=True, cmap=plt.cm.jet, extend='max')      
    plt.title(Time_months[ii])
plt.suptitle( ( 'Arctic monthly Sea Ice concentration - average of '+str(year_start)+'-'+str(year_end)+' - '+str(GCM)), fontsize=18)
plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.92, hspace=0.2, wspace=0.1) # the amount of height/width reserved for space between subplots
cbar_ax = fig.add_axes([0.87, 0.1, 0.015, 0.82]) # [right,bottom,width,height] 
fig.colorbar(im, cax=cbar_ax)
plt.show()
mng = plt.get_current_fig_manager()
mng.window.showMaximized() # Maximizes the plot window to save figures in full
fig.savefig(dir_pwd+'/'+'Fig_SeaIce_Arctic_monthly_'+str(GCM)+'.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
#plt.close()








































