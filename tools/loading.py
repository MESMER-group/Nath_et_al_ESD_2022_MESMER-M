#!/usr/bin/env python
# -*- coding: utf-8 -*-


#################
## Functions to process data
#################            
            
import numpy as np
import glob
#import geopandas # to read the shape files (there'd also be cartopy internal fct but there I'm lacking the knowledge)
#from cartopy.io.shapereader import Reader
#import fiona
import datetime
import copy
import mplotutils as mpu
import xarray as xr
import pandas as pd
import cf_units

from tools.gridding import norm_cos_wgt 

def load_data_obs(obs,Tref_start=1951,Tref_end=1980,Tblendglob_idx=False):
    """ Load the observations.

        Keyword argument:
        - obs: currently implemented for "best" and "cowtan"
        - Tref_start: starting point for the reference period with default 1951 (ie BEST Tref)
        - Tref_end: last year still INCLUDED for the reference period with default 1980 (ie BEST Tref) (ATTENTION: different from load_data_single_mod where is the first NOT included)
        - Tblend_idx: whether to compute the blended Tanglob anomaly or not (default = False)

        Output:
        - y: the land grid points of the anomalies of the variable on grid centered over 0 longitude (like the srexgrid) 
        - time: the time slots
        - Tblendglob = area weighted global mean temperature (blend from SST over ocean and tas over land + sea ice)

    """
    dir_data = '/net/cfc/landclim1/beuschl/magicc_plus/data/'
    
    # read in the land-sea mask
    file_ls = 'interim_invariant_lsmask_regrid.nc' # ERA-interim mask regridded by Richard from 73x144 to 72x144
    frac_l = xr.open_mfdataset(dir_data+file_ls) #land-sea mask of ERA-interim bilinearily interpolated 
    frac_l = frac_l.where(frac_l.lat>-60,0) # remove Antarctica from frac_l field (ie set frac l to 0)
    idx_l=np.squeeze(frac_l.lsm.values)>0.0 # idex land #-> everything >0 I consider land

    lons, lats = np.meshgrid(frac_l.lon.values,frac_l.lat.values) # the lon, lat grid (just to derive weights)   
    wgt = norm_cos_wgt(lats) # area weights of each grid point

    if obs == 'best':
        obs_file='best/best_yearmean_ann_g025.nc' # ATTENTION: yearmean was done because I was not able to make yearmonmean
            # work (however, impact really seems negligible small)
        ds_obs=xr.open_mfdataset(dir_data+obs_file).rename({'temperature':'tas'}).sel(time=slice(1870, 2018))
        ds_obs.time.values=np.asarray(ds_obs.time.values,dtype=int)

    if obs == 'cowtan':
        obs_file='cowtan_way/had4sst4_krig_ann_g025.nc'
        ds_obs=xr.open_mfdataset(dir_data+obs_file).rename({'temperature_anomaly':'tas'}).sel(time=slice('1870', '2018'))
        if len(ds_obs.time.values)==149:
            ds_obs.time.values=np.arange(1870,2019)
        
        
    T_ref = ds_obs.tas.sel(time=slice(Tref_start, Tref_end)).mean(dim='time')
    if Tblendglob_idx == True:
        tas=ds_obs.tas.values-T_ref.values # anomalies, ocean included

        Tblendglob=np.zeros(tas.shape[0])
        for t in np.arange(tas.shape[0]):
            idx_valid = ~np.isnan(tas[t])
            Tblendglob[t] = np.average(tas[t,idx_valid],weights=wgt[idx_valid]) #area weighted of available obs -> less data available at beginning        

    y=(ds_obs.tas.values-T_ref.values)[:,idx_l]
    time=ds_obs.time.values
    if Tblendglob_idx==True:
        return y,time,Tblendglob
    else:
        return y,time


def load_data_single_mod(gen,model,scenario,Tanglob_idx=False,Tref_all=True,Tref_start='1870-01-01',Tref_end='1900-01-01',usr_time_res="ann"):
	""" Load the all initial-condition members of a single model in cmip5 or cmip6 for given scenario plus associated historical period.

		Keyword argument:
		- gen: generation (cmip5 = 5 and cmip6 = 6 are implemented)
		- model: model str
		- scenario: scenario str
		- Tanglob_idx: decides if wgt Tanglob is computed (and returned) or not, default is not returned
		- Tref_all: decides if the Tref at each grid point is dervied based on all available runs or not, default is yes       
		- Tref_start: starting point for the reference period with default 1870
		- Tref_end: first year to no longer be included in reference period with default 1900

		Output:
		- y: the land grid points of the anomalies of the variable on grid centered over 0 longitude (like the srexgrid) 
		- time: the time slots
		- srex: the gridded srex regions
		- df_srex: data frame containing the shape files of the srex regions
		- lon_pc: longitudes for pcolormesh (needs 1 more than on grid)
		- lat_pc: latitudes for pcolormesh (needs 1 more than on grid)
		- idx_l: array with 0 where sea, 1 where land (assumption: land if frac land > 0)
		- wgt_l: land grid point weights to derive area weighted mean vars
		- Tan_wgt_globmean = area weighted global mean temperature

	"""
    # the dictionaries are NOT ordered properly + some other adjustments -> will need to be careful with my old scripts

    # see e-mail from Verena on 20191112 for additional infos how could read in several files at once with xarr
    # additionally: she transforms dataset into dataarray to make indexing easier -> for consistency reason with earlier
        # version of emulator (& thus to be able to reuse my scripts), I do not do this (fow now).
    
	# right now I keep reloading constants fields for each run I add -> does not really make sense. 
    # Maybe add boolean to decide instead. however they are small & I have to read them in at some point anyways
    # -> maybe path of least resistence is to not care about it
	print('start with model',model)

	# vars which used to be part of the inputs but did not really make sense as I employ the same ones all the time anyways (could be changed later if needed)
	var='tas'
	temp_res = usr_time_res # if not, reading the var file needs to be changed as time var is not named in the same way anymore
	spatial_res = 'g025'


    # load in the constants files
	dir_data = '/net/so4/landclim/snath/data/'
	file_ls = 'interim_invariant_lsmask_regrid.nc' # ERA-interim mask regridded by Richard from 73x144 to 72x144
	file_srex = 'srex-region-masks_20120709.srex_mask_SREX_masks_all.25deg.time-invariant.nc'
	file_srex_shape = 'referenceRegions.shp'


	#df_all_regs = geopandas.read_file(dir_data+file_srex_shape)
	srex_names = ['ALA','CGI','WNA','CNA','ENA','CAM','AMZ','NEB','WSA','SSA','NEU','CEU','MED','SAH','WAF','EAF','SAF',
             'NAS','WAS','CAS','TIB','EAS','SAS','SEA','NAU','SAU'] # SREX names ordered according to SREX mask I am 
                    # employing
	#df_srex = df_all_regs.loc[df_all_regs['LAB'].isin(srex_names)] # alternative indexing: search in column LAB for names
	srex_raw = xr.open_mfdataset(dir_data+file_srex, combine='by_coords',decode_times=False) # srex_raw nrs from 1-26
	#df_srex=srex_raw 
	#srex_raw["time"]=pd.to_datetime(srex_raw.time.values)
	lons, lats = np.meshgrid(srex_raw.lon.values,srex_raw.lat.values) # the lon, lat grid (just to derive weights)    
    
	frac_l = xr.open_mfdataset(dir_data+file_ls, combine='by_coords',decode_times=False) #land-sea mask of ERA-interim bilinearily interpolated 
	frac_l_raw = np.squeeze(copy.deepcopy(frac_l.lsm.values))
	#frac_1["time"]=pd.to_datetime(frac_1.time.values)
	frac_l = frac_l.where(frac_l.lat>-60,0) # remove Antarctica from frac_l field (ie set frac l to 0)

	idx_l=np.squeeze(frac_l.lsm.values)>0.0 # idex land #-> everything >0 I consider land
 

	wgt = norm_cos_wgt(lats) # area weights of each grid point
	wgt_l = (wgt*frac_l_raw)[idx_l] # area weights for land grid points (including taking fraction land into consideration)
    #wgt_l = wgt[idx_l] # area weights for land grid points
	lon_pc, lat_pc = mpu.infer_interval_breaks(frac_l.lon, frac_l.lat) # the lon / lat for the plotting with pcolormesh
	srex=(np.squeeze(srex_raw.srex_mask.values)-1)[idx_l] # srex indices on land

    
	y={}
	T_ref = np.zeros(idx_l.shape)
	run_nrs={}
	if Tanglob_idx == True:
		Tan_wgt_globmean = {}
	if gen == 5:
		dir_var='/net/atmos/data/cmip5-ng/tas/' 
		run_names_list=sorted(glob.glob(dir_var+var+'_'+temp_res+'_'+model+'_'+scenario+'_'+'r*i1p1'+'_'+spatial_res+'.nc'))
        # ATTENTION: are ordered but does not work for models with runs above digit 9 
		index_tr = [i for i, s in enumerate(run_names_list) if 'r1i1p1' in s][0] # find training run 
		#print(run_names_list.pop(index_tr))
		run_names_list.insert(0, run_names_list.pop(index_tr)) # move training run to begin of list

        # exception for cmip5 GISS-E2-H_rcp85_r2i1p1 ie the strange run I excluded from ESD paper
		if '/net/atmos/data/cmip5-ng/tas/tas_%s_GISS-E2-H_rcp85_r2i1p1_g025.nc'%usr_time_res in run_names_list:
			run_names_list.remove('/net/atmos/data/cmip5-ng/tas/tas_%s_GISS-E2-H_rcp85_r2i1p1_g025.nc'%usr_time_res )
            
        # loop over all runs to obtain the absolute values  
		print(run_names_list)
		for run_name in run_names_list:

			data = xr.open_mfdataset(run_name,decode_times=False)
			if usr_time_res=="ann":
				data=data.rename({'year':'time'})
			data["time"]=cf_units.num2date(data.time.values, 'days since 1800-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
			data=data.sel(time=slice('1870-01-01', '2101-01-01')).roll(lon=72)
            # rename so it is consisten with cmip6 
            # roll so that it is on same grid as others (no longer Pacific centered) 

			#print(data.time.values)
			data = data.assign_coords(lon= (((data.lon + 180) % 360) - 180)) # assign_coords so same labels as others
			run=int(data.attrs['source_ensemble'].split('r')[1].split('i')[0]) # extract ens member
			run_nrs[run_name]=run
           
			y[run] = data.tas.values # still absolute values + still contains sea pixels
			T_ref += data.tas.sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values*1.0/len(run_names_list) # sum up all ref climates
                
            
			if run==1 and Tref_all != True:
				T_ref_1=data.tas.sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values
				print('create ref for ',run_name)
            
			if Tanglob_idx == True:
				Tan_wgt_globmean[run] = np.asarray([np.average(y[run][t],weights=wgt) for t in np.arange(y[run].shape[0])]) #area weighted but abs values          
            
                  
	if gen == 6:
		dir_var = '/net/atmos/data/cmip6-ng/tas/%s/g025/'%usr_time_res#'/net/cfc/cmip6/Next_Generation/tas/' #<- switch once stable
		run_names_list=sorted(glob.glob(dir_var+var+'_'+temp_res+'_'+model+'_'+scenario+'_'+'r*i1p1f*'+'_'+spatial_res+'.nc'))
            # ATTENTION:  are ordered but does not work for models with runs above digit 9
            # idea is: every ssp one needs a corresponding hist one (vice versa not the case)
		if scenario=='ssp119' and model=='EC-Earth3':
			index_tr = [i for i, s in enumerate(run_names_list) if 'r4i1p1' in s][0] # find training run 
			run_names_list.insert(0, run_names_list.pop(index_tr)) # move training run to begin of list
		else:
			index_tr = [i for i, s in enumerate(run_names_list) if 'r1i1p1' in s][0] # find training run 
			run_names_list.insert(0, run_names_list.pop(index_tr)) # move training run to begin of list
            

		if model=='CESM2-WACCM':
			run_names_list.remove('/net/atmos/data/cmip6-ng/tas/%s/g025/tas_%s_CESM2-WACCM_ssp585_r4i1p1f1_g025.nc'%(usr_time_res,usr_time_res))
			run_names_list.remove('/net/atmos/data/cmip6-ng/tas/%s/g025/tas_%s_CESM2-WACCM_ssp585_r5i1p1f1_g025.nc'%(usr_time_res,usr_time_res))
		if model=='EC-Earth3' and scenario!='ssp119':
			run_names_list=[i for i in run_names_list if len(list(i.split('/')[-1].split('_')[-2].split('i')[0]))!=4]

		for run_name in run_names_list:
			run_name_ssp = run_name
			if scenario=='ssp119' and model=='EC-Earth3':
				run_name_hist = run_names_list[0].replace(scenario,'historical')
			else:
				run_name_hist = run_name.replace(scenario,'historical')
			data = xr.open_mfdataset([run_name_hist,run_name_ssp],concat_dim='time').sel(time=slice('1870-01-01', '2101-01-01')).roll(lon=72)
			data = data.assign_coords(lon= (((data.lon + 180) % 360) - 180))  # assign_coords so same labels as others
			if scenario=='ssp119' and model=='EC-Earth3':
				#print(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')) 
				run = int(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')[0]) 
				run_nrs[run_name] = run
			else:
				run = data.attrs['realization_index']
				run_nrs[run_name]=run
           
			y[run] = data.tas.values # still absolute values + still contains sea pixels
			T_ref += data.tas.sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values*1.0/len(run_names_list) # sum up all ref climates
 
			if run==1 and Tref_all != True:
				T_ref_1=data.tas.sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values
				print('create ref for ',run_name)       

			if model=='CAMS-CSM1-0': # temporary (?) fix added on 20191119 because CAMS-CSM1-0 data are currently only available till 2099 instead of 2100
				y[run]=y[run][:-1]
                
			if Tanglob_idx == True:
				Tan_wgt_globmean[run] = np.asarray([np.average(y[run][t],weights=wgt) for t in np.arange(y[run].shape[0])]) #area weighted but abs values          

       # obtain the anomalies
	for run_name in run_names_list:
		run = run_nrs[run_name]
		if Tref_all == True:
			try:
				y[run]=(y[run]-T_ref)[:,idx_l]
			except: 
				y[run]=(y[run]-T_ref[idx_l]) 
				print('exception dealt with, ', y.keys(),y[run].shape)
			if Tanglob_idx == True:
				Tan_wgt_globmean[run]=Tan_wgt_globmean[run]-np.average(T_ref,weights=wgt)
		else:
			y[run]=y[run][:,idx_l]#-T_ref_1)[:,idx_l]
			if Tanglob_idx == True:
				Tan_wgt_globmean[run]=Tan_wgt_globmean[run]-np.average(T_ref_1,weights=wgt)                
    
                
                
	if (data.lon!=srex_raw.lon).any() and (srex_raw.lon!=frac_l.lon).any():
		print('There is an error. The grids do not agree.')
	time=data["time"]
	if y[next(iter(y))].shape[0]==231: #hardcoded way to have a mini-check whether still the right amount of time slots  #next iter thing needed because of strange models with no r1 run
		time = np.arange(1870,2101)      
	elif y[next(iter(y))].shape[0]==230 and model=='CAMS-CSM1-0':
		time = np.arange(1870,2100)
		print('ATTENTION: runs go only until 2099 instead of 2100 because last time step not available on 20191119')
	else:
		print('There is an error. The selected time frame no longer corresponds to the hardcoded time vector.')
	if Tanglob_idx == False:  
		if Tref_all == False:
			return y,time,srex,srex_names,lon_pc,lat_pc,idx_l,wgt_l, T_ref #df_srex,
		else:
			return y,time,srex,srex_names,lon_pc,lat_pc,idx_l,wgt_l#df_srex,            
	else:
		return y,time,srex,srex_names,lon_pc,lat_pc,idx_l,wgt_l,Tan_wgt_globmean #df_srex,

def load_data_single_mod_var(gen,model,scenario,Tanglob_idx=False,Tref_all=True,Tref_start='1870-01-01',Tref_end='1900-01-01',usr_time_res="ann",var="ts"):
	""" Load the all initial-condition members of a single model in cmip5 or cmip6 for given scenario plus associated historical period.

		Keyword argument:
		- gen: generation (cmip5 = 5 and cmip6 = 6 are implemented)
		- model: model str
		- scenario: scenario str
		- Tanglob_idx: decides if wgt Tanglob is computed (and returned) or not, default is not returned
		- Tref_all: decides if the Tref at each grid point is dervied based on all available runs or not, default is yes       
		- Tref_start: starting point for the reference period with default 1870
		- Tref_end: first year to no longer be included in reference period with default 1900

		Output:
		- y: the land grid points of the anomalies of the variable on grid centered over 0 longitude (like the srexgrid) 
		- time: the time slots
		- srex: the gridded srex regions
		- df_srex: data frame containing the shape files of the srex regions
		- lon_pc: longitudes for pcolormesh (needs 1 more than on grid)
		- lat_pc: latitudes for pcolormesh (needs 1 more than on grid)
		- idx_l: array with 0 where sea, 1 where land (assumption: land if frac land > 0)
		- wgt_l: land grid point weights to derive area weighted mean vars
		- Tan_wgt_globmean = area weighted global mean temperature

	"""
    # the dictionaries are NOT ordered properly + some other adjustments -> will need to be careful with my old scripts

    # see e-mail from Verena on 20191112 for additional infos how could read in several files at once with xarr
    # additionally: she transforms dataset into dataarray to make indexing easier -> for consistency reason with earlier
        # version of emulator (& thus to be able to reuse my scripts), I do not do this (fow now).
    
	# right now I keep reloading constants fields for each run I add -> does not really make sense. 
    # Maybe add boolean to decide instead. however they are small & I have to read them in at some point anyways
    # -> maybe path of least resistence is to not care about it
	print('start with model',model)

	# vars which used to be part of the inputs but did not really make sense as I employ the same ones all the time anyways (could be changed later if needed)
	temp_res = usr_time_res # if not, reading the var file needs to be changed as time var is not named in the same way anymore
	spatial_res = 'g025'


    # load in the constants files
	dir_data = '/net/so4/landclim/snath/data/'
	file_ls = 'interim_invariant_lsmask_regrid.nc' # ERA-interim mask regridded by Richard from 73x144 to 72x144
	file_srex = 'srex-region-masks_20120709.srex_mask_SREX_masks_all.25deg.time-invariant.nc'
	file_srex_shape = 'referenceRegions.shp'


	#df_all_regs = geopandas.read_file(dir_data+file_srex_shape)
	srex_names = ['ALA','CGI','WNA','CNA','ENA','CAM','AMZ','NEB','WSA','SSA','NEU','CEU','MED','SAH','WAF','EAF','SAF',
             'NAS','WAS','CAS','TIB','EAS','SAS','SEA','NAU','SAU'] # SREX names ordered according to SREX mask I am 
                    # employing
	#df_srex = df_all_regs.loc[df_all_regs['LAB'].isin(srex_names)] # alternative indexing: search in column LAB for names
	srex_raw = xr.open_mfdataset(dir_data+file_srex, combine='by_coords',decode_times=False) # srex_raw nrs from 1-26
	#df_srex=srex_raw 
	#srex_raw["time"]=pd.to_datetime(srex_raw.time.values)
	lons, lats = np.meshgrid(srex_raw.lon.values,srex_raw.lat.values) # the lon, lat grid (just to derive weights)    
    
	frac_l = xr.open_mfdataset(dir_data+file_ls, combine='by_coords',decode_times=False) #land-sea mask of ERA-interim bilinearily interpolated 
	frac_l_raw = np.squeeze(copy.deepcopy(frac_l.lsm.values))
	#frac_1["time"]=pd.to_datetime(frac_1.time.values)
	frac_l = frac_l.where(frac_l.lat>-60,0) # remove Antarctica from frac_l field (ie set frac l to 0)

	idx_l=np.squeeze(frac_l.lsm.values)>0.0 # idex land #-> everything >0 I consider land
 

	wgt = norm_cos_wgt(lats) # area weights of each grid point
	wgt_l = (wgt*frac_l_raw)[idx_l] # area weights for land grid points (including taking fraction land into consideration)
    #wgt_l = wgt[idx_l] # area weights for land grid points
	lon_pc, lat_pc = mpu.infer_interval_breaks(frac_l.lon, frac_l.lat) # the lon / lat for the plotting with pcolormesh
	srex=(np.squeeze(srex_raw.srex_mask.values)-1)[idx_l] # srex indices on land

    
	y={}
	T_ref = np.zeros(idx_l.shape)
	run_nrs={}
	if Tanglob_idx == True:
		Tan_wgt_globmean = {}
	if gen == 5:
		dir_var='/net/atmos/data/cmip5-ng/tas/' 
		run_names_list=sorted(glob.glob(dir_var+var+'_'+temp_res+'_'+model+'_'+scenario+'_'+'r*i1p1'+'_'+spatial_res+'.nc'))
        # ATTENTION: are ordered but does not work for models with runs above digit 9 
		index_tr = [i for i, s in enumerate(run_names_list) if 'r1i1p1' in s][0] # find training run 
		#print(run_names_list.pop(index_tr))
		run_names_list.insert(0, run_names_list.pop(index_tr)) # move training run to begin of list

        # exception for cmip5 GISS-E2-H_rcp85_r2i1p1 ie the strange run I excluded from ESD paper
		if '/net/atmos/data/cmip5-ng/tas/tas_%s_GISS-E2-H_rcp85_r2i1p1_g025.nc'%usr_time_res in run_names_list:
			run_names_list.remove('/net/atmos/data/cmip5-ng/tas/tas_%s_GISS-E2-H_rcp85_r2i1p1_g025.nc'%usr_time_res )
            
        # loop over all runs to obtain the absolute values  
		print(run_names_list)
		for run_name in run_names_list:

			data = xr.open_mfdataset(run_name,decode_times=False)
			if usr_time_res=="ann":
				data=data.rename({'year':'time'})
			data["time"]=cf_units.num2date(data.time.values, 'days since 1800-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
			data=data.sel(time=slice('1870-01-01', '2101-01-01')).roll(lon=72)
            # rename so it is consisten with cmip6 
            # roll so that it is on same grid as others (no longer Pacific centered) 

			#print(data.time.values)
			data = data.assign_coords(lon= (((data.lon + 180) % 360) - 180)) # assign_coords so same labels as others
			run=int(data.attrs['source_ensemble'].split('r')[1].split('i')[0]) # extract ens member
			run_nrs[run_name]=run
           
			y[run] = data[var].values # still absolute values + still contains sea pixels
			T_ref += data[var].sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values*1.0/len(run_names_list) # sum up all ref climates
                
            
			if run==1 and Tref_all != True:
				T_ref_1=data[var].sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values
				print('create ref for ',run_name)
            
			if Tanglob_idx == True:
				Tan_wgt_globmean[run] = np.asarray([np.average(y[run][t],weights=wgt) for t in np.arange(y[run].shape[0])]) #area weighted but abs values          
            
                  
	if gen == 6:
		dir_var = '/net/atmos/data/cmip6-ng/%s/%s/g025/'%(var,usr_time_res)#'/net/cfc/cmip6/Next_Generation/tas/' #<- switch once stable
		run_names_list=sorted(glob.glob(dir_var+var+'_'+temp_res+'_'+model+'_'+scenario+'_'+'r*i1p1f*'+'_'+spatial_res+'.nc'))
            # ATTENTION:  are ordered but does not work for models with runs above digit 9
            # idea is: every ssp one needs a corresponding hist one (vice versa not the case)
		if scenario=='ssp119' and model=='EC-Earth3':
			index_tr = [i for i, s in enumerate(run_names_list) if 'r4i1p1' in s][0] # find training run 
			run_names_list.insert(0, run_names_list.pop(index_tr)) # move training run to begin of list
		else:
			index_tr = [i for i, s in enumerate(run_names_list) if 'r1i1p1' in s][0] # find training run 
			run_names_list.insert(0, run_names_list.pop(index_tr)) # move training run to begin of list
            

		if model=='CESM2-WACCM':
			run_names_list.remove('/net/atmos/data/cmip6-ng/%s/%s/g025/%s_%s_CESM2-WACCM_ssp585_r4i1p1f1_g025.nc'%(var,usr_time_res,var,usr_time_res))
			run_names_list.remove('/net/atmos/data/cmip6-ng/%s/%s/g025/%s_%s_CESM2-WACCM_ssp585_r5i1p1f1_g025.nc'%(var,usr_time_res,var,usr_time_res))
		if model=='EC-Earth3' and scenario!='ssp119':
			run_names_list=[i for i in run_names_list if len(list(i.split('/')[-1].split('_')[-2].split('i')[0]))!=4]

		for run_name in run_names_list:
			data = xr.open_mfdataset(run_name,concat_dim='time').sel(time=slice('1870-01-01', '2101-01-01')).roll(lon=72)
			data = data.assign_coords(lon= (((data.lon + 180) % 360) - 180))  # assign_coords so same labels as others
			if scenario=='ssp119' and model=='EC-Earth3':
				#print(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')) 
				run = int(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')[0]) 
				run_nrs[run_name] = run
			else:
				run = data.attrs['realization_index']
				run_nrs[run_name]=run
           
			y[run] = data[var].values # still absolute values + still contains sea pixels
			T_ref += data[var].sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values*1.0/len(run_names_list) # sum up all ref climates
 
			if run==1 and Tref_all != True:
				T_ref_1=data[var].sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values
				print('create ref for ',run_name)       

			if model=='CAMS-CSM1-0': # temporary (?) fix added on 20191119 because CAMS-CSM1-0 data are currently only available till 2099 instead of 2100
				y[run]=y[run][:-1]
                
			if Tanglob_idx == True:
				Tan_wgt_globmean[run] = np.asarray([np.average(y[run][t],weights=wgt) for t in np.arange(y[run].shape[0])]) #area weighted but abs values          

       # obtain the anomalies
	for run_name in run_names_list:
		run = run_nrs[run_name]
		if Tref_all == True:
			try:
				y[run]=(y[run]-T_ref)[:,idx_l]
			except: 
				y[run]=(y[run]-T_ref[idx_l]) 
				print('exception dealt with, ', y.keys(),y[run].shape)
			if Tanglob_idx == True:
				Tan_wgt_globmean[run]=Tan_wgt_globmean[run]-np.average(T_ref,weights=wgt)
		else:
			y[run]=y[run][:,idx_l]#-T_ref_1)[:,idx_l]
			if Tanglob_idx == True:
				Tan_wgt_globmean[run]=Tan_wgt_globmean[run]-np.average(T_ref_1,weights=wgt)                
    
                
                
	if (data.lon!=srex_raw.lon).any() and (srex_raw.lon!=frac_l.lon).any():
		print('There is an error. The grids do not agree.')
	time=data["time"]
	if y[next(iter(y))].shape[0]==231: #hardcoded way to have a mini-check whether still the right amount of time slots  #next iter thing needed because of strange models with no r1 run
		time = np.arange(1870,2101)      
	elif y[next(iter(y))].shape[0]==230 and model=='CAMS-CSM1-0':
		time = np.arange(1870,2100)
		print('ATTENTION: runs go only until 2099 instead of 2100 because last time step not available on 20191119')
	else:
		print('There is an error. The selected time frame no longer corresponds to the hardcoded time vector.')
	if Tanglob_idx == False:  
		if Tref_all == False:
			return y,time,srex,srex_names,lon_pc,lat_pc,idx_l,wgt_l, T_ref #df_srex,
		else:
			return y,time,srex,srex_names,lon_pc,lat_pc,idx_l,wgt_l#df_srex,            
	else:
		return y,time,srex,srex_names,lon_pc,lat_pc,idx_l,wgt_l,Tan_wgt_globmean #df_srex,
        
def load_data_single_mod_rh(gen,model,scenario,Tanglob_idx=False,Tref_all=True,Tref_start='1870-01-01',Tref_end='1900-01-01',usr_time_res="ann"):
	""" Load the all initial-condition members of a single model in cmip5 or cmip6 for given scenario plus associated historical period.

		Keyword argument:
		- gen: generation (cmip5 = 5 and cmip6 = 6 are implemented)
		- model: model str
		- scenario: scenario str
		- Tanglob_idx: decides if wgt Tanglob is computed (and returned) or not, default is not returned
		- Tref_all: decides if the Tref at each grid point is dervied based on all available runs or not, default is yes       
		- Tref_start: starting point for the reference period with default 1870
		- Tref_end: first year to no longer be included in reference period with default 1900

		Output:
		- y: the land grid points of the anomalies of the variable on grid centered over 0 longitude (like the srexgrid) 
		- time: the time slots
		- srex: the gridded srex regions
		- df_srex: data frame containing the shape files of the srex regions
		- lon_pc: longitudes for pcolormesh (needs 1 more than on grid)
		- lat_pc: latitudes for pcolormesh (needs 1 more than on grid)
		- idx_l: array with 0 where sea, 1 where land (assumption: land if frac land > 0)
		- wgt_l: land grid point weights to derive area weighted mean vars
		- Tan_wgt_globmean = area weighted global mean temperature

	"""
    # the dictionaries are NOT ordered properly + some other adjustments -> will need to be careful with my old scripts

    # see e-mail from Verena on 20191112 for additional infos how could read in several files at once with xarr
    # additionally: she transforms dataset into dataarray to make indexing easier -> for consistency reason with earlier
        # version of emulator (& thus to be able to reuse my scripts), I do not do this (fow now).
    
	# right now I keep reloading constants fields for each run I add -> does not really make sense. 
    # Maybe add boolean to decide instead. however they are small & I have to read them in at some point anyways
    # -> maybe path of least resistence is to not care about it
	print('start with model',model)

	# vars which used to be part of the inputs but did not really make sense as I employ the same ones all the time anyways (could be changed later if needed)
	var='hurs'
	temp_res = usr_time_res # if not, reading the var file needs to be changed as time var is not named in the same way anymore
	spatial_res = 'g025'


    # load in the constants files
	dir_data = '/net/so4/landclim/snath/data/'
	file_ls = 'interim_invariant_lsmask_regrid.nc' # ERA-interim mask regridded by Richard from 73x144 to 72x144
	file_srex = 'srex-region-masks_20120709.srex_mask_SREX_masks_all.25deg.time-invariant.nc'
	file_srex_shape = 'referenceRegions.shp'


	#df_all_regs = geopandas.read_file(dir_data+file_srex_shape)
	srex_names = ['ALA','CGI','WNA','CNA','ENA','CAM','AMZ','NEB','WSA','SSA','NEU','CEU','MED','SAH','WAF','EAF','SAF',
             'NAS','WAS','CAS','TIB','EAS','SAS','SEA','NAU','SAU'] # SREX names ordered according to SREX mask I am 
                    # employing
	#df_srex = df_all_regs.loc[df_all_regs['LAB'].isin(srex_names)] # alternative indexing: search in column LAB for names
	srex_raw = xr.open_mfdataset(dir_data+file_srex, combine='by_coords',decode_times=False) # srex_raw nrs from 1-26
	#df_srex=srex_raw 
	#srex_raw["time"]=pd.to_datetime(srex_raw.time.values)
	lons, lats = np.meshgrid(srex_raw.lon.values,srex_raw.lat.values) # the lon, lat grid (just to derive weights)    
    
	frac_l = xr.open_mfdataset(dir_data+file_ls, combine='by_coords',decode_times=False) #land-sea mask of ERA-interim bilinearily interpolated 
	frac_l_raw = np.squeeze(copy.deepcopy(frac_l.lsm.values))
	#frac_1["time"]=pd.to_datetime(frac_1.time.values)
	frac_l = frac_l.where(frac_l.lat>-60,0) # remove Antarctica from frac_l field (ie set frac l to 0)

	idx_l=np.squeeze(frac_l.lsm.values)>0.0 # idex land #-> everything >0 I consider land
 

	wgt = norm_cos_wgt(lats) # area weights of each grid point
	wgt_l = (wgt*frac_l_raw)[idx_l] # area weights for land grid points (including taking fraction land into consideration)
    #wgt_l = wgt[idx_l] # area weights for land grid points
	lon_pc, lat_pc = mpu.infer_interval_breaks(frac_l.lon, frac_l.lat) # the lon / lat for the plotting with pcolormesh
	srex=(np.squeeze(srex_raw.srex_mask.values)-1)[idx_l] # srex indices on land

    
	y={}
	RH_ref = np.zeros(idx_l.shape)
	run_nrs={}
	if Tanglob_idx == True:
		Tan_wgt_globmean = {}
	if gen == 5:
		dir_var='/net/atmos/data/cmip5-ng/hurs/' 
		run_names_list=sorted(glob.glob(dir_var+var+'_'+temp_res+'_'+model+'_'+scenario+'_'+'r*i1p1'+'_'+spatial_res+'.nc'))
        # ATTENTION: are ordered but does not work for models with runs above digit 9 
		index_tr = [i for i, s in enumerate(run_names_list) if 'r1i1p1' in s][0] # find training run 
		#print(run_names_list.pop(index_tr))
		run_names_list.insert(0, run_names_list.pop(index_tr)) # move training run to begin of list

        # exception for cmip5 GISS-E2-H_rcp85_r2i1p1 ie the strange run I excluded from ESD paper
		if '/net/atmos/data/cmip5-ng/hurs/hurs_%s_GISS-E2-H_rcp85_r2i1p1_g025.nc'%usr_time_res in run_names_list:
			run_names_list.remove('/net/atmos/data/cmip5-ng/hurs/hurs_%s_GISS-E2-H_rcp85_r2i1p1_g025.nc'%usr_time_res )
            
        # loop over all runs to obtain the absolute values  
		print(run_names_list)
		for run_name in run_names_list:

			data = xr.open_mfdataset(run_name,decode_times=False)
			if usr_time_res=="ann":
				data=data.rename({'year':'time'})
			data["time"]=cf_units.num2date(data.time.values, 'days since 1800-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
			data=data.sel(time=slice('1870-01-01', '2101-01-01')).roll(lon=72)
            # rename so it is consisten with cmip6 
            # roll so that it is on same grid as others (no longer Pacific centered) 

			#print(data.time.values)
			data = data.assign_coords(lon= (((data.lon + 180) % 360) - 180)) # assign_coords so same labels as others
			if scenario=='ssp119' and model=='EC-Earth3':
				#print(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')) 
				run = int(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')[0]) 
				run_nrs[run_name] = run
			else:
				run = data.attrs['realization_index']
				run_nrs[run_name]=run
           
			y[run] = data.hurs.values # still absolute values + still contains sea pixels
			RH_ref += data.hurs.sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values*1.0/len(run_names_list) # sum up all ref climates
                
            
			if run==1 and Tref_all != True:
				RH_ref_1=data.hurs.sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values
				print('create ref for ',run_name)
            
			if Tanglob_idx == True:
				Tan_wgt_globmean[run] = np.asarray([np.average(y[run][t],weights=wgt) for t in np.arange(y[run].shape[0])]) #area weighted but abs values          
                               
	if gen == 6:
		dir_var = '/net/atmos/data/cmip6-ng/hurs/%s/g025/'%usr_time_res#'/net/cfc/cmip6/Next_Generation/tas/' #<- switch once stable
		run_names_list=sorted(glob.glob(dir_var+var+'_'+temp_res+'_'+model+'_'+scenario+'_'+'r*i1p1f*'+'_'+spatial_res+'.nc'))
            # ATTENTION:  are ordered but does not work for models with runs above digit 9
            # idea is: every ssp one needs a corresponding hist one (vice versa not the case)

		if scenario=='ssp119' and model=='EC-Earth3':
			print(run_names_list) 
			index_tr = [i for i, s in enumerate(run_names_list) if 'r4i1p1' in s][0] # find training run 
			run_names_list.insert(0, run_names_list.pop(index_tr)) # move training run to begin of list
		else:
			index_tr = [i for i, s in enumerate(run_names_list) if 'r1i1p1' in s][0] # find training run 
			run_names_list.insert(0, run_names_list.pop(index_tr)) # move training run to begin of list
            

		if model=='CESM2-WACCM':
			run_names_list.remove('/net/atmos/data/cmip6-ng/hurs/%s/g025/hurs_%s_CESM2-WACCM_ssp585_r4i1p1f1_g025.nc'%(usr_time_res,usr_time_res))
			run_names_list.remove('/net/atmos/data/cmip6-ng/hurs/%s/g025/hurs_%s_CESM2-WACCM_ssp585_r5i1p1f1_g025.nc'%(usr_time_res,usr_time_res))
		if model=='EC-Earth3' and scenario!='ssp119':
			run_names_list=[i for i in run_names_list if len(list(i.split('/')[-1].split('_')[-2].split('i')[0]))!=4]

		for run_name in run_names_list:
			run_name_ssp = run_name
			if scenario=='ssp119' and model=='EC-Earth3':
				run_name_hist = run_names_list[0].replace(scenario,'historical')
			else:
				run_name_hist = run_name.replace(scenario,'historical')
			data = xr.open_mfdataset([run_name_hist,run_name_ssp],concat_dim='time').sel(time=slice('1870-01-01', '2101-01-01')).roll(lon=72)
			data = data.assign_coords(lon= (((data.lon + 180) % 360) - 180))  # assign_coords so same labels as others
			if scenario=='ssp119' and model=='EC-Earth3':
				#print(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')) 
				run = int(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')[0]) 
				run_nrs[run_name] = run
			else:
				run = data.attrs['realization_index']
				run_nrs[run_name]=run
           
			y[run] = data.hurs.values # still absolute values + still contains sea pixels
			RH_ref += data.hurs.sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values*1.0/len(run_names_list) # sum up all ref climates
 
			if run==1 and Tref_all != True:
				RH_ref_1=data.hurs.sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values
				print('create ref for ',run_name)       

			if model=='CAMS-CSM1-0': # temporary (?) fix added on 20191119 because CAMS-CSM1-0 data are currently only available till 2099 instead of 2100
				y[run]=y[run][:-1]
                
			if Tanglob_idx == True:
				Tan_wgt_globmean[run] = np.asarray([np.average(y[run][t],weights=wgt) for t in np.arange(y[run].shape[0])]) #area weighted but abs values          

       # obtain the anomalies
	for run_name in run_names_list:
		run = run_nrs[run_name]
		if Tref_all == True:
			y[run]=(y[run]-RH_ref)[:,idx_l]
			if Tanglob_idx == True:
				Tan_wgt_globmean[run]=Tan_wgt_globmean[run]-np.average(RH_ref,weights=wgt)
		else:
			y[run]=y[run][:,idx_l]#-RH_ref_1)[:,idx_l]
			if Tanglob_idx == True:
				Tan_wgt_globmean[run]=Tan_wgt_globmean[run]-np.average(RH_ref_1,weights=wgt)                
    
                
                
	if (data.lon!=srex_raw.lon).any() and (srex_raw.lon!=frac_l.lon).any():
		print('There is an error. The grids do not agree.')
	time=data["time"]
	if y[next(iter(y))].shape[0]==231: #hardcoded way to have a mini-check whether still the right amount of time slots  #next iter thing needed because of strange models with no r1 run
		time = np.arange(1870,2101)      
	elif y[next(iter(y))].shape[0]==230 and model=='CAMS-CSM1-0':
		time = np.arange(1870,2100)
		print('ATTENTION: runs go only until 2099 instead of 2100 because last time step not available on 20191119')
	else:
		print('There is an error. The selected time frame no longer corresponds to the hardcoded time vector.')
	if Tanglob_idx == False:  
		if Tref_all == False:
			return y,time,srex,srex_names,lon_pc,lat_pc,idx_l,wgt_l, RH_ref #df_srex,
		else:
			return y,time,srex,srex_names,lon_pc,lat_pc,idx_l,wgt_l#df_srex,            
	else:
		return y,time,srex,srex_names,lon_pc,lat_pc,idx_l,wgt_l,Tan_wgt_globmean #df_srex,
        
def load_data_single_mod_lclm(gen,model,scenario,usr_time_res='mon',var='treeFrac'):
	""" Load the all initial-condition members of a single model in cmip5 or cmip6 for given scenario plus associated historical period.

		Keyword argument:
		- gen: generation (cmip5 = 5 and cmip6 = 6 are implemented)
		- model: model str
		- scenario: scenario str
		- usr_time_res: time resolution str
		- var: variable str
        
		Output:
		- y: the land grid points of the anomalies of the variable on grid centered over 0 longitude (like the srexgrid) 
		- time: the time slots

	"""
    # the dictionaries are NOT ordered properly + some other adjustments -> will need to be careful with my old scripts

    # see e-mail from Verena on 20191112 for additional infos how could read in several files at once with xarr
    # additionally: she transforms dataset into dataarray to make indexing easier -> for consistency reason with earlier
        # version of emulator (& thus to be able to reuse my scripts), I do not do this (fow now).
    
	# right now I keep reloading constants fields for each run I add -> does not really make sense. 
    # Maybe add boolean to decide instead. however they are small & I have to read them in at some point anyways
    # -> maybe path of least resistence is to not care about it
	print('start with model',model)

	# vars which used to be part of the inputs but did not really make sense as I employ the same ones all the time anyways (could be changed later if needed)

	temp_res = usr_time_res # if not, reading the var file needs to be changed as time var is not named in the same way anymore
	spatial_res = 'g025'


    # load in the constants files
	dir_data = '/net/so4/landclim/snath/data/'
	file_ls = 'interim_invariant_lsmask_regrid.nc' # ERA-interim mask regridded by Richard from 73x144 to 72x144
	file_srex = 'srex-region-masks_20120709.srex_mask_SREX_masks_all.25deg.time-invariant.nc'
	file_srex_shape = 'referenceRegions.shp'


	#df_all_regs = geopandas.read_file(dir_data+file_srex_shape)
	srex_names = ['ALA','CGI','WNA','CNA','ENA','CAM','AMZ','NEB','WSA','SSA','NEU','CEU','MED','SAH','WAF','EAF','SAF',
             'NAS','WAS','CAS','TIB','EAS','SAS','SEA','NAU','SAU'] # SREX names ordered according to SREX mask I am 
                    # employing
	#df_srex = df_all_regs.loc[df_all_regs['LAB'].isin(srex_names)] # alternative indexing: search in column LAB for names
	srex_raw = xr.open_mfdataset(dir_data+file_srex, combine='by_coords',decode_times=False) # srex_raw nrs from 1-26
	#df_srex=srex_raw 
	#srex_raw["time"]=pd.to_datetime(srex_raw.time.values)
	lons, lats = np.meshgrid(srex_raw.lon.values,srex_raw.lat.values) # the lon, lat grid (just to derive weights)    
    
	frac_l = xr.open_mfdataset(dir_data+file_ls, combine='by_coords',decode_times=False) #land-sea mask of ERA-interim bilinearily interpolated 
	frac_l_raw = np.squeeze(copy.deepcopy(frac_l.lsm.values))
	#frac_1["time"]=pd.to_datetime(frac_1.time.values)
	frac_l = frac_l.where(frac_l.lat>-60,0) # remove Antarctica from frac_l field (ie set frac l to 0)

	idx_l=np.squeeze(frac_l.lsm.values)>0.0 # idex land #-> everything >0 I consider land
 

	wgt = norm_cos_wgt(lats) # area weights of each grid point
	wgt_l = (wgt*frac_l_raw)[idx_l] # area weights for land grid points (including taking fraction land into consideration)
    #wgt_l = wgt[idx_l] # area weights for land grid points
	lon_pc, lat_pc = mpu.infer_interval_breaks(frac_l.lon, frac_l.lat) # the lon / lat for the plotting with pcolormesh
	srex=(np.squeeze(srex_raw.srex_mask.values)-1)[idx_l] # srex indices on land

    
	y={}
	run_nrs={}
	if gen == 5:
		dir_var='/net/atmos/data/cmip5-ng/'+var+'/' 
		run_names_list=sorted(glob.glob(dir_var+var+'_'+temp_res+'_'+model+'_'+scenario+'_'+'r*i1p1'+'_'+spatial_res+'.nc'))
        # ATTENTION: are ordered but does not work for models with runs above digit 9 
		index_tr = [i for i, s in enumerate(run_names_list) if 'r1i1p1' in s][0] # find training run 
		#print(run_names_list.pop(index_tr))
		run_names_list.insert(0, run_names_list.pop(index_tr)) # move training run to begin of list

		print(run_names_list)
		for run_name in run_names_list:

			data = xr.open_mfdataset(run_name,decode_times=False)
			if usr_time_res=="ann":
				data=data.rename({'year':'time'})
			data["time"]=cf_units.num2date(data.time.values, 'days since 1800-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
			data=data.sel(time=slice('1870-01-01', '2101-01-01')).roll(lon=72)
            # rename so it is consisten with cmip6 
            # roll so that it is on same grid as others (no longer Pacific centered) 

			#print(data.time.values)
			data = data.assign_coords(lon= (((data.lon + 180) % 360) - 180)) # assign_coords so same labels as others
			if scenario=='ssp119' and model=='EC-Earth3':
				#print(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')) 
				run = int(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')[0]) 
				run_nrs[run_name] = run
			else:
				run = data.attrs['realization_index']
				run_nrs[run_name]=run
           
			y[run] = data[var].values # still absolute values + still contains sea pixels
                
	if gen == 6:
		dir_var = '/net/atmos/data/cmip6-ng/%s/%s/g025/'%(var,usr_time_res)
		if var=='irrLut':
			dir_var = '/net/so4/landclim/snath/data/Emon/irrLut/'
        
		run_names_list=sorted(glob.glob(dir_var+var+'_'+temp_res+'_'+model+'_'+scenario+'_'+'r*i1p1f*'+'_'+spatial_res+'.nc'))
            # ATTENTION:  are ordered but does not work for models with runs above digit 9
            # idea is: every ssp one needs a corresponding hist one (vice versa not the case)

		if (scenario=='ssp119' and model=='EC-Earth3') or (scenario=='ssp585' and model=='CESM2' and var=='irrLut'):
			print(run_names_list) 
			index_tr = [i for i, s in enumerate(run_names_list) if 'r4i1p1' in s][0] # find training run 
			run_names_list.insert(0, run_names_list.pop(index_tr)) # move training run to begin of list
		else:
			index_tr = [i for i, s in enumerate(run_names_list) if 'r1i1p1' in s][0] # find training run 
			run_names_list.insert(0, run_names_list.pop(index_tr)) # move training run to begin of list
            
		if model=='EC-Earth3' and scenario!='ssp119':
			run_names_list=[i for i in run_names_list if len(list(i.split('/')[-1].split('_')[-2].split('i')[0]))!=4]

		for run_name in run_names_list:
			run_name_ssp = run_name
			if (scenario=='ssp119' and model=='EC-Earth3') or (scenario=='ssp585' and model=='CESM2' and var=='treeFrac' and ('r10i1p1' in run_name)):
				run_name_hist = run_names_list[0].replace(scenario,'historical')
			else:
				run_name_hist = run_name.replace(scenario,'historical')
			data = xr.open_mfdataset([run_name_hist,run_name_ssp],concat_dim='time').sel(time=slice('1870-01-01', '2101-01-01')).roll(lon=72)
			data = data.assign_coords(lon= (((data.lon + 180) % 360) - 180))  # assign_coords so same labels as others
			if (scenario=='ssp119' and model=='EC-Earth3') or var=='irrLut' or (scenario=='ssp585' and model=='CESM2' and var=='treeFrac' and ('r10i1p1' in run_name)):
				#print(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')) 
				run = int(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')[0]) 
				run_nrs[run_name] = run
			else:
				run = data.attrs['realization_index']
				run_nrs[run_name]=run
           
			y[run] = data[var].values # still absolute values + still contains sea pixels

	if model=='CESM2':
		if var=='treeFrac':
			dir_lc_ctl='/net/so4/landclim/snath/data/WP1/ctl_crop_frst/treeFrac/CTL_cesm_TreeFrac_g025.nc'
			y_ref=xr.open_mfdataset(dir_lc_ctl).roll(lon=72)['TreeFrac'].values[idx_l]
		elif var=='irrLut':
			dir_lc_ctl='/net/so4/landclim/snath/data/WP1/irr-crop/cesm/QIRRIG/QIRRIG_ctl_cesm_g025.nc'
			y_ref=np.nanmean(xr.open_mfdataset(dir_lc_ctl).roll(lon=72)['QIRRIG'].values[:,idx_l].reshape(-1,12,idx_l.sum()),axis=0)
	elif 'MPI-ESM' in model: 
		if var=='treeFrac':
			dir_lc_ctl='/net/so4/landclim/snath/data/WP1/ctl_crop_frst/treeFrac/CTL_mpiesm_TreeFrac_g025.nc'
			y_ref=xr.open_mfdataset(dir_lc_ctl).roll(lon=72)['TreeFrac'].values[idx_l]
		elif var=='irrLut':
			y_ref=np.zeros([12,idx_l.sum()])
     
      # obtain the anomalies
	for run_name in run_names_list:
		run = run_nrs[run_name]
		#print(y[run].shape)
		if len(y[run].shape)==3:
			y[run]=y[run][:,idx_l]  
		if len(y[run].shape)==4:
			y[run]=y[run][:,2,idx_l]  
		if var=='irrLut':
			y[run]=(y[run].reshape(-1,12,idx_l.sum())-y_ref).reshape(-1,idx_l.sum())
		else:
			y[run]=(y[run]-y_ref*100)
            
    
                
                
	if (data.lon!=srex_raw.lon).any() and (srex_raw.lon!=frac_l.lon).any():
		print('There is an error. The grids do not agree.')
	time=data["time"]
	return y,y_ref,time
        
