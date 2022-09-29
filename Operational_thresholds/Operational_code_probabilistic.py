#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from functions_operational_risk import temperature_processing, wind_processing
from functions_operational_risk import wave_processing, overtopping_processing

pd.options.mode.chained_assignment = None  # default='warn'

from pathos.multiprocessing import ProcessPool, cpu_count


def post_process(temp_parallel, wind_parallel, overtop_parallel, wave_parallel):
    #### post-process
    exceedance_temp= temp_parallel[['port_name','country','continent','annual','run']].rename(columns = {'annual':'freq_temp'})
    exceedance_temp.drop_duplicates(subset = ['port_name','country','run'],inplace = True)

    exceedance_wind= wind_parallel[['port_name','country','continent','annual','run']].rename(columns = {'annual':'freq_wind'})
    exceedance_wind.drop_duplicates(subset = ['port_name','country','run'],inplace = True)

    exceedance_overtopping_parallel= overtop_parallel[['port_name','country','continent','annual','run']].rename(columns = {'annual':'freq_overtop'})
    exceedance_overtopping_parallel.drop_duplicates(subset = ['port_name','country','run'],inplace = True)

    exceedance_wave_parallel= wave_parallel[['port_name','country','continent','annual','run']].rename(columns = {'annual':'freq_wave'})
    exceedance_wave_parallel.drop_duplicates(subset = ['port_name','country','run'],inplace = True)

    #### operational risk
    operational_risk = pd.concat([exceedance_temp.set_index(['port_name','country','continent','run']),exceedance_wind.set_index(['port_name','country','continent','run']),exceedance_overtopping_parallel.set_index(['port_name','country','continent','run']), exceedance_wave_parallel.set_index(['port_name','country','continent','run'])], axis = 1).reset_index().replace(np.nan,0.0)
    operational_risk['downtime'] = operational_risk['freq_temp']+ operational_risk['freq_wind']+ operational_risk['freq_wave'] + operational_risk['freq_overtop']


    return operational_risk


### input paths
path_temp = 'Input/ERA5/Temperature/'
path_wind = 'Input/ERA5/Wind_u10/'
path_wave = 'Input/ERA5/Waves/'

path_parameters = 'Input/'

### input files
parameters = pd.read_csv(path_parameters+'parameters_sampled.csv')
## port id is the file that links ID in input data to port
port_id = pd.read_csv('Input/port_centroids_ID.csv').replace('South-Korea','South Korea')
## wave and surge data for overtopping analysis
max_wave_new_surge_path = 'Input/wave_surge_time_series_ERA5.csv'
max_wave_new_surge_day_path = 'Input/wave_surge_time_series_ERA5_day_select.csv'
## wave input data used for the analysis
wave_input_path = 'Input/wave_height_input.csv'

### set thresholds that vary in sensitivity analysis

### fixed thresholds
Q_coast = 1 ### overtopping discharge coastal breakwater (l/s/m)
Q_revet = 0.4 ### overtopping discharge revetments/seawalls (l/s/m)

N = len(parameters)
print(len(parameters))

### create df for uncertainty analysis
unc_list =  list(range(1,N+1))
wave_ME = parameters['thres_wave'].iloc[0:N].values
wave_HE = wave_ME-1
temp_list = parameters['thres_temp'].iloc[0:N].values
wind_list = parameters['thres_wind'].iloc[0:N].values
uncertainty_height_list= parameters['uncertainty_height'].iloc[0:N].values
rp_wave_list = parameters['rp_wave'].iloc[0:N].values

#### some additional input data
years_wave = np.linspace(1979,2018,40).astype(int).astype(str)
ME_list = port_id[port_id['continent']=='Middle-East']['ID'].to_list()
id_list = port_id['ID'].unique()


### Climate extremes
#pool = ProcessPool(nodes=cpu_count()-3)
print(cpu_count())

#### temperature
#temp_run = pool.map(temperature_processing,[path_temp] * N, [id_list] * N, [ME_list]*N, temp_list, unc_list)
#temp_parallel = pd.concat(temp_run, ignore_index = True, sort= False)

#temp_df_unc =  temp_parallel.merge(port_id,on = 'ID')
#temp_df_unc.to_csv('Output/temperature_output.csv',index = False)

print('temperature finished')
##### wind
#wind_run = pool.map(wind_processing,[path_wind] * N,[id_list] * N, wind_list, unc_list)
#wind_parallel = pd.concat(wind_run, ignore_index = True, sort= False)
#wind_df_unc =  wind_parallel.merge(port_id,on = 'ID')
#wind_df_unc.to_csv('Output/wind_output.csv',index = False)

print('wind finished')
### breakwater overtopping
# run the overtopping functions
with ProcessPool(cpu_count()-3) as pool:
    exceedance_overtopping = pool.map(overtopping_processing, [max_wave_new_surge_path]*N,[max_wave_new_surge_day_path]*N, rp_wave_list, [Q_coast]*N, [Q_revet]*N, uncertainty_height_list, unc_list)

exceedance_overtopping_parallel = pd.concat(exceedance_overtopping, ignore_index = True, sort = False)
exceedance_overtopping_parallel.to_csv('Output/overtopping_output.csv',index = False)

print('overtopping finished')
#### wave overtopping
exceedance_wave = pool.map(wave_processing, [path_wave]*N, [wave_input_path]*N, [years_wave]*N, wave_ME, wave_HE, unc_list)
exceedance_wave_parallel = pd.concat(exceedance_wave,ignore_index = True, sort =False)
exceedance_wave_parallel.to_csv('Output/waves_output.csv',index = False)

print('waves finished')
