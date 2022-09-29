#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
import pandas as pd
import numpy as np
import rioxarray
import xarray as xr
from scipy.stats import triang
pd.options.mode.chained_assignment = None  # default='warn'


import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

def process_infra(input_df, port_buffer, infra_type):
    """Process the infrastructure datasets

    Parameters
    ----------
    input_df : input dataframe of infra
    port_buffer: port buffer area to perform spatial join with
    infra_type: string of infrastructure name to add
    Returns
    ------
    input_df: return the original DataFrame with extra columns added
    """
    input_df['land_use'] = str(infra_type)
    input_df['ID']= input_df['land_use'].astype(str)+input_df.index.astype(str)
    input_df = gpd.sjoin(input_df, port_buffer, how = 'inner',op = 'within')
    input_df['lat_centroid'] = input_df.geometry.centroid.y
    input_df['lon_centroid'] = input_df.geometry.centroid.x
    #input_df = input_df.to_crs({'proj':'cea'})
    input_df['length'] = input_df['geometry'].to_crs({'proj':'cea'}).length  #### length in m
    input_df['number'] = input_df['length']/1000  #### length in km
    input_df = input_df[['ID','land_use','port_name','country','continent','number','lat_centroid','lon_centroid','geometry']]
    return input_df


def prep_hinterland_infrastructure(path_infra, port_buffer, buffer = '1km'):
    """Prepare the infrastructure datasets and add them together """
    ### rail
    rail  = gpd.read_file(path_infra+'rail_buffer_'+str(buffer)+'.gpkg')[['geometry']]
    rail = process_infra(rail, port_buffer,'Rail')
    rail_length_port = rail.groupby(['port_name','country','continent','land_use'])['number'].sum().reset_index()

    ### electricity
    electricity  = gpd.read_file(path_infra+'electricity_buffer_'+str(buffer)+'.gpkg')[['geometry']]
    electricity = process_infra(electricity, port_buffer,'Electricity')
    electricity_length_port = electricity.groupby(['port_name','country','continent','land_use'])['number'].sum().reset_index()


    ### road
    road  = gpd.read_file(path_infra+'road_buffer_'+str(buffer)+'.gpkg')[['GP_RSE','geometry']]
    road = process_infra(road, port_buffer,'Road')
    road_length_port = road.groupby(['port_name','country','continent','land_use'])['number'].sum().reset_index()

    ### power plants
    power_plant = gpd.read_file(path_infra+'power_port_buffer_'+str(buffer)+'.gpkg').rename(columns = {'latitude':'lat_centroid','longitude':'lon_centroid'})
    power_plant['land_use'] = 'Power'
    power_plant['number'] = 1
    power_plant['ID']= power_plant['land_use'].astype(str)+power_plant.index.astype(str)
    power_plant = power_plant[['ID','land_use','port_name','country','continent','number','lat_centroid','lon_centroid','geometry']]
    power_plant_port = power_plant.groupby(['port_name','country','continent','land_use'])['number'].sum().reset_index()


    ## derive the maximum length for road,rail,electricity, power plants
    infra_dataset = pd.concat([road,electricity,rail,power_plant])  ### this one is used
    infra_port_total = pd.concat([rail_length_port,road_length_port,electricity_length_port,power_plant_port],ignore_index = True)
    return infra_dataset, infra_port_total



def prep_port_areas(path_infra):
    """Prepare the port terminal datasets """
    ### read the port areas
    port_area = gpd.read_file(path_infra+'port_landuse.gpkg').dropna().replace({'industry':'Industry','refinery':'Refinery'})
    port_area['lat_centroid'] = port_area.centroid.y
    port_area['lon_centroid'] = port_area.centroid.x
    port_area['ID']= port_area['land_use'].astype(str)+port_area.index.astype(str)
    port_area['area'] = port_area['geometry'].to_crs({'proj':'cea'}).area

    ### industry ports:  only land use type is classified as a Industry, meaning that that land-use type is esssential for that port
    industry_port =port_area.drop_duplicates(subset = ['port_name','country'],keep = 'first')[port_area.drop_duplicates(subset = ['port_name','country'],keep = 'first')['land_use'].isin(['Industry','Refinery'])]
    industry_port['industry_port'] = 1

    ### add 1 to landuse that is considered essential

    ### main terminal types, ignore storage and warehouses etc.
    essential_landuse_list = ['General Cargo','Container', 'Dry Bulk', 'Raw','RoRo','Liquid']
    port_area['industry_port'] = np.where(port_area['port_name'].isin(industry_port['port_name'].unique()),1,0)
    port_area['essential'] = np.where(port_area['industry_port']== 1,1,0)
    port_area['essential'] = np.where(port_area['land_use'].isin(essential_landuse_list),1,port_area['essential'])

    ## derive the essential port locations
    port_essential = port_area[port_area['essential']==1].groupby(['port_name','country','continent'])['area'].sum().reset_index().rename(columns = {'area':'area_essential'})

    ### find maximum total area and essential area
    port_area_total = port_area.groupby(['port_name','country','continent'])['area'].sum().reset_index()
    port_area_total = port_area_total.merge(port_essential,on = ['port_name','country','continent'])

    return port_area, port_area_total


def crane_capacity(daily_port_calls, port_buffer, number_cranes):
    """Estimate the capacity of cranes in the ports

    Parameters
    ----------
    daily_port_calls : DataFrame of daily port calls per vessel type
    port_buffer: buffer area of ports
    number_cranes: number of cranes on average per container vessels, standard = 3

    Returns
    ------
    container_cranes: DataFrame with the number of container cranes per port

    """

    #### find the container capacity
    container_vessel_capacity = daily_port_calls[daily_port_calls['vessel_type_UN']=='Container'].sort_values(by = 'port_calls',ascending = False).groupby(['port_name','country'])['port_calls'].quantile(0.95).reset_index().rename(columns = {'port_calls':'container_vessels'})
    container_vessel_mean = daily_port_calls[daily_port_calls['vessel_type_UN']=='Container'].sort_values(by = 'port_calls',ascending = False).groupby(['port_name','country'])['port_calls'].mean().reset_index().rename(columns = {'port_calls':'container_vessels_mean'})

    all_vessel_mean = daily_port_calls.groupby(['date-entry','port_name','country'])['port_calls'].sum().reset_index().sort_values(by = 'port_calls',ascending = False).groupby(['port_name','country'])['port_calls'].mean().reset_index().rename(columns = {'port_calls':'all_vessels_mean'})

    ### number of cranes determined by number of container vessel capacity
    container_vessel_capacity['number'] = (container_vessel_capacity['container_vessels']*number_cranes).astype(int)
    container_vessel_capacity = container_vessel_capacity.merge(all_vessel_mean,on = ['port_name','country'])
    container_vessel_capacity = container_vessel_capacity.merge(container_vessel_mean,on = ['port_name','country'])

    ### percentage of port that is container terminal
    container_vessel_capacity['perc_container'] = container_vessel_capacity['container_vessels_mean']/container_vessel_capacity['all_vessels_mean']
    container_vessel_capacity = container_vessel_capacity.merge(port_buffer[['port_name','country','continent']],on = ['port_name','country'])
    container_vessel_capacity['land_use']='Crane'

    container_cranes = container_vessel_capacity[['land_use','port_name','country','continent','number','perc_container']]
    return container_cranes


def utilization_rate(port_calls, cranes):
    """Estimate the utilization rate of ports based on weekly port calls"""

    ### weekly port calls
    weekly_port_calls = port_calls.groupby(['port_name','country','week','year'])['port_calls'].sum().reset_index()
    ### median port calls
    median_calls = weekly_port_calls.groupby(['port_name','country'])['port_calls'].median().reset_index().rename(columns = {'port_calls':'median'})
    ### maximum port calls
    capacity_calls = weekly_port_calls.groupby(['port_name','country'])['port_calls'].max().reset_index().rename(columns = {'port_calls':'peak'})
    utilization = median_calls.merge(capacity_calls, on = ['port_name','country'])

    #### utilization = calls_median / calls maximum
    utilization['utilization'] = np.round(utilization['median']/utilization['peak'],2)

    ### merge container cranes to the dataframe, we need this later
    utilization = utilization.merge(cranes[['port_name','country','perc_container']], on = ['port_name','country'], how = 'outer').replace(np.nan,0)
    return utilization


def port_freight_prep(port_value, port_weight):
    """prepare the port freight data"""

    port_value['freight_total'] = port_value['freight_total']*1000 ### value conversion
    port_weight['freight_tonnes'] = port_weight['freight_total']
    port_value = port_value.merge(port_weight[['port_name','country','freight_tonnes']], on = ['port_name','country'])
    port_value['value_tonnes']= port_value['freight_total']/port_value['freight_tonnes']
    return port_value


def prep_curves(path_curves, unc_recovery, unc_dd, unc_rec_curve, hazard):
    """prepare the depth damage and recovery curves per hazard and add the uncertainty to it"""

    if hazard == 'flood':
        curves = pd.read_excel(path_curves+'curves_flooding.xlsx')
        curves = curves.set_index('depth').stack().reset_index().rename(columns = {'level_1':'land_use',0:'fraction'})
        curves['depth'] = curves['depth'].astype(float)
        curves['depth'] = curves['depth']+unc_dd
        curves['depth'] = np.where(curves['depth']<0, 0,curves['depth'])

        curves_recovery = pd.read_excel(path_curves+'curves_recovery_flooding.xlsx')
        curves_recovery = curves_recovery.set_index('frac_damage').stack().reset_index().rename(columns = {'level_1':'asset',0:'duration'})
        curves_recovery = curves_recovery.merge(unc_recovery, on = 'asset').sort_values(by ='frac_damage')
        curves_recovery['duration'] = curves_recovery['duration']*curves_recovery['unc_factor']
        curves_recovery['frac_damage'] = curves_recovery['frac_damage']+unc_rec_curve
        curves_recovery['frac_damage'] = np.where(curves_recovery['frac_damage']<0, 0,curves_recovery['frac_damage'])

    elif hazard == 'TC':
        curves = pd.read_excel(path_curves+'curves_TC.xlsx')
        curves = curves.set_index('max_wind').stack().reset_index().rename(columns = {'level_1':'land_use',0:'fraction'})
        curves['max_wind'] = curves['max_wind'].astype(float)
        curves['max_wind'] = curves['max_wind']+unc_dd
        curves['max_wind'] = np.where(curves['max_wind']<0, 0,curves['max_wind'])

        curves_recovery = pd.read_excel(path_curves+'curves_recovery_TC.xlsx')
        curves_recovery = curves_recovery.set_index('frac_damage').stack().reset_index().rename(columns = {'level_1':'asset',0:'duration'})
        curves_recovery = curves_recovery.merge(unc_recovery, on = 'asset').sort_values(by ='frac_damage')
        curves_recovery['duration'] = curves_recovery['duration']*curves_recovery['unc_factor']
        curves_recovery['frac_damage'] = curves_recovery['frac_damage']+unc_rec_curve
        curves_recovery['frac_damage'] = np.where(curves_recovery['frac_damage']<0, 0,curves_recovery['frac_damage'])


    elif hazard == 'earthquake':
        curves = pd.read_excel(path_curves+'curves_earthquake.xlsx')
        curves = curves.set_index('PGA').stack().reset_index().rename(columns = {'level_1':'land_use',0:'fraction'})
        curves['PGA'] = curves['PGA'].astype(float)
        curves['PGA'] = curves['PGA']+unc_dd
        curves['PGA'] = np.where(curves['PGA']<0, 0,curves['PGA'])

        curves_recovery = pd.read_excel(path_curves+'curves_recovery_earthquake.xlsx')
        curves_recovery = curves_recovery.set_index('frac_damage').stack().reset_index().rename(columns = {'level_1':'asset',0:'duration'})
        curves_recovery = curves_recovery.merge(unc_recovery, on = 'asset').sort_values(by ='frac_damage')
        curves_recovery['duration'] = curves_recovery['duration']*curves_recovery['unc_factor']
        curves_recovery['frac_damage'] = curves_recovery['frac_damage']+unc_rec_curve
        curves_recovery['frac_damage'] = np.where(curves_recovery['frac_damage']<0, 0,curves_recovery['frac_damage'])


    return curves, curves_recovery


def process_operational(path_operational):
    """prepare the results of the operational risk analysis"""
    #### post-process
    temp_parallel = pd.read_csv(path_operational+'temperature_output.csv')
    exceedance_temp= temp_parallel[['port_name','country','continent','annual','run']].rename(columns = {'annual':'freq_temp'})
    exceedance_temp.drop_duplicates(subset = ['port_name','country','run'],inplace = True)
    exceedance_temp['freq_temp'] = exceedance_temp['freq_temp']/6

    wind_parallel = pd.read_csv(path_operational+'wind_output.csv')
    exceedance_wind= wind_parallel[['port_name','country','continent','annual','run']].rename(columns = {'annual':'freq_wind'})
    exceedance_wind.drop_duplicates(subset = ['port_name','country','run'],inplace = True)

    overtop_parallel = pd.read_csv(path_operational+'overtopping_output.csv')
    exceedance_overtopping_parallel= overtop_parallel[['port_name','country','continent','annual','run']].rename(columns = {'annual':'freq_overtop'})
    exceedance_overtopping_parallel.drop_duplicates(subset = ['port_name','country','run'],inplace = True)

    wave_parallel = pd.read_csv(path_operational+'waves_output.csv')
    exceedance_wave_parallel= wave_parallel[['port_name','country','continent','annual','run']].rename(columns = {'annual':'freq_wave'})
    exceedance_wave_parallel.drop_duplicates(subset = ['port_name','country','run'],inplace = True)

    #### operational risk
    operational_risk = pd.concat([exceedance_temp.set_index(['port_name','country','continent','run']),exceedance_wind.set_index(['port_name','country','continent','run']),exceedance_overtopping_parallel.set_index(['port_name','country','continent','run']), exceedance_wave_parallel.set_index(['port_name','country','continent','run'])], axis = 1).reset_index().replace(np.nan,0.0)
    operational_risk['downtime'] = operational_risk['freq_temp']+ operational_risk['freq_wind']+ operational_risk['freq_wave'] + operational_risk['freq_overtop']


    return operational_risk
