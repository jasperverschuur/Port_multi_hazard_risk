#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rioxarray
import xarray as xr
from scipy.stats import triang
pd.options.mode.chained_assignment = None  # default='warn'

from pathos.multiprocessing import ProcessPool, cpu_count

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

from functions_generic_risk import *
from functions_hazard_risk import *
from functions_data_preparation import *

def global_port_infrastructure_risk_code(mode, path_port_data, path_infra, path_curves, path_max_damage, path_processed, path_operational, path_probabilistic, *run):

    '''#####--------------------------#### load the input parameters  ######--------------------------#####'''
    if mode == 'P':
        run = run[0]
        print(run, 'start')
        prob_parameters = pd.read_csv(path_probabilistic+'parameters_sampled.csv')
        prob_parameters = prob_parameters.rename(columns = {'factor_RA':'factor_RAW','factor_RA.1':'factor_RA'})
        prob_parameters['run'] = run
        prob_parameters = prob_parameters.iloc[run-1]

    else:
        prob_parameters = pd.read_csv(path_probabilistic+'parameters_sampled.csv')
        prob_parameters = prob_parameters.rename(columns = {'factor_RA':'factor_RAW','factor_RA.1':'factor_RA'})
        prob_parameters = prob_parameters.mean(axis = 0)


    #####--------------------------#### set the general parameters based on input dataset  ######--------------------------#####

    #####---------Damage values-----------####
    unc_cranes = prob_parameters.unc_cranes ### 1
    damage_conversion = prob_parameters.damage_conversion #0.6 ### 0.5-0.7

    factor_GC = prob_parameters.factor_GC  ### General Cargo
    factor_RR = prob_parameters.factor_RR  ### RoRo
    factor_LI = prob_parameters.factor_LI  ### Liquid
    factor_CO = prob_parameters.factor_CO  ### Container
    factor_BU = prob_parameters.factor_BU ### Dry Bulk
    factor_RAW = prob_parameters.factor_RAW  ### Raw
    factor_RE = prob_parameters.factor_RE  ### Refinery
    factor_IN = prob_parameters.factor_IN  ### Industry
    factor_WA = prob_parameters.factor_WA  ### Warehouse
    factor_CR = prob_parameters.factor_CR  ### Crane

    factor_RO = prob_parameters.factor_RO  ### Road
    factor_RA = prob_parameters.factor_RA  ### Rail
    factor_EL = prob_parameters.factor_EL  ### Electricity
    factor_PO = prob_parameters.factor_PO  ### Power

    #### create dataframe to merge land use maximum damage
    list_landuse = ['General Cargo','RoRo','Liquid','Container','Dry Bulk','Raw','Refinery','Industry','Warehouse','Crane','Road','Rail','Electricity','Power']
    list_factors = [factor_GC, factor_RR, factor_LI, factor_CO, factor_BU, factor_RAW, factor_RE, factor_IN, factor_WA, factor_CR, factor_RO, factor_RA, factor_EL, factor_PO]
    unc_damage = pd.DataFrame({'land_use': list_landuse,'unc_factor': list_factors})


    #####---------Depth-damage curves-----------####
    dd_curve_flood = prob_parameters.dd_curve_flood #0  #### varies between -0.5 m and +0.5 m
    dd_curve_TC = prob_parameters.dd_curve_TC #0  #### varies between -10 m/s and +10 m/s
    dd_curve_eq = prob_parameters.dd_curve_eq #0  #### varies between -0.2 PGA and +0.2 pGA

    #####---------Recovery curves-----------####
    rec_curve_flood = prob_parameters.rec_curve_flood  #### varies between -0.1 and +0.1 frac
    rec_curve_TC = prob_parameters.rec_curve_TC  #### varies between -0.1 and +0.1 frac
    rec_curve_eq = prob_parameters.rec_curve_eq  #### varies between -0.1 and +0.1 frac


    #####---------recovery durations-----------####
    ### flood
    rec_port_flood = prob_parameters.rec_port_flood ### Port
    rec_road_flood = prob_parameters.rec_road_flood ### Road
    rec_rail_flood = prob_parameters.rec_rail_flood ### Rail
    rec_power_flood = prob_parameters.rec_power_flood ### Power
    ## make a list
    list_asset_flood = ['Port','Road','Rail','Power']
    unc_recovery_flood = pd.DataFrame({'asset': list_asset_flood,'unc_factor':[rec_port_flood, rec_road_flood, rec_rail_flood, rec_power_flood]})

    ### EQ
    rec_port_earthquake = prob_parameters.rec_port_earthquake ### Port
    rec_road_earthquake = prob_parameters.rec_road_earthquake ### Road
    rec_rail_earthquake = prob_parameters.rec_rail_earthquake ### Rail
    rec_electr_earthquake = prob_parameters.rec_electr_earthquake ### Electricity
    rec_power_earthquake = prob_parameters.rec_power_earthquake ### Power
    ## make a list
    list_asset_earthquake = ['Port','Road','Rail','Electricity','Power']
    unc_recovery_earthquake = pd.DataFrame({'asset': list_asset_earthquake,'unc_factor':[rec_port_earthquake, rec_road_earthquake, rec_rail_earthquake, rec_electr_earthquake, rec_power_earthquake]})

    ### TC
    rec_port_TC = prob_parameters.rec_port_TC #0.5 ### 0-1, normal at 0.5, not in list but used later on
    rec_crane_TC = prob_parameters.rec_crane_TC ### Crane
    rec_electr_TC = prob_parameters.rec_electr_TC ### Electricity
    rec_power_TC = prob_parameters.rec_power_TC ### Power
    ## make a list
    list_asset_TC = ['Crane','Electricity','Power']
    unc_recovery_TC = pd.DataFrame({'asset': list_asset_TC,'unc_factor':[rec_crane_TC, rec_electr_TC, rec_power_TC]})


    #####---------Logistics costs-----------####
    freight_rate = prob_parameters.freight_rate
    port_charge = prob_parameters.port_charge
    inventory = prob_parameters.inventory
    vot = prob_parameters.vot
    days_spare = 0
    rerout_normal = 0

    #####--------------------------#### Import all essential port data files ######--------------------------#####
    buffer = '1km'
    ### import the port areas
    port_area, port_area_total = prep_port_areas(path_infra)

    ### infrastructure files
    port_buffer = gpd.read_file(path_infra+'port_buffer_'+str(buffer)+'.gpkg')
    port_buffer['dissolvefield'] = 1

    infra_dataset, infra_port = prep_hinterland_infrastructure(path_infra, port_buffer, buffer = '1km')

    ### port value and weight
    port_freight = port_freight_prep(pd.read_csv(path_port_data+'port_annual_value.csv'), pd.read_csv(path_port_data+'port_annual_freight.csv'))
    port_freight = port_freight.drop_duplicates(subset = ['port_name','GID_0'])

    port_id_operational = pd.read_csv(path_port_data+'port_centroids_ID.csv')
    port_id_operational = port_id_operational[port_id_operational['port_name'].isin(port_area['port_name'].unique())]
    port_freight = port_freight[port_freight['port_name'].isin(port_id_operational['port_name'].unique())]
    ### port to port trade network for indirect losses
    port_to_port = pd.read_csv(path_port_data+'port_to_port_dataset.csv')
    port_to_port['v_mar_port'] = port_to_port['v_mar_port']*1000

    ### conversion from countries and ISO3
    iso3_conversion =pd.read_excel(path_port_data+'GADM_conversion.xlsx',sheet_name = 'selection')

    ### daily port calls
    daily_port_calls = pd.read_csv(path_port_data+'daily_port_calls.csv').rename(columns = {'port-name':'port_name'})

    ### cranes
    container_cranes = crane_capacity(daily_port_calls, port_buffer, 3)  ### this one is used
    container_cranes['number'] = container_cranes['number']*unc_cranes  #### add uncertainty
    ### utilization rate
    utilization = utilization_rate(daily_port_calls, container_cranes) ### this one is used


    #####--------------------------#### Import the depth-damage and recovery curves ######--------------------------######
    ### max damage and add uncertainty
    max_damage = pd.read_excel(path_max_damage+'/max_damage_values.xlsx')
    max_damage = max_damage.merge(unc_damage,on = ['land_use'])
    max_damage['cost'] = max_damage['cost']*max_damage['unc_factor']

    ### continent scaling factors for reconstruction costs
    scaling = pd.read_excel(path_max_damage+'/scaling_factors.xlsx')

    #### curves flood including uncertainty
    curves_flood, curves_recovery_flood = prep_curves(path_curves, unc_recovery_flood, dd_curve_flood, rec_curve_flood, 'flood')
    #### curves earthquake including uncertainty
    curves_earthquake, curves_recovery_earthquake = prep_curves(path_curves, unc_recovery_earthquake, dd_curve_eq, rec_curve_eq, 'earthquake')
    #### curves TC including uncertainty
    curves_TC, curves_recovery_TC = prep_curves(path_curves, unc_recovery_TC, dd_curve_TC, rec_curve_TC, 'TC')



    ######--------------------------#### Risk processing ######--------------------------#####

    #####---------Operational-----------####

    if mode =='EV':
        operational_risk = operational_risk_processing(path_operational, port_freight, 'EV')
    else:
        operational_risk = operational_risk_processing(path_operational, port_freight, 'P', run)
    #logistics losses
    operational_additional_risk = additional_losses(operational_risk, utilization, freight_rate, port_charge, inventory, vot, days_spare, rerout_normal)
    operational_additional_risk['hazard'] = 'operational'

    #####---------Earthquake-----------####
    earthquake_port_risk, earthquake_infra_risk, earthquake_downtime_risk = earthquake_risk_processing(path_processed, port_area_total, infra_port, port_freight, curves_earthquake, curves_recovery_earthquake, max_damage, scaling, damage_conversion)
    #logistics losses
    earthquake_additional_risk = additional_losses(earthquake_downtime_risk, utilization, freight_rate, port_charge, inventory, vot, days_spare, rerout_normal)

    # Aggregate risk
    earthquake_total_risk = risk_summation(earthquake_infra_risk, earthquake_port_risk, earthquake_additional_risk)
    earthquake_total_risk['hazard'] = 'earthquake'


    #####---------TC-----------####
    TC_port_risk, TC_infra_risk, TC_cranes_risk, TC_downtime_risk = TC_risk_processing(path_processed, infra_dataset, port_area, infra_port, container_cranes, port_freight, curves_TC, curves_recovery_TC, rec_port_TC, max_damage, scaling, damage_conversion)
    #logistics losses
    TC_additional_risk = additional_losses(TC_downtime_risk, utilization, freight_rate, port_charge, inventory, vot, days_spare, rerout_normal)

    # Aggregate risk
    TC_total_risk = risk_summation(TC_infra_risk, TC_port_risk, TC_cranes_risk, TC_additional_risk)
    TC_total_risk['hazard'] = 'TC'

    #####---------Fluvial-----------####
    fluvial_port_risk, fluvial_infra_risk, fluvial_downtime_risk = flood_risk_processing('fluvial', path_processed, infra_dataset, infra_port, port_area, port_area_total, port_freight, curves_flood, curves_recovery_flood, max_damage, scaling, damage_conversion)
    #logistics losses
    fluvial_additional_risk = additional_losses(fluvial_downtime_risk, utilization, freight_rate, port_charge, inventory, vot, days_spare, rerout_normal)

    # Aggregate risk
    fluvial_total_risk = risk_summation(fluvial_infra_risk, fluvial_port_risk, fluvial_additional_risk)
    fluvial_total_risk['hazard'] = 'fluvial'

    #####---------Pluvial-----------####
    pluvial_port_risk, pluvial_infra_risk, pluvial_downtime_risk = flood_risk_processing('pluvial', path_processed, infra_dataset, infra_port, port_area, port_area_total, port_freight, curves_flood, curves_recovery_flood, max_damage, scaling, damage_conversion)
    #logistics losses
    pluvial_additional_risk = additional_losses(pluvial_downtime_risk, utilization, freight_rate, port_charge, inventory, vot, days_spare, rerout_normal)

    # Aggregate risk
    pluvial_total_risk = risk_summation(pluvial_infra_risk, pluvial_port_risk, pluvial_additional_risk)
    pluvial_total_risk['hazard'] = 'pluvial'

    #####---------Coastal-----------####
    coastal_port_risk, coastal_infra_risk, coastal_downtime_risk = flood_risk_processing('coastal', path_processed, infra_dataset, infra_port, port_area, port_area_total, port_freight, curves_flood, curves_recovery_flood, max_damage, scaling, damage_conversion)
    #logistics losses
    coastal_additional_risk = additional_losses(coastal_downtime_risk, utilization, freight_rate, port_charge, inventory, vot, days_spare, rerout_normal)

    # Aggregate risk
    coastal_total_risk = risk_summation(coastal_infra_risk, coastal_port_risk, coastal_additional_risk)
    coastal_total_risk['hazard'] = 'coastal'


    #####--------------------------#### create the multi-hazard datasets  ######--------------------------#####

    ### Multi-hazard risk
    multi_hazard_risk = risk_summation(earthquake_total_risk, pluvial_total_risk, fluvial_total_risk,coastal_total_risk, TC_total_risk,  operational_additional_risk)
    multi_hazard_risk['continent'] = np.where(multi_hazard_risk['port_name']=='Arkhangels\'k', 'Eastern-Europe',multi_hazard_risk['continent'])

    multi_hazard_risk_port = risk_summation(earthquake_port_risk, fluvial_port_risk, pluvial_port_risk,coastal_port_risk, TC_port_risk, TC_cranes_risk)
    multi_hazard_risk_port['continent'] = np.where(multi_hazard_risk_port['port_name']=='Arkhangels\'k', 'Eastern-Europe',multi_hazard_risk_port['continent'])

    multi_hazard_risk_infra = risk_summation(earthquake_infra_risk, fluvial_infra_risk, pluvial_infra_risk,coastal_infra_risk, TC_infra_risk)
    multi_hazard_risk_infra['continent'] = np.where(multi_hazard_risk_infra['port_name']=='Arkhangels\'k', 'Eastern-Europe',multi_hazard_risk_infra['continent'])

    multi_hazard_downtime = risk_summation_downtime(TC_downtime_risk, fluvial_downtime_risk, pluvial_downtime_risk, earthquake_downtime_risk, coastal_downtime_risk, operational_risk)
    multi_hazard_downtime = multi_hazard_downtime.merge(port_freight[['port_name','country','freight_total','value_tonnes']], on = ['port_name','country'])
    multi_hazard_downtime['continent'] = np.where(multi_hazard_downtime['port_name']=='Arkhangels\'k', 'Eastern-Europe',multi_hazard_downtime['continent'])

    ### concatenate hazards
    hazard_concat = pd.concat([earthquake_total_risk, TC_total_risk, fluvial_total_risk, pluvial_total_risk, coastal_total_risk, operational_additional_risk[coastal_total_risk.columns] ], ignore_index = True, sort = False)
    hazard_concat = hazard_concat.merge(port_freight[['port_name','country','lat','lon','freight_total']],on = ['port_name','country'])

    #####--------------------------#### trade risk, direct and indirect  ######--------------------------#####
    indirect_risk = process_indirect_losses(port_to_port, multi_hazard_downtime, port_freight, iso3_conversion)

    ### merge
    multi_hazard_risk_trade = risk_summation(TC_downtime_risk, fluvial_downtime_risk, pluvial_downtime_risk, coastal_downtime_risk, earthquake_downtime_risk, operational_risk)
    multi_hazard_risk_trade = multi_hazard_risk_trade.merge(indirect_risk, on = ['port_name','country','continent'])

    #### indirect risk metric
    multi_hazard_risk_trade['metric'] = multi_hazard_risk_trade['indirect_risk']/(multi_hazard_risk_trade['risk']+multi_hazard_risk_trade['indirect_risk'])
    multi_hazard_risk_trade['continent'] = np.where(multi_hazard_risk_trade['port_name']=='Arkhangels\'k', 'Eastern-Europe',multi_hazard_risk_trade['continent'])


    #####--------------------------#### Resilience of logistics costs ######--------------------------#####

    ## Sample rerouting fraction and
    days = np.random.uniform(low=0, high=10, size=1)
    rerout = np.random.uniform(low=0, high=1, size=1)

    ## Process logistics losses including resilience
    risk_additional_losses = additional_losses(multi_hazard_downtime, utilization, freight_rate, port_charge, inventory, vot, days, rerout)
    ## Asset value, which is the risk without logistics costs
    asset_val = ((multi_hazard_risk_port['risk'].sum())+(multi_hazard_risk_infra['risk'].sum()))/1e9

    ## create dataframe of resilience
    df_resilience = pd.DataFrame({'asset_risk':[asset_val],'operational':[risk_additional_losses['risk_all'].sum()/1e9],'recapture':[risk_additional_losses['risk_all_recapture'].sum()/1e9],'rerout':[risk_additional_losses['risk_all_rerout'].sum()/1e9],'both':[risk_additional_losses['risk'].sum()/1e9]})

    #####--------------------------#### find the contributions of individual hazards to aggregate risk ######--------------------------#####

    multi_hazard_risk = multi_hazard_risk.merge(port_freight[['port_name','country','lat','lon','freight_total']],on = ['port_name','country'])
    multi_hazard_risk_contribution = multi_hazard_risk_components(multi_hazard_risk, multi_hazard_risk_port, multi_hazard_risk_infra)

    #####--------------------------#### Output file ######--------------------------#####
    print(run, 'Output files')

    if mode =='EV':
        ### infrastructure and logistics
        multi_hazard_risk_contribution.to_csv('Output/Aggregate/multi_hazard_risk.csv', index = False)
        ### trade exposure
        multi_hazard_risk_trade.to_csv('Output/Trade/multi_hazard_risk_trade.csv',index = False)
        ### hazard-specific risk
        hazard_concat.to_csv('Output/Hazards/hazard_risk.csv',index = False)
        ### logistics resilience
        df_resilience.to_csv('Output/Resilience/logistics_resilience.csv', index = False)

    else:
        ### infrastructure and logistics
        multi_hazard_risk_contribution['run'] = run
        multi_hazard_risk_contribution.to_csv('Output/Aggregate/multi_hazard_risk'+str(run)+'.csv', index = False)
        ### trade exposure
        multi_hazard_risk_trade['run'] = run
        multi_hazard_risk_trade.to_csv('Output/Trade/multi_hazard_risk_trade'+str(run)+'.csv',index = False)
        ### hazard-specific risk
        hazard_concat['run'] = run
        hazard_concat.to_csv('Output/Hazards/hazard_risk'+str(run)+'.csv',index = False)
        ### logistics resilience
        df_resilience['run'] = run
        df_resilience.to_csv('Output/Resilience/logistics_resilience'+str(run)+'.csv', index = False)



########################## ------------------CODE ------------------------------- ########################################

'''#####--------------------------#### Paths  ######--------------------------#####'''

# data preparation
path_port_data = 'Input/Port_data/'
path_infra = 'Input/Infrastructure_data/'

# risk calculation
path_curves = 'Input/Fragility_curves/'
path_max_damage = 'Input/Max_damage/'
path_processed = 'Processed/Hazard_data/'
path_operational = 'Processed/Operational/'
path_probabilistic = 'Input/Probabilistic/'


'''#####--------------------------#### Mode of running the code  ######--------------------------#####'''
mode = 'P' ###### EITHER Expected Value (EV) or Probabilistic (P)

if mode == 'P':
    prob_parameters = pd.read_csv(path_probabilistic+'parameters_sampled.csv')
    N = len(prob_parameters)
    unc_list =  list(range(1,N+1))#[4900:N]

    with ProcessPool(cpu_count()-3) as pool:
        pool.map(global_port_infrastructure_risk_code, [mode]*len(unc_list), [path_port_data]*len(unc_list), [path_infra]*len(unc_list), [path_curves]*len(unc_list), [path_max_damage]*len(unc_list), [path_processed]*len(unc_list), [path_operational]*len(unc_list), [path_probabilistic]*len(unc_list), unc_list)

else:
    global_port_infrastructure_risk_code('EV', path_port_data, path_infra, path_curves, path_max_damage, path_processed, path_operational,path_probabilistic)

print('code complete')
