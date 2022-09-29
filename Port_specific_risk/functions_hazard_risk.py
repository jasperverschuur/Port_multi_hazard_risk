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

from functions_generic_risk import *
from functions_data_preparation import *

def operational_risk_processing(path_operational, port_freight, mode, *run):
    ### load in the data
    operational_risk_data =  process_operational(path_operational)
    if mode =='EV':
        operational_risk_max = operational_risk_data.groupby(['port_name','country','continent'])['downtime'].max().reset_index().rename(columns = {'downtime':'downtime_max'})
        operational_risk = operational_risk_data.groupby(['port_name','country','continent'])['downtime'].sum()/operational_risk_data['run'].nunique()
        operational_risk = operational_risk.reset_index()
        operational_risk = operational_risk.merge(operational_risk_max, on = ['port_name','country','continent'])
        operational_risk['unc'] = (operational_risk['downtime_max'] - operational_risk['downtime'])

    else:
        operational_risk = operational_risk_data[operational_risk_data['run']==run]

    ### add the port freight data
    operational_risk = operational_risk.merge(port_freight, on = ['port_name','country','continent'])

    ### direct risk estimates
    operational_risk['risk'] = operational_risk['downtime']*operational_risk['freight_total']/365
    operational_risk = operational_risk.drop_duplicates(subset = ['port_name','country'])

    return operational_risk


def earthquake_risk_processing(path_processed, port_area_total, infra_port, port_freight, dd_curves, recovery_curves, max_damage, scaling, damage_conversion):
    ### load the datasets
    earthquake_exposure_infra_asset = pd.read_csv(path_processed+'hinterland_earthquake_all.csv')
    earthquake_exposure_port_asset = pd.read_csv(path_processed+'port_areas_earthquake_all.csv')

    #### direct damage
    earthquake_damage_infra_asset, earthquake_damage_infra_total = damage_estimate(earthquake_exposure_infra_asset, dd_curves, max_damage, scaling,'earthquake','infra',damage_conversion)
    earthquake_damage_port_asset, earthquake_damage_port_total = damage_estimate(earthquake_exposure_port_asset, dd_curves, max_damage, scaling,'earthquake','port', 1.0)

    ### recovery of earthquake infrastructure
    earthquake_port_recovery = fraction_damage_recovery(earthquake_damage_port_asset, port_area_total, recovery_curves, 'port')
    earthquake_infra_recovery, earthquake_infra_recovery_leading = fraction_damage_recovery(earthquake_damage_infra_asset, infra_port, recovery_curves, 'infra')

    ### find the maximum recovery
    earthquake_recovery_both = recovery_max('earthquake', earthquake_port_recovery, earthquake_infra_recovery_leading)

    ### risk calculation
    earthquake_infra_risk = risk_processing(earthquake_damage_infra_total, earthquake_damage_infra_total['port_country'].unique(), port_freight, 'damage', 'infra')
    earthquake_port_risk = risk_processing(earthquake_damage_port_total, earthquake_damage_port_total['port_country'].unique(), port_freight, 'damage', 'port')
    earthquake_downtime_risk = risk_processing(earthquake_recovery_both, earthquake_recovery_both['port_country'].unique(), port_freight, 'duration_max', 'downtime')

    return earthquake_port_risk, earthquake_infra_risk, earthquake_downtime_risk


def TC_risk_processing(path_processed, infra_dataset, port_area, infra_port, container_cranes, port_freight, dd_curves, recovery_curves, rec_port_TC, max_damage, scaling, damage_conversion):
    #### read data
    TC_return_period = pd.read_csv(path_processed+'TC_return_period_port.csv')[['port_name','country','max_wind','rp','prediction','prediction_l','prediction_u']].rename(columns = {'prediction':'duration','prediction_l':'duration_l','prediction_u':'duration_u'})
    TC_return_period = TC_return_period.merge(port_area[['port_name','country','continent']].drop_duplicates(), on = ['port_name','country'])

    TC_exposure_infra_asset = infra_dataset.merge(TC_return_period, on = ['port_name','country','continent'])
    TC_exposure_port_asset = port_area.merge(TC_return_period, on = ['port_name','country','continent'])
    TC_exposure_cranes_asset = container_cranes.merge(TC_return_period, on = ['port_name','country','continent'])


    #### direct damage
    TC_damage_infra_asset, TC_damage_infra_total  = damage_estimate(TC_exposure_infra_asset, dd_curves, max_damage, scaling, 'TC','infra',damage_conversion)
    TC_damage_port_asset, TC_damage_port_total = damage_estimate(TC_exposure_port_asset, dd_curves, max_damage, scaling, 'TC','port',1.0)
    TC_damage_cranes_asset, TC_damage_cranes_total = damage_estimate(TC_exposure_cranes_asset, dd_curves, max_damage, scaling, 'TC','crane',1.0)

    ## recovery of TC affected infrastructure
    TC_infra_recovery, TC_infra_recovery_leading = fraction_damage_recovery(TC_damage_infra_asset, infra_port, recovery_curves, 'infra')
    TC_cranes_recovery = fraction_damage_recovery(TC_damage_cranes_asset, infra_port, recovery_curves, 'crane')
    TC_port_recovery = TC_return_period  ### regression results

    ### find the maximum recovery
    TC_recovery_both = recovery_max('TC', TC_port_recovery, TC_infra_recovery_leading, rec_port_TC, TC_cranes_recovery)

    ### risk calculation
    TC_infra_risk = risk_processing(TC_damage_infra_total, TC_damage_infra_total['port_country'].unique(), port_freight, 'damage', 'infra')
    TC_port_risk = risk_processing(TC_damage_port_total, TC_damage_port_total['port_country'].unique(), port_freight, 'damage', 'port')
    TC_cranes_risk = risk_processing(TC_damage_cranes_total, TC_damage_cranes_total['port_country'].unique(), port_freight, 'damage', 'cranes')
    TC_downtime_risk = risk_processing(TC_recovery_both, TC_recovery_both['port_country'].unique(), port_freight, 'duration_max', 'downtime')


    return TC_port_risk, TC_infra_risk, TC_cranes_risk, TC_downtime_risk


def flood_risk_processing(hazard_type, path_processed, infra_dataset, infra_port, port_area, port_area_total, port_freight, dd_curves, recovery_curves, max_damage, scaling, damage_conversion):
    ### read data
    if hazard_type == 'fluvial':
        flood_exposure_infra_asset = pd.read_csv(path_processed+'hinterland_fluvial_all.csv')
        flood_exposure_port_asset = pd.read_csv(path_processed+'port_areas_fluvial_all.csv')
        flood_exposure_port_asset = flood_exposure_port_asset.merge(port_area[['ID','essential']], on = ['ID'])

    elif hazard_type == 'pluvial':
        flood_exposure_infra_asset = pd.read_csv(path_processed+'hinterland_pluvial_all.csv')
        flood_exposure_port_asset = pd.read_csv(path_processed+'port_areas_pluvial_all.csv')
        flood_exposure_port_asset = flood_exposure_port_asset.merge(port_area[['ID','essential']], on = ['ID'])

    elif hazard_type =='coastal':
        flood_exposure_infra_asset = pd.read_csv(path_processed+'hinterland_coastal_all.csv')
        flood_exposure_port_asset = pd.read_csv(path_processed+'port_areas_coastal_all.csv')
        flood_exposure_port_asset = flood_exposure_port_asset.merge(port_area[['ID','essential']], on = ['ID'])

    else:
        print('wrong hazard type put in')

    #### direct damage
    flood_damage_infra_asset, flood_damage_infra_total = damage_estimate(flood_exposure_infra_asset, dd_curves, max_damage, scaling,'flood','infra',damage_conversion)
    flood_damage_port_asset, flood_damage_port_total = damage_estimate(flood_exposure_port_asset, dd_curves, max_damage, scaling,'flood','port', 1.0)

    ### recovery of fluvial affected infrastructure
    flood_port_recovery = fraction_damage_recovery(flood_damage_port_asset, port_area_total, recovery_curves, 'port')
    flood_infra_recovery, flood_infra_recovery_leading = fraction_damage_recovery(flood_damage_infra_asset, infra_port, recovery_curves, 'infra')

    ### find the maximum recovery
    flood_recovery_both = recovery_max('flood', flood_port_recovery, flood_infra_recovery_leading)

    ### risk calculation
    flood_infra_risk = risk_processing(flood_damage_infra_total, flood_damage_infra_total['port_country'].unique(), port_freight, 'damage', 'infra')
    flood_port_risk = risk_processing(flood_damage_port_total, flood_damage_port_total['port_country'].unique(), port_freight, 'damage', 'port')
    flood_downtime_risk = risk_processing(flood_recovery_both, flood_recovery_both['port_country'].unique(), port_freight, 'duration_max', 'downtime')

    return flood_port_risk, flood_infra_risk, flood_downtime_risk
