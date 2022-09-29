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

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def earthquake_hazard(path, rp_val, df_points,lat='lat_centroid',lon = 'lon_centroid',rp_setting = 'multiple'):
    """Perform earthquake exposure analysis
    path: path to earthquake gridded Data
    rp_val: list of return periods
    df_points: point dataset of infrastructure to merge the earthquake data on
    """
    earthquake_exposure = pd.DataFrame()
    for rp in rp_val:
        ### multiple return period
        earthquake_exposure_rp = df_points.copy()
        earthq = rioxarray.open_rasterio(path+'Seismic hazard_PGA_RT'+str(rp)+'years.tif')

        ### list of lat long
        lons = earthquake_exposure_rp[lon].values
        lats = earthquake_exposure_rp[lat].values
        lats = xr.DataArray(lats, dims='z') #'z' is an arbitrary name placeholder
        lons = xr.DataArray(lons, dims='z')

        val = earthq.sel(x=lons, y=lats, method='nearest')
        earth_val = val.values*0.0010197162129779282
        earthquake_exposure_rp['rp'] = rp
        earthquake_exposure_rp['rp'] = earthquake_exposure_rp['rp'].astype(int)
        earthquake_exposure_rp['PGA'] = earth_val[0]
        earthquake_exposure_rp['PGA'] = np.where(earthquake_exposure_rp['PGA']<0,0,earthquake_exposure_rp['PGA'])
        earthquake_exposure = pd.concat([earthquake_exposure,earthquake_exposure_rp],ignore_index = True, sort = False)
    return earthquake_exposure


def damage_estimate(df_hazard, curve, max_damage, scaling, hazard, type_asset, damage_conversion=1.0):
    """Perform the direct damage estimates

    Parameters
    ----------
    df_hazard : input dataframe of exposed assets and the severity of hazard
    curve: depth-damage curve
    max_damage: maximum damage value per land-use type
    scaling: scaling factors between continents for the maximum damage
    hazard: string of the hazard, either 'flood', 'TC' or 'earthquake'
    type_asset: type of asset, either 'port' or 'infra'
    damage_conversion: conversion of asset construction to reconstruction ratio, should be around 60%

    Returns
    ------
    df_hazard: DataFrame of asset-level damage estimates
    df_hazard_total:DataFrame of asset-level damage estimates aggregated to the port-level

    """

    ### merge the curves on the data
    df_hazard = df_hazard.merge(max_damage, on = 'land_use')
    df_hazard = df_hazard.merge(scaling, on = 'continent')

    #### convert reconstruction to damage values using damage conversion
    if type_asset!='port':
        df_hazard['cost'] = df_hazard['cost']*damage_conversion

    #### merge depth damage curve on hazard data
    if hazard =='earthquake':
        df_hazard =  pd.merge_asof(df_hazard.sort_values(by = 'PGA'), curve, by = 'land_use',on='PGA')
    elif hazard == 'TC':
        df_hazard =  pd.merge_asof(df_hazard.sort_values(by = 'max_wind'), curve, by = 'land_use',on='max_wind')
    elif hazard == 'flood':
        df_hazard =  pd.merge_asof(df_hazard.sort_values(by = 'depth'), curve, by = 'land_use',on='depth')


    #### estimate the damage per hazard
    if hazard =='flood':
        if type_asset=='port':
            df_hazard['area_damaged']  = df_hazard['fraction']*df_hazard['area_flooded']
            df_hazard['area_affected']  = df_hazard['area_flooded']
            df_hazard['damage']  = df_hazard['scaling']*df_hazard['cost']*df_hazard['fraction']*df_hazard['area_flooded']
            df_hazard['risk_frac'] = (1/df_hazard['rp'])*(df_hazard['damage'])
        else:
            df_hazard['number_damaged']  = df_hazard['fraction']*df_hazard['length_flooded']
            df_hazard['number_affected']  = df_hazard['length_flooded']
            df_hazard['damage']  = df_hazard['scaling']*df_hazard['cost']*df_hazard['fraction']*df_hazard['length_flooded']
            df_hazard['risk_frac'] = (1/df_hazard['rp'])*(df_hazard['damage'])

    else:
        if type_asset=='port':
            df_hazard['area_damaged']  = df_hazard['fraction']*df_hazard['area']
            df_hazard['area_affected']  = df_hazard['area_damaged']
            df_hazard['damage']  = df_hazard['scaling']*df_hazard['cost']*df_hazard['fraction']*df_hazard['area']
            df_hazard['risk_frac'] = (1/df_hazard['rp'])*(df_hazard['damage'])
        else:
            df_hazard['number_damaged']  = df_hazard['fraction']*df_hazard['number']
            df_hazard['number_affected']  = df_hazard['number_damaged']
            df_hazard['damage']  = df_hazard['scaling']*df_hazard['cost']*df_hazard['fraction']*df_hazard['number']
            df_hazard['risk_frac'] = (1/df_hazard['rp'])*(df_hazard['damage'])


    #### estimate the total damage aggregated to the port level
    df_hazard_total =df_hazard.groupby(['port_name','country','continent','rp'])['damage'].sum().reset_index()
    df_hazard_total['port_country'] = df_hazard_total['port_name'].astype(str) + '_'+df_hazard_total['country'].astype(str)

    ### only return when damage is greater than zero
    df_hazard = df_hazard[df_hazard['damage']>0]
    return df_hazard, df_hazard_total

def trapezoidal(list_prob,list_losses,metric='prob'):
    """Perform the trapezoidal rule of risk calculation. input is a list of probability and losses,
                sorted by the return periods from small to large (rp 1, rp 5, rp 10...)
    metric = 'prob': list_prob is a list of probability (1/rp) or list of return periods

    Output: risk value
    """

    if metric=='prob':
        list_prob = list_prob
    elif metric =='rp':
        list_prob = 1/list_prob

    if len(list_prob) == 1:
        val = list_prob[0] * list_losses[0]
    else:
        risk_sum = 0
        for i in range(0,len(list_prob)-1):
            risk_frac =  0.5 * (list_prob[i+1]-list_prob[i]) * (list_losses[i]+list_losses[i+1]) #list_prob[i]*(list_prob[i+1]-list_prob[i])*(list_losses[i]+list_losses[i+1])
            risk_sum = risk_sum+risk_frac
        val = risk_sum + list_prob[0] * list_losses[0]
    return val

def risk_processing(df_damage, list_port_country, port_value_df, loss_metric, category, metric = 'prob'):
    """Perform the risk estimates

    Parameters
    ----------
    df_damage : input dataframe of damaged assets
    list_port_country: list of port_country pairs to loop over
    port_value_df: dataframe of the annual freight value per port to merge to estimate downtime risk
    loss_metric: risk metric used for the risk calculation, either 'damage' or 'duration_max'
    category: category of risk calculation, either 'downtime' or 'port' or 'infra'
    metric: metric for risk calculation, see 'trapezoidal rule function'

    Returns
    ------
    df_risk: dataframe of risk results per port.
        For downtime this risk is in 'trade value at risk in dollar per year'
        For damage this risk is in 'assets at risk in dollar per year'

    """
    ### loop over the countries
    df_risk = pd.DataFrame()
    for port_country in list_port_country:
        port_risk_values = df_damage[df_damage['port_country']==port_country].sort_values(by = 'rp',ascending = False)
        if len(port_risk_values)==0:
            continue
        else:
            ### run the risk calculation
            risk = trapezoidal(1/port_risk_values['rp'].values, port_risk_values[str(loss_metric)].values,metric = metric)
            df_risk_port = pd.DataFrame({'port_name':port_risk_values['port_name'].iloc[0],'country':port_risk_values['country'].iloc[0],'risk':[risk],'cat':[str(category)]})
            df_risk = pd.concat([df_risk,df_risk_port],ignore_index = True)

    #### merge the port value information
    df_risk = df_risk.merge(port_value_df,on = ['port_name','country'])
    if category == 'downtime':
        df_risk['downtime'] = df_risk['risk']
        df_risk['risk'] = df_risk['risk']*df_risk['freight_total']/365
    return df_risk


def recovery_time(df_damage_port, curve, asset):
    """Merge the recovery time curve to the damage dataframe per asset type """

    if asset =='port':
        df_damage_port['asset']='Port'
        df_damage_port = pd.merge_asof(df_damage_port.sort_values(by = 'frac_damage'),curve, by = 'asset',on='frac_damage',direction = 'nearest')
    else:
        df_damage_port['asset']=df_damage_port['land_use']
        df_damage_port = pd.merge_asof(df_damage_port.sort_values(by = 'frac_damage'),curve, by = 'asset',on='frac_damage',direction = 'nearest')
    return df_damage_port




def fraction_damage_recovery(df_asset_damage, df_asset_total, curves_recovery, asset_type):
    """Estimate the recovery duration based on the port-aggregated level of damage. Only look at the essential port areas for recovery

        Parameters
        ----------
        df_asset_damage : DataFrame of asset damages per port
        df_asset_total: DataFrame of the total asset inventory per port
        curves_recovery: recovery curves that relate percentage of damage to recovery duration
        asset_type: either port, infra or crane
        Returns
        ------
        asset_type = 'port' --> recovery: recovery duration per port
        asset_type = 'infra' --> recovery: recovery duration per infra-type, recovery_leading: dominant failure mode per return period using fault tree principles
        asset_type = 'cranes' --> recovery: recovery duration per port, taking into account that only fraction of port is container

    """
    if asset_type == 'port':
        ### damages essential port areas
        df_asset_damage = df_asset_damage[df_asset_damage['essential']==1].groupby(['port_name','country','continent','rp'])['area_damaged','area_affected'].sum().reset_index()

        ### merge total essential port area
        df_asset_damage = df_asset_damage.merge(df_asset_total[['port_name','country','continent','area_essential']], on = ['port_name','country','continent'])
        df_asset_damage['frac_damage'] = df_asset_damage['area_affected']/df_asset_damage['area_essential']
        ### find the recovery time
        recovery = recovery_time(df_asset_damage, curves_recovery,'port')
        return recovery

    elif asset_type =='infra':
        df_asset_damage = df_asset_damage.groupby(['port_name','country','continent','rp','land_use'])['number_damaged','number_affected'].sum().reset_index()
        ### merge total asset length
        df_asset_damage = df_asset_damage.merge(df_asset_total, on = ['port_name','country','continent','land_use'])
        df_asset_damage['frac_damage'] = df_asset_damage['number_affected']/df_asset_damage['number']
        ### find the recovery time
        recovery = recovery_time(df_asset_damage,curves_recovery,'infra')
        ### find the large recovery time per return period
        recovery_leading = recovery.sort_values(by = 'duration',ascending = False).drop_duplicates(subset = ['port_name','country','rp'])
        return recovery, recovery_leading

    elif asset_type =='crane':
        df_asset_damage = df_asset_damage.drop(columns = ['duration','duration_l','duration_u'])
        ### damage related to damaged number of cranes over the total number of cranes
        df_asset_damage['frac_damage'] = df_asset_damage['number_affected']/df_asset_damage['number']
        df_asset_damage['asset'] = 'Crane'
        df_asset_damage = pd.merge_asof(df_asset_damage.sort_values(by = 'frac_damage'),curves_recovery,by = 'asset',on='frac_damage',direction = 'nearest')

        ### duration is related to the fraction of cranes that is damages, the percentage of containers in a port and the duration of the port disruption
        df_asset_damage['duration_crane'] = df_asset_damage['perc_container']*df_asset_damage['frac_damage']*df_asset_damage['duration']
        return df_asset_damage

def triangular(ECDF_val,mode,low,high):
    """Sample from a triangular distribution function"""
    return triang.ppf(ECDF_val,mode, low, high-low)

def recovery_max(hazard, df1, df2, TC_unc=0.5, *df3):
    """Find the leading recovery duration using Fault Tree Analysis

        Parameters
        ----------
        hazard : string of the type of hazard, this because TC has three dataframes and EQ and Flood two
        df1: dataframe with port recovery duration
        df2: dataframe with infra recovery duration
        df3: dataframe with cranes recovery duration
        TC_unc: sample TC uncertainty from a triangular distribution function
        Returns
        ------
        recovery_both: maximum recovery duration per return period for hazard
    """
    if hazard == 'earthquake' or hazard == 'flood':
        ### df1 and df2 (port, infra)
        recovery_both = df1[['port_name','country','continent','rp','duration']].rename(columns = {'duration':'duration1'}).merge(df2[['port_name','country','continent','rp','duration']].rename(columns = {'duration':'duration2'}),on = ['port_name','country','continent','rp'], how = 'outer')
        recovery_both = recovery_both[recovery_both['duration1'].notna()]
        recovery_both['duration_max'] = recovery_both[["duration1", "duration2"]].max(axis=1)
        recovery_both['port_country'] = recovery_both['port_name'].astype(str) + '_'+recovery_both['country'].astype(str)

    elif hazard == 'TC':
        ### df1, df2, df3 (port, infra, cranes)
        df3 = df3[0]
        recovery_both = df1[['port_name','country','continent','rp','duration','duration_u','duration_l']].rename(columns = {'duration':'duration1'}).merge(df2[['port_name','country','continent','rp','duration']].rename(columns = {'duration':'duration2'}),on = ['port_name','country','continent','rp'], how = 'outer')
        recovery_both = recovery_both[recovery_both['duration1'].notna()]
        recovery_both['duration1'] = recovery_both.apply(lambda x: triangular(TC_unc,0.5,x['duration_l'],x['duration_u']), axis=1) #triang.ppf(TC_unc,0.5, df1['duration_l'], high-low)
        recovery_both['duration_max'] = recovery_both[["duration1", "duration2"]].max(axis=1)

        recovery_both = recovery_both.merge(df3[['port_name','country','continent','rp','duration_crane']], on = ['port_name','country','continent','rp'], how = 'outer')
        recovery_both['duration_max'] = recovery_both[["duration_max", "duration_crane"]].max(axis=1)
        recovery_both['port_country'] = recovery_both['port_name'].astype(str) + '_'+recovery_both['country'].astype(str)

    return recovery_both


def risk_summation(*df):
    """sum multiple risk calculations together for asset risk"""

    columns_include = ['port_name','country','continent','risk']

    df_concat = pd.DataFrame()
    for i in range(0, len(df)):
        df_concat = pd.concat([df_concat, df[i][columns_include]], ignore_index= True, sort = False)

    df_all = df_concat.groupby(['port_name','country','continent'])['risk'].sum().reset_index()

    return df_all

def risk_summation_downtime(*df):
    """sum multiple risk calculations together for downtime risk"""

    columns_include = ['port_name','country','continent','downtime']

    df_concat = pd.DataFrame()
    for i in range(0, len(df)):
        df_concat = pd.concat([df_concat, df[i][columns_include]], ignore_index= True, sort = False)

    df_all = df_concat.groupby(['port_name','country','continent'])['downtime'].sum().reset_index()

    return df_all


def process_indirect_losses(port_trade_network, downtime, port_value, iso3_conversion):
    """Indirect risk analysis based on port-to-port trade network

        Parameters
        ----------
        port_trade_network : port-to-port trade network based on Verschuur et al. 2021
        downtime: Dataframe with downtime risk per port
        port_value: port freight dataset to merge
        iso3_conversion: conversion from country to iso3 to merge datasets

        Returns
        ------
        indirect_risk: dataframe with indirect risk, consisting of forward and backward risk
    """
    #### prepare trade network
    port_trade_network = port_trade_network.groupby(['O','GID_0_i','D','GID_0_j'])['v_mar_port'].sum().reset_index()
    port_to_port_forward = port_trade_network.rename(columns = {'O':'port_name','GID_0_i':'GID_0'}).merge(iso3_conversion[['country','GID_0']], on = ['GID_0'])
    port_to_port_backward = port_trade_network.rename(columns = {'D':'port_name','GID_0_j':'GID_0'}).merge(iso3_conversion[['country','GID_0']], on = ['GID_0'])

    ### indirect effect
    port_to_port_forward = port_to_port_forward.merge(downtime, on = ['port_name','country'])
    port_to_port_forward['risk'] = port_to_port_forward['downtime']*port_to_port_forward['v_mar_port']/365

    port_to_port_backward = port_to_port_backward.merge(downtime, on = ['port_name','country'])
    port_to_port_backward['risk'] = port_to_port_backward['downtime']*port_to_port_backward['v_mar_port']/365

    ### calculate risk
    forward_risk = port_to_port_forward.groupby(['D','GID_0_j'])['risk'].sum().reset_index().rename(columns = {'D':'port_name','GID_0_j':'GID_0','risk':'forward_risk'})
    forward_risk = forward_risk.merge(iso3_conversion[['country','GID_0']], on = ['GID_0'])
    backward_risk = port_to_port_backward.groupby(['O','GID_0_i'])['risk'].sum().reset_index().rename(columns = {'O':'port_name','GID_0_i':'GID_0','risk':'backward_risk'})
    backward_risk = backward_risk.merge(iso3_conversion[['country','GID_0']], on = ['GID_0'])

    ### find total and add port level value
    indirect_risk = forward_risk.merge(backward_risk, on = ['port_name','country','GID_0'], how = 'outer').replace(np.nan,0)
    indirect_risk['indirect_risk'] = indirect_risk['forward_risk'] + indirect_risk['backward_risk']
    indirect_risk = indirect_risk.merge(port_value[['port_name','country','continent','lon','lat','freight_total']], on = ['port_name','country'])

    return indirect_risk

def additional_losses(downtime_risk, utilization_df, freight_rate, port_charge, inventory, vot, days_spare, rerout):
    """Process the logistics losses

        Parameters
        ----------
        downtime_risk : downtime risk estimates
        utilization_df: utilization rate per port
        freight_rate: freight rate value
        port_charge: port charge value
        inventory: inventory cost value
        value of time: value of time (VoT) value
        days_spare: days of utilizing spare capacity (0-10 days)
        rerout: ability of container flows to reroute (0-1)

        Returns
        ------
        downtime_risk: original dataframe with the additional logistics losses added
    """

    ### amount of equivalent days that can be recaptured in a year
    utilization_df['recapture'] = ((1-utilization_df['utilization'])/utilization_df['utilization'])*days_spare

    #### additional losses to ports, shippers and carriers
    downtime_risk['tonnes'] = downtime_risk['freight_total']/downtime_risk['value_tonnes']
    downtime_risk = downtime_risk.merge(utilization_df[['port_name','country','recapture','perc_container']], on = ['port_name','country'])
    downtime_risk['downtime_adjust'] = np.where(downtime_risk['downtime']>downtime_risk['recapture'],downtime_risk['downtime']-downtime_risk['recapture'],0)


    #### estimate losses without adaptation
    downtime_risk['risk_carrier'] = downtime_risk['downtime']*(downtime_risk['tonnes']/(365))*freight_rate
    downtime_risk['risk_port'] = downtime_risk['downtime']*(downtime_risk['tonnes']/(365))*port_charge
    downtime_risk['risk_shipper'] = downtime_risk['downtime']*(downtime_risk['tonnes']/(365))*inventory + (downtime_risk['downtime']**2)*(downtime_risk['tonnes']/365)*vot
    downtime_risk['risk_all'] = downtime_risk['risk_carrier']+downtime_risk['risk_port']+downtime_risk['risk_shipper']

    #### estimate losses with adaptation

    ### rerouting
    downtime_risk['risk_carrier_rerout'] = (downtime_risk['downtime']*(1-downtime_risk['perc_container']*rerout) * downtime_risk['tonnes']/(365))*freight_rate
    downtime_risk['risk_port_rerout'] = (downtime_risk['downtime']*downtime_risk['tonnes']/(365))*port_charge
    downtime_risk['risk_shipper_rerout'] = downtime_risk['downtime']*((1-downtime_risk['perc_container']*rerout)* downtime_risk['tonnes']/(365))*inventory + (downtime_risk['downtime']**2)*((1-downtime_risk['perc_container']*rerout)* downtime_risk['tonnes']/365)*vot
    downtime_risk['risk_all_rerout'] = downtime_risk['risk_carrier_rerout']+downtime_risk['risk_port_rerout']+downtime_risk['risk_shipper_rerout']

    ### recapture
    downtime_risk['risk_port_recapture'] = downtime_risk['downtime_adjust']*(downtime_risk['tonnes']/(365))*port_charge
    downtime_risk['risk_all_recapture'] = downtime_risk['risk_carrier']+downtime_risk['risk_port_recapture']+downtime_risk['risk_shipper']

    ### both
    downtime_risk['risk'] = downtime_risk['risk_carrier_rerout']+downtime_risk['risk_port_recapture']+downtime_risk['risk_shipper_rerout']


    return downtime_risk

def multi_hazard_risk_components(multi_hazard_risk, multi_hazard_risk_port, multi_hazard_risk_infra):
    """find the contributions of risk to port asset, hinterland infra and logistics losses per port"""

    multi_hazard_risk_components = multi_hazard_risk.merge(multi_hazard_risk_port.rename(columns = {'risk':'risk_port'}), on = ['port_name','country','continent'], how = 'outer').replace(np.nan,0)
    multi_hazard_risk_components = multi_hazard_risk_components.merge(multi_hazard_risk_infra.rename(columns = {'risk':'risk_infra'}), on = ['port_name','country','continent'], how = 'outer').replace(np.nan,0)
    multi_hazard_risk_components['risk_additional'] = multi_hazard_risk_components['risk']-multi_hazard_risk_components['risk_port'] - multi_hazard_risk_components['risk_infra']
    multi_hazard_risk_components['risk_additional'] = np.where(multi_hazard_risk_components['risk_additional']<0, 0, multi_hazard_risk_components['risk_additional'])

    ### get the contributions
    multi_hazard_risk_components['contribution_port'] = multi_hazard_risk_components['risk_port']/multi_hazard_risk_components['risk']
    multi_hazard_risk_components['contribution_infra'] = multi_hazard_risk_components['risk_infra']/multi_hazard_risk_components['risk']
    multi_hazard_risk_components['contribution_additional'] = multi_hazard_risk_components['risk_additional']/multi_hazard_risk_components['risk']
    return multi_hazard_risk_components

def risk_contribution(earthquake_total_risk, pluvial_total_risk, fluvial_total_risk,coastal_total_risk, TC_total_risk,  operational_additional_risk):
    """Add the contribution of the individual hazards into one dataframe"""

    earthquake_total_risk['hazard'] = 'earthquake'
    earthquake_total_risk = earthquake_total_risk[['port_name','country','continent','hazard','risk']]
    pluvial_total_risk['hazard'] = 'pluvial'
    pluvial_total_risk = pluvial_total_risk[['port_name','country','continent','hazard','risk']]

    fluvial_total_risk['hazard'] = 'fluvial'
    fluvial_total_risk = fluvial_total_risk[['port_name','country','continent','hazard','risk']]
    coastal_total_risk['hazard'] = 'coastal'
    coastal_total_risk = coastal_total_risk[['port_name','country','continent','hazard','risk']]
    TC_total_risk['hazard'] = 'TC'
    TC_total_risk = TC_total_risk[['port_name','country','continent','hazard','risk']]
    operational_additional_risk['hazard'] = 'operational'
    operational_additional_risk = operational_additional_risk[['port_name','country','continent','hazard','risk']]
    all_risk = pd.concat([earthquake_total_risk, pluvial_total_risk, fluvial_total_risk, coastal_total_risk, TC_total_risk, operational_additional_risk], ignore_index = True, sort = False)
    all_risk = all_risk.set_index(['port_name','country','continent','hazard']).unstack(level = 'hazard').replace(np.nan,0)
    all_risk.columns = ['TC','coastal','earthquake','fluvial','operational','pluvial']

    return all_risk.reset_index()
