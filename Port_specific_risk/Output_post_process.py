#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from mpl_toolkits import axes_grid1
import glob
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"


def port_freight_prep(port_value, port_weight):
    port_value['freight_total'] = port_value['freight_total']*1000 ### value conversion
    port_weight['freight_tonnes'] = port_weight['freight_total']
    port_value = port_value.merge(port_weight[['port_name','country','freight_tonnes']], on = ['port_name','country'])
    port_value['value_tonnes']= port_value['freight_total']/port_value['freight_tonnes']
    #port_value = port_value[port_value['freight_total']>0]
    return port_value


def hazard_ensemble_process(hazard_data, hazard_name):

    hazard_data_subset  = hazard_data.set_index(['port_name','country','continent','lat','lon','freight_total','run'])[hazard_name].unstack(level = 'run').replace(np.nan,0)

    #### get the globalensemble
    hazard_ensemble = hazard_data_subset.sum(axis = 0).reset_index().rename(columns = {0:'risk'})
    hazard_ensemble['hazard'] = hazard_name
    #### get the port-level mean, median and range
    hazard_data_subset_mean = hazard_data_subset.mean(axis = 1)
    hazard_data_subset_median = hazard_data_subset.quantile(0.5, axis = 1)
    hazard_data_subset_up = hazard_data_subset.quantile(0.05, axis = 1)
    hazard_data_subset_low = hazard_data_subset.quantile(0.95, axis = 1)

    hazard_all = pd.concat([hazard_data_subset_mean, hazard_data_subset_median, hazard_data_subset_up, hazard_data_subset_low], axis = 1).reset_index().rename(columns = {0:'risk_mean',0.5:'risk_median',0.05:'risk_low',0.95:'risk_high'})

    name = hazard_name
    locals()[name] = pd.concat([hazard_data_subset_mean, hazard_data_subset_median, hazard_data_subset_up, hazard_data_subset_low], axis = 1).reset_index().rename(columns = {0:'risk_mean',0.5:'risk_median',0.05:'risk_low',0.95:'risk_high'})

    df_hazard = pd.DataFrame(locals()[name])
    df_hazard['hazard'] = hazard_name
    df_hazard.to_csv('Processed_output/Hazard_ensemble/'+hazard_name+'_ensemble.csv',index = False)
    hazard_ensemble.to_csv('Processed_output/Histograms/'+hazard_name+'_histogram.csv',index = False)

    return pd.DataFrame(locals()[name]), hazard_ensemble


def process_hazard_output(path_output_data, df_color):
    hazard_data = pd.DataFrame()
    for file in glob.glob(path_output_data+'Hazards/*.csv'):
        column_list = ['port_name','country','continent','lat','lon','freight_total','run','TC','coastal','earthquake','fluvial','operational','pluvial']
        data_run = pd.read_csv(file)
        data_run = data_run.set_index(['port_name','country','continent','lat','lon','freight_total','run','hazard']).unstack(level = 'hazard').replace(np.nan,0).reset_index()
        data_run.columns = column_list
        hazard_data = pd.concat([hazard_data,data_run], ignore_index = True, sort = False)


    ### process the individual hazard datasets
    hazard_list = ['TC','coastal','earthquake','fluvial','operational','pluvial']
    for hazard in hazard_list:
        hazard_ensemble_process(hazard_data, hazard)
    ### hazard_mean
    N = hazard_data['run'].nunique()
    hazard_sum = hazard_data.groupby(['port_name','country','continent','lat','lon','freight_total'])['TC','coastal','earthquake','fluvial','operational','pluvial'].sum()
    hazard_mean = hazard_sum/N
    hazard_mean = hazard_mean.reset_index()

    ### find the dominant
    df_color = pd.DataFrame({'hazard_max':['TC','earthquake','operational','pluvial','fluvial','coastal'], 'color':['darkred','darkgreen','sienna','indigo','darkblue','darkorange']})

    hazard_mean['hazard_max'] = hazard_mean[['TC','earthquake','operational','pluvial','fluvial','coastal']].idxmax(axis=1)
    #dominant_hazard = hazard_mean.stack().reset_index().rename(columns = {'level_6':'hazard',0:'risk'}).sort_values(by = 'risk',ascending = False).drop_duplicates(subset = ['port_name','country'])
    hazard_mean = hazard_mean.merge(df_color, on = ['hazard_max']).rename(columns = {'hazard_max':'hazard_dominant'})

    #### find the total risk and reset index
    hazard_mean['risk'] = hazard_mean[['TC','earthquake','operational','pluvial','fluvial','coastal']].sum(axis = 1)


    #hazard_mean = hazard_mean.merge(dominant_hazard[['port_name','country','hazard_dominant','color']], on = ['port_name','country'])
    hazard_mean.to_csv('Processed_output/Hazard_ensemble/Hazard_dominant.csv', index = False)
    #return hazard_mean


# In[139]:


##### resilience
def process_resilience_data(path_output_data):
    resilience_data = pd.DataFrame()
    for file in glob.glob(path_output_data+'Resilience/*.csv'):
        resilience_data = pd.concat([resilience_data, pd.read_csv(file)], ignore_index = True, sort = False)
    resilience_data['total_risk'] = resilience_data['asset_risk'] + resilience_data['operational']
    resilience_data['total_risk_rerout'] = resilience_data['asset_risk'] + resilience_data['rerout']
    resilience_data['total_risk_recapture'] = resilience_data['asset_risk'] + resilience_data['recapture']
    resilience_data['total_risk_both'] = resilience_data['asset_risk'] + resilience_data['both']
    resilience_data.to_csv('Processed_output/Resilience/resilience.csv')
    #return resilience_data


# In[140]:


def process_trade_data(path_output_data):
    trade_data = pd.DataFrame()
    for file in glob.glob(path_output_data+'Trade/*.csv'):
        trade_data = pd.concat([trade_data, pd.read_csv(file)], ignore_index = True, sort = False)

    trade_data[['port_name','country','continent','risk','indirect_risk','run','lat','lon','freight_total']].to_csv('Processed_output/trade_risk_port_full.csv', index = False)
    trade_data['total_risk'] = trade_data['risk'] + trade_data['indirect_risk']
    #### total risk uncertainty
    total_risk_ensemble = trade_data[['port_name','country','continent','GID_0','total_risk','run']].set_index(['port_name','country','continent','GID_0','run']).unstack(level = 'run').replace(np.nan,0)
    total_risk_mean =  total_risk_ensemble.mean(axis = 1).reset_index().rename(columns = {0:'total_risk_mean'})
    total_risk_median =  total_risk_ensemble.quantile(0.5, axis = 1).reset_index().rename(columns = {0.5:'total_risk_median'})
    total_risk_low =  total_risk_ensemble.quantile(0.05, axis = 1).reset_index().rename(columns = {0.05:'total_risk_low'})
    total_risk_up =  total_risk_ensemble.quantile(0.95, axis = 1).reset_index().rename(columns = {0.95:'total_risk_high'})

    ####  direct risk uncertainty
    direct_risk_ensemble = trade_data[['port_name','country','continent','GID_0','risk','run']].set_index(['port_name','country','continent','GID_0','run']).unstack(level = 'run').replace(np.nan,0)
    direct_risk_mean =  direct_risk_ensemble.mean(axis = 1).reset_index().rename(columns = {0:'direct_risk_mean'})
    direct_risk_median =  direct_risk_ensemble.quantile(0.5, axis = 1).reset_index().rename(columns = {0.5:'direct_risk_median'})
    direct_risk_low =  direct_risk_ensemble.quantile(0.05, axis = 1).reset_index().rename(columns = {0.05:'direct_risk_low'})
    direct_risk_up =  direct_risk_ensemble.quantile(0.95, axis = 1).reset_index().rename(columns = {0.95:'direct_risk_high'})

    ####  indirect risk uncertainty
    indirect_risk_ensemble = trade_data[['port_name','country','continent','GID_0','indirect_risk','run']].set_index(['port_name','country','continent','GID_0','run']).unstack(level = 'run').replace(np.nan,0)
    indirect_risk_mean =  indirect_risk_ensemble.mean(axis = 1).reset_index().rename(columns = {0:'indirect_risk_mean'})
    indirect_risk_median =  indirect_risk_ensemble.quantile(0.5, axis = 1).reset_index().rename(columns = {0.5:'indirect_risk_median'})
    indirect_risk_low =  indirect_risk_ensemble.quantile(0.05, axis = 1).reset_index().rename(columns = {0.05:'indirect_risk_low'})
    indirect_risk_up =  indirect_risk_ensemble.quantile(0.95, axis = 1).reset_index().rename(columns = {0.95:'indirect_risk_high'})


    N = trade_data['run'].nunique()
    trade_data_sum = trade_data.groupby(['port_name','country','continent','GID_0','lat','lon','freight_total'])['risk','forward_risk','backward_risk','indirect_risk'].sum()
    trade_data_mean = trade_data_sum/N
    trade_data_mean['metric'] = trade_data_mean['indirect_risk']/(trade_data_mean['risk'] + trade_data_mean['indirect_risk'])

    trade_data_mean.reset_index(inplace = True)


    ### merge the limits
    trade_data_mean = trade_data_mean.merge(direct_risk_low, on = ['port_name','country','continent','GID_0'])
    trade_data_mean = trade_data_mean.merge(direct_risk_up, on = ['port_name','country','continent','GID_0'])
    trade_data_mean = trade_data_mean.merge(direct_risk_mean, on = ['port_name','country','continent','GID_0'])
    trade_data_mean = trade_data_mean.merge(direct_risk_median, on = ['port_name','country','continent','GID_0'])

    trade_data_mean = trade_data_mean.merge(indirect_risk_low, on = ['port_name','country','continent','GID_0'])
    trade_data_mean = trade_data_mean.merge(indirect_risk_up, on = ['port_name','country','continent','GID_0'])
    trade_data_mean = trade_data_mean.merge(indirect_risk_mean, on = ['port_name','country','continent','GID_0'])
    trade_data_mean = trade_data_mean.merge(indirect_risk_median, on = ['port_name','country','continent','GID_0'])

    trade_data_mean = trade_data_mean.merge(total_risk_low, on = ['port_name','country','continent','GID_0'])
    trade_data_mean = trade_data_mean.merge(total_risk_up, on = ['port_name','country','continent','GID_0'])
    trade_data_mean = trade_data_mean.merge(total_risk_mean, on = ['port_name','country','continent','GID_0'])
    trade_data_mean = trade_data_mean.merge(total_risk_median, on = ['port_name','country','continent','GID_0'])

    ### estimate uncertainty
    trade_data_mean['unc_direct'] = trade_data_mean['direct_risk_high']-trade_data_mean['direct_risk_low']
    trade_data_mean['unc_indirect'] = trade_data_mean['indirect_risk_high']-trade_data_mean['indirect_risk_low']

    trade_data_mean['rel_unc_direct'] = np.where((trade_data_mean['unc_direct']/trade_data_mean['direct_risk_mean'])>0, (trade_data_mean['unc_direct']/trade_data_mean['direct_risk_mean']), 0)
    trade_data_mean['rel_unc_indirect'] = np.where((trade_data_mean['unc_indirect']/trade_data_mean['indirect_risk_mean'])>0, (trade_data_mean['unc_indirect']/trade_data_mean['direct_risk_mean']), 0)

    trade_data_mean['risk_total'] = trade_data_mean['direct_risk_mean'] + trade_data_mean['indirect_risk_mean']
    #### get the histogram of the different runs
    direct_risk_hist = direct_risk_ensemble.sum(axis = 0).reset_index().rename(columns = {0:'risk'})
    indirect_risk_hist = indirect_risk_ensemble.sum(axis = 0).reset_index().rename(columns = {0:'risk'})

    trade_data_mean.to_csv('Processed_output/Trade_ensemble/Trade_ensemble.csv',index = False)
    direct_risk_hist.to_csv('Processed_output/Histograms/direct_trade_risk_histogram.csv',index = False)
    indirect_risk_hist.to_csv('Processed_output/Histograms/indirect_trade_risk_histogram.csv',index = False)
    #return trade_data_mean, direct_risk_hist, indirect_risk_hist


def process_risk_contribution(path_output_data):
    economic_risk_data = pd.DataFrame()
    for file in glob.glob(path_output_data+'Aggregate/*.csv'):
        economic_risk_data = pd.concat([economic_risk_data, pd.read_csv(file)], ignore_index = True, sort = False)

    economic_risk_data[['port_name','country','continent','risk','run','lat','lon','freight_total']].to_csv('Processed_output/economic_risk_port_full.csv', index = False)
    ### get the statistics per total port
    economic_risk_ensemble = economic_risk_data[['port_name','country','continent','risk','run','lat','lon','freight_total']].set_index(['port_name','country','continent','run', 'lat','lon','freight_total']).unstack(level = 'run').replace(np.nan,0)
    economic_risk_mean =  economic_risk_ensemble.mean(axis = 1).reset_index().rename(columns = {0:'risk_mean'})
    economic_risk_median =  economic_risk_ensemble.quantile(0.5, axis = 1).reset_index().rename(columns = {0.5:'risk_median'})
    economic_risk_low =  economic_risk_ensemble.quantile(0.05, axis = 1).reset_index().rename(columns = {0.05:'risk_low'})
    economic_risk_up =  economic_risk_ensemble.quantile(0.95, axis = 1).reset_index().rename(columns = {0.95:'risk_high'})

    economic_risk= economic_risk_mean.merge(economic_risk_low, on = ['port_name','country','continent','lat','lon','freight_total'])
    economic_risk= economic_risk.merge(economic_risk_up, on = ['port_name','country','continent','lat','lon','freight_total'])
    economic_risk= economic_risk.merge(economic_risk_median, on = ['port_name','country','continent','lat','lon','freight_total'])

    economic_risk['unc_risk'] = economic_risk['risk_high']  - economic_risk['risk_low']
    economic_risk['rel_unc_risk'] = economic_risk['unc_risk']/economic_risk['risk_mean']


    #### port assets
    port_risk_ensemble = economic_risk_data[['port_name','country','continent','risk_port','run','lat','lon','freight_total']].set_index(['port_name','country','continent','run', 'lat','lon','freight_total']).unstack(level = 'run').replace(np.nan,0)
    port_risk_mean =  port_risk_ensemble.mean(axis = 1).reset_index().rename(columns = {0:'risk_mean'})
    port_risk_median =  port_risk_ensemble.quantile(0.5, axis = 1).reset_index().rename(columns = {0.5:'risk_median'})
    port_risk_low =  port_risk_ensemble.quantile(0.05, axis = 1).reset_index().rename(columns = {0.05:'risk_low'})
    port_risk_up =  port_risk_ensemble.quantile(0.95, axis = 1).reset_index().rename(columns = {0.95:'risk_high'})

    port_risk= port_risk_mean.merge(port_risk_low, on = ['port_name','country','continent','lat','lon','freight_total'])
    port_risk= port_risk.merge(port_risk_up, on = ['port_name','country','continent','lat','lon','freight_total'])
    port_risk= port_risk.merge(port_risk_median, on = ['port_name','country','continent','lat','lon','freight_total'])

    port_risk['unc_risk'] = port_risk['risk_high']  - port_risk['risk_low']

    #### infra assets
    infra_risk_ensemble = economic_risk_data[['port_name','country','continent','risk_infra','run','lat','lon','freight_total']].set_index(['port_name','country','continent','run', 'lat','lon','freight_total']).unstack(level = 'run').replace(np.nan,0)
    infra_risk_mean =  infra_risk_ensemble.mean(axis = 1).reset_index().rename(columns = {0:'risk_mean'})
    infra_risk_median =  infra_risk_ensemble.quantile(0.5, axis = 1).reset_index().rename(columns = {0.5:'risk_median'})
    infra_risk_low =  infra_risk_ensemble.quantile(0.05, axis = 1).reset_index().rename(columns = {0.05:'risk_low'})
    infra_risk_up =  infra_risk_ensemble.quantile(0.95, axis = 1).reset_index().rename(columns = {0.95:'risk_high'})

    infra_risk= infra_risk_mean.merge(infra_risk_low, on = ['port_name','country','continent','lat','lon','freight_total'])
    infra_risk= infra_risk.merge(infra_risk_up, on = ['port_name','country','continent','lat','lon','freight_total'])
    infra_risk= infra_risk.merge(infra_risk_median, on = ['port_name','country','continent','lat','lon','freight_total'])

    infra_risk['unc_risk'] = infra_risk['risk_high']  - infra_risk['risk_low']

    #### operational
    operational_risk_ensemble = economic_risk_data[['port_name','country','continent','risk_additional','run','lat','lon','freight_total']].set_index(['port_name','country','continent','run', 'lat','lon','freight_total']).unstack(level = 'run').replace(np.nan,0)
    operational_risk_mean =  operational_risk_ensemble.mean(axis = 1).reset_index().rename(columns = {0:'risk_mean'})
    operational_risk_median =  operational_risk_ensemble.quantile(0.5, axis = 1).reset_index().rename(columns = {0.5:'risk_median'})
    operational_risk_low =  operational_risk_ensemble.quantile(0.05, axis = 1).reset_index().rename(columns = {0.05:'risk_low'})
    operational_risk_up =  operational_risk_ensemble.quantile(0.95, axis = 1).reset_index().rename(columns = {0.95:'risk_high'})

    operational_risk= operational_risk_mean.merge(operational_risk_low, on = ['port_name','country','continent','lat','lon','freight_total'])
    operational_risk= operational_risk.merge(operational_risk_up, on = ['port_name','country','continent','lat','lon','freight_total'])
    operational_risk= operational_risk.merge(operational_risk_median, on = ['port_name','country','continent','lat','lon','freight_total'])

    operational_risk['unc_risk'] = operational_risk['risk_high']  - operational_risk['risk_low']

    #### get the histogram of the different runs
    economic_risk_hist = economic_risk_ensemble.sum(axis = 0).reset_index().rename(columns = {0:'risk'})

    ### get the mean contribution across the runs
    N = economic_risk_data['run'].nunique()
    economic_risk_contribution_sum = economic_risk_data.groupby(['port_name','country','continent','lat','lon','freight_total'])['risk_port','risk_infra','risk_additional','contribution_port','contribution_infra','contribution_additional'].sum()
    economic_risk_contribution = economic_risk_contribution_sum /N
    economic_risk_contribution = economic_risk_contribution.reset_index()



    economic_risk.to_csv('Processed_output/Aggregate_risk_ensemble/Economic_risk.csv',index = False)
    port_risk.to_csv('Processed_output/Aggregate_risk_ensemble/Port_asset_risk.csv',index = False)
    infra_risk.to_csv('Processed_output/Aggregate_risk_ensemble/Infra_asset_risk.csv',index = False)
    operational_risk.to_csv('Processed_output/Aggregate_risk_ensemble/Logistic_losses_risk.csv',index = False)
    economic_risk_contribution.to_csv('Processed_output/Aggregate_risk_ensemble/Economic_risk_contribution.csv',index = False)
    economic_risk_hist.to_csv('Processed_output/Histograms/economic_risk_histogram.csv',index = False)

    #return economic_risk, economic_risk_hist, economic_risk_contribution



path_output_data = 'Output/'

### set a color per hazard
df_color = pd.DataFrame({'hazard':['TC','earthquake','operational','pluvial','fluvial','coastal'], 'color':['darkred','darkgreen','sienna','indigo','darkblue','darkorange']})

process_output = 'yes'

if process_output == 'yes':
    #### hazard-specific risk estimates


    ### economic risk contribition
    #economic_risk, economic_risk_hist, economic_risk_contribution = process_risk_contribution(path_output_data)
    process_risk_contribution(path_output_data)

    print('contribution done')

    ### trade data
    #trade_mean, trade_direct_hist, trade_indirect_hist = process_trade_data(path_output_data)
    process_trade_data(path_output_data)

    print('trade done')
    
    #hazard_mean = process_hazard_output(path_output_data, df_color)
    process_hazard_output(path_output_data, df_color)
    print('hazard done')
    #### resilience
    #resilience_data = process_resilience_data(path_output_data)
    process_resilience_data(path_output_data)
    print('resilience done')


else:
    #### read the data in
    None
