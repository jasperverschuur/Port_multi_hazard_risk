#!/usr/bin/env python
# coding: utf-8
#from dask import dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gumbel_r
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def temperature_processing(path, ID_list, ME_list, threshold, unc = None):
    """Calculate number of times the temperature threshold is exceeded

    Parameters
    ----------
    path : path to input data
    ID : ID of the port as defined in port_centroids_ID file
    threshold : operational threshold set

    unc : int (optional)
        None if doing expected value
        value to add the run identifier to the dataframe for the uncertainty analysis

    Returns
    ------
    temp_exceed_port : pandas dataframe
        Dataframe with the port and total number of threshold exceedances + annual frequency
    """

    temp_df_unc = pd.DataFrame()
    for ID in ID_list:
        if ID in ME_list:
            threshold = threshold + 10
        else:
            threshold = threshold

        ### read file
        temp_data = xr.open_dataset(path+'tasmax_daily_era5_1979-2020_port'+str(ID)+'.nc').mx2t.to_dataframe().reset_index()
        temp_data_update = xr.open_dataset(path+'update_tasmax_daily_era5_2000to2006_port'+str(ID)+'.nc').mx2t.to_dataframe().reset_index()
        temp_data['year'] = pd.to_datetime(temp_data['time']).dt.year
        temp_data_update['year'] = pd.to_datetime(temp_data_update['time']).dt.year
        ### remove 1979, some wrong input in ERA
        years_correct = [2000, 2001, 2002, 2003, 2004, 2005,2006]
        temp_data = temp_data[temp_data['year']>1979]
        temp_data = temp_data[~temp_data['year'].isin(years_correct)]

        temp_data = pd.concat([temp_data, temp_data_update], ignore_index = True, sort = False).sort_values(by = 'time')
        ### convert K to Celsius
        temp_data['Celsius'] =temp_data['mx2t']-273.15
        ### exceedance
        #if ID in ME_list:
            #print('yes')
            #temp_data['exceed'] = np.where(temp_data['Celsius'] >= threshold+10,1,0)
        #else:
        temp_data['exceed'] = np.where(temp_data['Celsius'] >= threshold,1,0)


        temp_exceed_port = pd.DataFrame({'ID':[ID]})
        temp_exceed_port['exceed'] = temp_data['exceed'].sum()
        temp_exceed_port['threshold'] = threshold
        temp_exceed_port['annual'] = np.round(temp_exceed_port['exceed']/temp_data['year'].nunique(),2)

        ### add run of sensitivity analysis if defined
        if unc != None:
            temp_exceed_port['run'] = unc
        else:
            temp_exceed_port['run'] = 0

        temp_df_unc = pd.concat([temp_df_unc, temp_exceed_port], ignore_index = True, sort = False)

    ### output
    temp_df_unc = temp_df_unc[temp_df_unc['exceed']>0]
    return temp_df_unc

def precipitation_processing(path, ID, threshold, unc = None):
    """Calculate number of times the temperature threshold is exceeded

    Parameters
    ----------
    path : path to input data
    ID : ID of the port as defined in port_centroids_ID file
    threshold : operational threshold set

    unc : int (optional)
        None if doing expected value
        value to add the run identifier to the dataframe for the uncertainty analysis

    Returns
    ------
    temp_exceed_port : pandas dataframe
        Dataframe with the port and total number of threshold exceedances + annual frequency
    """
    ### read file
    prec_data = xr.open_dataset(path+'tp_daily_era5_1979-2020_port'+str(ID)+'.nc').tp.to_dataframe().reset_index()
    prec_data_update = xr.open_dataset(path+'update_tp_daily_era5_2000to2006_port'+str(ID)+'.nc').tp.to_dataframe().reset_index()

    prec_data['year'] = pd.to_datetime(prec_data['time']).dt.year
    prec_data_update['year'] = pd.to_datetime(prec_data_update['time']).dt.year
    ### remove 1979, some wrong input in ERA
    years_correct = [2000, 2001, 2002, 2003, 2004, 2005,2006]
    prec_data = prec_data[prec_data['year']>1979]
    prec_data = prec_data[~prec_data['year'].isin(years_correct)]

    prec_data = pd.concat([prec_data, prec_data_update], ignore_index = True, sort = False).sort_values(by = 'time')
    ### convert m/day to mm/day
    prec_data['mm_day'] =prec_data['tp']*1000

    ### exceedance
    prec_data['exceed'] = np.where(prec_data['mm_day'] >= threshold,1,0)
    prec_exceed_port = pd.DataFrame({'ID':[ID]})
    prec_exceed_port['exceed'] = prec_data['exceed'].sum()
    prec_exceed_port['threshold'] = threshold
    prec_exceed_port['annual'] = np.round(prec_exceed_port['exceed']/prec_data['year'].nunique(),2)

    ### add run of sensitivity analysis if defined
    if unc != None:
        prec_exceed_port['run'] = unc
    else:
        prec_exceed_port['run'] = 0
    return prec_exceed_port

def wind_processing(path,ID_list,threshold,unc = None):
    """Calculate number of times the temperature threshold is exceeded

    Parameters
    ----------
    path : path to input data
    ID : ID of the port as defined in port_centroids_ID file
    threshold : operational threshold set

    unc : int (optional)
        None if doing expected value
        value to add the run identifier to the dataframe for the uncertainty analysis

    Returns
    ------
    temp_exceed_port : pandas dataframe
        Dataframe with the port and total number of threshold exceedances + annual frequency
    """

    wind_df_unc = pd.DataFrame()
    for ID in ID_list:
        ### read file
        wind_data = xr.open_dataset(path+'windspd_10m_daily_era5_1979-2020_port'+str(ID)+'.nc').uv10spd.to_dataframe().reset_index()
        wind_data_update = xr.open_dataset(path+'update_windspd_10m_daily_era5_2000to2006_port'+str(ID)+'.nc').uv10spd.to_dataframe().reset_index()

        wind_data['year'] = pd.to_datetime(wind_data['time']).dt.year
        wind_data_update['year'] = pd.to_datetime(wind_data_update['time']).dt.year
        ### remove 1979, some wrong input in ERA
        years_correct = [2000, 2001, 2002, 2003, 2004, 2005,2006]
        wind_data = wind_data[wind_data['year']>1979]
        wind_data = wind_data[~wind_data['year'].isin(years_correct)]

        wind_data = pd.concat([wind_data, wind_data_update], ignore_index = True, sort = False).sort_values(by = 'time')
        wind_data = wind_data[wind_data['year']>1979]
        wind_data['wind_10m'] =wind_data['uv10spd']

        ### exceedance
        wind_data['exceed'] = np.where(wind_data['wind_10m'] >= threshold, 1 , 0)
        wind_exceed_port = pd.DataFrame({'ID': [ID]})
        wind_exceed_port['exceed'] = wind_data['exceed'].sum()
        wind_exceed_port['threshold'] = threshold
        wind_exceed_port['annual'] = np.round(wind_exceed_port['exceed']/wind_data['year'].nunique(),2)

        ### add run of sensitivity analysis if defined
        if unc != None:
            wind_exceed_port['run'] = unc
        else:
            wind_exceed_port['run'] = 0

        ### concat
        wind_df_unc = pd.concat([wind_df_unc, wind_exceed_port], ignore_index = True, sort = False)
    ### output
    wind_df_unc = wind_df_unc[wind_df_unc['exceed']>0]
    return wind_df_unc



def design_conditions(max_wave_new_surge, rp_wave):
    """Estimate the design wave and surge parameter per breakwater

        Parameters
        ----------
        max_wave_new_surge : input dataframe with wave and surge time series
        rp_wave: return period of the design wave height

        Returns
        ------
        df_design : Dataframe with design wave height, wave period, wave direction and storm surge height
        """

    ### set the probability
    prob = 1-(1/rp_wave)
    #### get the design criteria
    df_design = pd.DataFrame()
    for test_id in max_wave_new_surge['id'].unique():
        test_df = max_wave_new_surge[max_wave_new_surge['id']==test_id].sort_values(by = 'H_input',ascending = False)
        test_df = test_df[test_df['swh']<9]
        test_df_yearly_max_model = test_df
        ### peak threshold
        peak_thres = test_df_yearly_max_model.head(108) ### 4 events per year ~ 99 percentile
        ### annual max
        annual_max = test_df_yearly_max_model.drop_duplicates(subset = ['year'],keep = 'first')
        ### get the relationship between T and Hs
        if (annual_max['H_input'].max() == 0) or (len(annual_max)==0):
            continue
        else:
            a = np.median(peak_thres[peak_thres['H_input']>0]['mwp']/np.sqrt(peak_thres[peak_thres['H_input']>0]['H_input']))

            loc, scale = gumbel_r.fit(annual_max['H_input'].values)

            ### extract wave return period
            h_design=gumbel_r.ppf(prob,  loc=loc, scale=scale)
            ### get the T and alpha
            alpha_design = annual_max[annual_max['alpha_input']>0]['alpha_input'].median()
            T_design = a*np.sqrt(h_design)
            surge_design = annual_max['tide_surge'].max()
            df_100 = pd.DataFrame({'H_design':[h_design],'T_design':[T_design],'alpha_design':[alpha_design],'S_design':[surge_design]})
            df_100['id']= test_id
            df_100['type_bw'] = test_df_yearly_max_model['type_bw'].iloc[0]
            df_rp_id =df_100
        df_design = pd.concat([df_design,df_rp_id])

    df_design = df_design[df_design['alpha_design'].notna()]
    df_design = df_design[df_design['T_design'].notna()]
    df_design = df_design[df_design['H_design']>0]
    return df_design


def design_freeboard(df_design, H_design, T_design, S_design, alpha_design,Q_coast,Q_revet,r=0.5, A=9.39e-3, B=21.6, W_hs=1.7):
    """Estimate the design freeboard of the breakwater

        Parameters
        ----------
        df_design : dataframe with input data
        H_design : column name of design wave height
        T_design : column name of design wave period
        S_design : column name of design storm surge
        alpha_design : column name of design wave direction
        Q_coast : overtopping discharge for coastal breakwater
        Q_revet : overtopping discharge for revetments and seawalls


        r, A, B, W_hs are additional design parameters that are fixed

        Returns
        ------
        Rc : Freeboard height for wave conditions only
        Rc_surge : Freeboard height for wave + surge conditions
        """

    ### get the design parameters
    alpha_design_input = np.where(df_design[alpha_design] > 80, 80, df_design[alpha_design])
    O = 1 - (0.0033 * alpha_design_input)
    Q_val = np.where(df_design['type_bw'].isin(['OR', 'revetment_stone', 'revetment']), Q_revet,Q_coast)
    Cr = np.where(df_design['type_bw'].isin(['OR', 'revetment_stone', 'revetment']), 1.0, (3.06 * np.exp(-1.5 * W_hs)))

    ### apply correction factors
    Q_design = Q_val * (1 / O) * (1 / Cr)

    Q_star = (Q_design) / (df_design[T_design] * 9.81 * df_design[H_design])
    R_star = np.log(Q_star / A) * (r / -B)

    R_star_correct = np.where(R_star > 0.3, 0.3, R_star)
    R_star_correct = np.where(R_star < 0.05, 0.05, R_star)
    Rc = R_star_correct * (df_design[T_design] * ((9.81 * df_design[H_design]) ** (0.5)))
    Rc_surge = Rc + df_design[S_design]
    return Rc, Rc_surge


def overtopping_volume(df_input, H, T, S, alpha, R_design, r=0.5, A=9.39e-3, B=21.6, W_hs=1.7):
    """Estimate the overtopping discharge of the breakwater

            Parameters
            ----------
            df_input : dataframe with input data
            H : column name of wave height
            T : column name of wave period
            S : column name of surge
            alpha : column name of wave direction
            R_design: the design freeboard

            r, A, B, W_hs are additional design parameters that are fixed

            Returns
            ------
            Q : overtopping discharge
            R_star : relative freeboard height
            """

    alpha_input = np.where(df_input[alpha] > 80, 80, df_input[alpha])
    O = 1 - (0.0033 * alpha_input)
    Cr = np.where(df_input['type_bw'].isin(['OR', 'revetment_stone', 'revetment']), 1.0, (3.06 * np.exp(-1.5 * W_hs)))

    R_relative = df_input[R_design] - df_input[S]
    R_star = R_relative / (df_input[T] * ((9.81 * df_input[H]) ** 0.5))
    R_star_correct = np.where(R_star > 0.3, 0.3, R_star)
    R_star_correct = np.where(R_star < 0.05, 0.05, R_star)

    Q_star = A * np.exp(-B * R_star_correct / r)
    Q = Q_star * df_input[T] * 9.81 * df_input[H] * Cr * O

    return Q, R_star


def overtopping_processing(wave_input_path, wave_input_day_path, rp_wave, Q_coast, Q_revet, uncertainty_height, unc=None):
    """process the overtopping analaysis

                Parameters
                ----------
                wave_input : dataframe with input data
                rp_wave : return period of the design wave height
                Q_coast : overtopping discharge coastal breakwater
                Q_revet : overtopping discharge revetments/seawalls
                uncertainty_height: design parameters for quality of breakwater

                unc : int (optional)
                     None if doing expected value
                     value to add the run identifier to the dataframe for the uncertainty analysis
                Returns
                ------
                exceedance : dataframe with the yearly frequency of breakwater overtopping per port

            """
    print('Overtopping', unc)
    wave_input_month = pd.read_csv(wave_input_path, low_memory=False)
    wave_input_day  = pd.read_csv(wave_input_day_path, low_memory=False)
    #wave_input_day = wave_input_day[wave_input_day['H_input']>2]
    #wave_input_day.to_csv(wave_input_day_path, index = False)
    #wave_input_day = wave_input_day.compute()

    #### get the design parameters
    design_parameters = design_conditions(wave_input_month, rp_wave)

    ### estimate the design
    design_parameters['Rc'], design_parameters['Rc_surge'] = design_freeboard(design_parameters, 'H_design', 'T_design',
                                                                              'S_design', 'alpha_design', Q_coast,Q_revet)
    wave_input = wave_input_day[
        ['id', 'year', 'time', 'type_bw', 'port_name', 'country', 'continent', 'lat', 'lon', 'time', 'swh', 'mwd',
         'mwp', 'H_input', 'alpha_input', 'tide_surge']]

    ### delete
    del wave_input_month
    del wave_input_day
    ## merge the design parameters on id: id is the breakwater id
    wave_input = wave_input.merge(design_parameters[['id', 'Rc', 'Rc_surge']], on='id')

    ## add the design discharge
    wave_input['Q_design'] = np.where(wave_input['type_bw'].isin(['OR', 'revetment_stone', 'revetment']), Q_revet,Q_coast)

    ## add the uncertainty for the sensitivity analysis, standard set to 1.0
    wave_input['Rc_surge_unc'] = wave_input['Rc_surge'] * uncertainty_height

    ## run Q and R_star
    wave_input['Q'], wave_input['R_star'] = overtopping_volume(wave_input, 'H_input', 'swh', 'tide_surge',
                                                               'alpha_input', 'Rc_surge_unc')

    ### find exceedance
    wave_input['exceed'] = np.where(wave_input['Q'] > wave_input['Q_design'], 1, 0)

    ### surge data only till 2014
    #wave_input = wave_input[wave_input['year'] != 2015]

    exceedance = wave_input.groupby(['id', 'type_bw', 'port_name', 'country', 'continent', 'lat', 'lon'])[
        'exceed'].sum().reset_index()

    exceedance['annual'] = np.round(exceedance['exceed'] / wave_input.year.nunique(), 2)
    exceedance['threshold'] = uncertainty_height
    ### only return the most exposed breakwater per port
    exceedance = exceedance.sort_values(by='exceed', ascending=False).drop_duplicates(
        subset=['port_name', 'country', 'continent']).drop(columns=['id', 'lat', 'lon'])
    if unc != None:
        exceedance['run'] = unc
    else:
        exceedance['run'] = 0
    exceedance = exceedance[exceedance['exceed']>0]
    return exceedance


def snell_law(d0, d, H0, T0, alpha0):
    """Process Snell's law on wave data

            Parameters
            ----------
            d0 : depth in deep water
            d: depth at point of interest
            H0: wave height in deep water
            T0: wave period in deep water
            alpha0: wave direction in deep water

            Returns
            ------
            H_channel: wave height at point of interest
            alpha_channel : wave direction at point of interest
            """

    degree_rad = np.pi/180
    d_center = np.abs(d0)
    d_channel = np.abs(d)
    H_center = H0
    alpha_center = alpha0
    Tm = T0
    #####
    L0 = 9.81*Tm**2 / (2*np.pi)
    L_center = L0*(np.tanh((2*np.pi*np.sqrt(d_center/9.81)/Tm)**(3/2)))**(2/3)
    L_channel = L0*(np.tanh((2*np.pi*np.sqrt(d_channel/9.81)/Tm)**(3/2)))**(2/3)

    C_center = L_center/Tm
    k_center = (2*np.pi)/L_center
    kh_center = k_center*d_center

    C_channel = L_channel/Tm
    k_channel = (2*np.pi)/L_channel
    kh_channel = k_channel*d_channel

    n_center = 0.5*(1+(2*kh_center/np.sinh(2*kh_center)))
    Cg_center = n_center*C_center

    n_channel = 0.5*(1+(2*kh_channel/np.sinh(2*kh_channel)))
    Cg_channel = n_channel*C_channel

    Ks = np.sqrt(Cg_center/Cg_channel)


    alpha_channel = np.arcsin(np.sin(alpha_center*degree_rad)*L_channel/L_center)*(180/np.pi)

    Kr = np.where(alpha_channel>0,(np.sqrt(np.cos(alpha_center*degree_rad)/np.cos(alpha_channel*degree_rad))),0)

    H_channel = H_center * Ks * Kr

    ### check if wave breaking occurs
    H_channel = np.where(H_channel > (d_channel/1.28),0,H_channel)
    H_channel = np.where(H_channel == np.nan,0,H_channel)

    return H_channel,alpha_channel


def wave_processing(path_wave, wave_input_path, years, wave_ME, wave_HE, unc=None):
    """Calculate number of times the wave threshold is exceeded

    Parameters
    ----------
    path_wave : path to input wave data
    wave_input : input file with necessary parameters per port
    year: year to extract data for
    wave_ME: wave threshold for moderately exposed port
    wave_HE: wave threshold for highly exposed port

    unc : int (optional)
        None if doing expected value
        value to add the run identifier to the dataframe for the uncertainty analysis

    Returns
    ------
    temp_exceed_port : pandas dataframe
        Dataframe with the port and total number of threshold exceedances + annual frequency
    """
    wave_input = pd.read_csv(wave_input_path)


    ### loop over the years
    wave_height_df = pd.DataFrame()
    for year in years:
    #print(wave_input.head(5))
        file = pd.read_csv(path_wave + 'ERA5_ports_wave_height_' + str(year) + '.csv')
        file['month'] = file['month'].astype(str).apply(lambda x: x.zfill(2))
        file['day'] = file['day'].astype(str).apply(lambda x: x.zfill(2))
        file['time'] = file['year'].astype(str) + '-' + file['month'] + '-' + file['day']
        file = file.drop_duplicates(subset=['z', 'time'], keep='first')
        wave_input_subset = wave_input[['z', 'id', 'HARBORTYPE', 'type_bw', 'port_name', 'country', 'continent', 'depth_center', 'x_wave_center',
             'y_wave_center', 'angle', 'angle_bw', 'angle_channel', 'depth_channel', 'wave_channel']]

        #### merge wave data on port data
        wave_input_subset = wave_input_subset.merge(file[['z', 'time', 'swh', 'mwp', 'mwd']], on=['z'])

        wave_input_subset = wave_input_subset[wave_input_subset['swh'] != -9999.00000]
        rot = wave_input_subset
        rot['incidence'] = rot['mwd'] - wave_input_subset['angle_channel']
        rot['incidence'] = np.where(rot['incidence'] < 0, 360 + rot['incidence'], rot['incidence'])
        rot['alpha'] = np.abs(rot['incidence'] - 270)
        #### remove max values
        rot = rot[rot['swh'] < 9]

        ### run the function
        H_channel, alpha_channel = snell_law(rot['depth_center'], rot['depth_channel'], rot['swh'], rot['mwp'],
                                             rot['alpha'])
        rot['H_channel'] = H_channel
        rot['alpha_bw'] = alpha_channel
        rot['H_channel'] = np.where(rot['H_channel'].isna(), 0, rot['H_channel'])

        ### set the thresholds
        rot['threshold'] = np.where(rot['wave_channel'] == 'HE', wave_HE, wave_ME)

        ### find exceedance
        rot['exceed'] = np.where(rot['H_channel'] > rot['threshold'], 1, 0)
        rot['year'] = year
        exceedance = rot.groupby(['port_name', 'country', 'continent', 'HARBORTYPE', 'wave_channel', 'year'])[
            'exceed'].sum().reset_index()

        if unc != None:
            exceedance['run'] = unc
        else:
            exceedance['run'] = 0
        ### add to df
        wave_height_df = pd.concat([wave_height_df, exceedance],ignore_index = True, sort = False)

    #### aggregate
    exceedance_wave = wave_height_df.groupby(['port_name','country','continent','wave_channel','HARBORTYPE','run'])['exceed'].sum().reset_index()
    exceedance_wave['annual'] = np.round(exceedance_wave['exceed']/len(years),2)
    exceedance_wave['threshold'] = wave_ME
    exceedance_wave = exceedance_wave[exceedance_wave['exceed']>0]
    return exceedance_wave
