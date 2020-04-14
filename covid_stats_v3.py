# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:55:40 2020

@author: Erick
"""
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import confidence as cf
from scipy.linalg import svd
import matplotlib.gridspec as gridspec
import os
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import EngFormatter
import seir_model
import datetime
import h5py
from scipy.stats import chisquare


location = 'Mexico'

beta_intervention_factor = np.array([0.9,0.5,0.1])
mild_percent = 0.8
severe_percent = 0.15
critical_percent = 0.05

add_days = 365
latency_time = 5.1

plot_pbands = True
xpos = -50
ypos = -60
start_idx = 1
before_date = datetime.datetime(year=2020, month=4, day=9)
inflection_point1 = 8
inflection_point2 = 11
inflection_point3 = 20
inflection_point4 = 20

save_results = False
b0 = [1.0, 1.0, 1.0]
b1 = [0.01, 1.0, 1.0]
#b_lower_limit = [-10,-np.log10(30),-np.log10(60),-10] # Italy
b_lower_limit = [-50,-np.log10(365),-10] # 
all_tol = np.finfo(np.float64).eps*1

results_folder = './SEIR_v3'
csv_results = 'fitting_results_seir.csv'

data_color = 'C0'
removed_color = 'C4'
this_year = datetime.datetime.now().strftime('%Y')

df_confd = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_death = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df_recvd = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

if location != 'global':
    df_beds_wb = pd.read_excel('http://api.worldbank.org/v2/en/indicator/SH.MED.BEDS.ZS?downloadformat=excel', skiprows=3)
    df_country_codes = pd.read_csv('./UN_country_Codes.csv', usecols=['Country or Area', 'M49 Code','ISO-alpha3 Code']).rename(columns={'Country or Area': 'Country', 'M49 Code': 'M49','ISO-alpha3 Code': 'ISO3-Code'})
    df_country_codes = df_country_codes[df_country_codes['M49'].str.isnumeric()].astype({'M49': 'int16'})
    df_beds_acute_who_max_year = pd.read_excel('https://dw.euro.who.int/api/v3/export/download/a8cb2f20d3e74b75a38769fb44c2dc9b', sheet_name='Data (table)', usecols=['COUNTRY_REGION', 'YEAR','VALUE']).rename(columns={'COUNTRY_REGION':'ISO3-Code'}).groupby(['ISO3-Code']).max()['YEAR'].reset_index()
    df_beds_acute_who = pd.merge(pd.read_excel('https://dw.euro.who.int/api/v3/export/download/a8cb2f20d3e74b75a38769fb44c2dc9b', sheet_name='Data (table)', usecols=['COUNTRY_REGION', 'YEAR','VALUE']).rename(columns={'COUNTRY_REGION':'ISO3-Code'}), df_beds_acute_who_max_year, on=['ISO3-Code', 'YEAR'], how='inner')
    df_who_codes = pd.read_excel('https://dw.euro.who.int/api/v3/export/download/a8cb2f20d3e74b75a38769fb44c2dc9b', sheet_name='Countries').rename(columns={'ISO 3': 'ISO3-Code'})
    
    df_beds_acute_oecd_max_year = pd.read_csv('OECD_acute_care_beds.csv', usecols=['LOCATION', 'TIME', 'Value']).rename(columns={'LOCATION': 'ISO3-Code', 'TIME': 'YEAR', 'Value': 'VALUE'}).groupby(['ISO3-Code']).max()['YEAR'].reset_index()
    df_beds_acute_oecd = pd.merge(pd.read_csv('OECD_acute_care_beds.csv', usecols=['LOCATION', 'TIME', 'Value']).rename(columns={'LOCATION': 'ISO3-Code', 'TIME': 'YEAR', 'Value': 'beds per1000'}), df_beds_acute_oecd_max_year, on=['ISO3-Code', 'YEAR'], how='inner')

df_un_population = pd.read_excel('https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/EXCEL_FILES/1_Population/WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx', sheet_name='ESTIMATES', skiprows=16, usecols=['Region, subregion, country or area *','Country code',this_year]).rename(columns={this_year: 'population', 'Country code': 'M49'})

if location != 'global':
    df_beds_acute_who_merged = pd.merge(df_country_codes, df_beds_acute_who, on='ISO3-Code', how='inner')
    df_beds_acute_who_merged = pd.merge(df_beds_acute_who_merged, df_un_population, on='M49', how='inner')
    df_beds_acute_who_merged['beds per1000'] = df_beds_acute_who_merged['VALUE']/df_beds_acute_who_merged['population']
    df_beds_acute_who = df_beds_acute_who_merged.filter(['ISO3-Code','YEAR','beds per1000'])
    
    df_beds_acute = df_beds_acute_who.append(df_beds_acute_oecd)
    df_beds_acute_max_year = df_beds_acute.groupby(['ISO3-Code']).max()['YEAR'].reset_index()
    df_beds_acute = pd.merge(df_beds_acute, df_beds_acute_max_year, on=['ISO3-Code', 'YEAR'], how='inner')

    del df_beds_acute_who_max_year
    del df_who_codes
    del df_beds_acute_oecd_max_year
    del df_beds_acute_max_year
    del df_beds_acute_who
    del df_beds_acute_oecd
    del df_beds_acute_who_merged

xfmt = ScalarFormatter(useMathText=True)
xfmt.set_powerlimits((-3,3))
engfmt = EngFormatter(places=1, sep=u"\N{THIN SPACE}")  # U+2009
datefmt = mpl.dates.DateFormatter("%m/%d")
datefmt_yr = mpl.dates.DateFormatter("%Y/%m/%d")
engfmt_ax = EngFormatter(places=0, sep=u"\N{THIN SPACE}")  # U+2009

if location == 'United States':
    alt_loc = 'US'
elif location == 'South Korea':
    alt_loc = 'Korea, South'
else:
    alt_loc = location

if location == 'Iran':
    country_name = 'Iran (Islamic Republic of)'
elif location == 'United States':
    country_name = 'United States of America'
elif location == 'United Kingdom':
    country_name = 'United Kingdom of Great Britain and Northern Ireland'
elif location == 'South Korea' :
    country_name = 'Republic of Korea'
elif location == 'Russia':
    country_name = 'Russian Federation'
else:
    country_name = location
    




def fobj_seir(p: np.ndarray, time: np.ndarray, infected_: np.ndarray, 
             removed_: np.ndarray, population: float, latency_time_: float,
             I0_: int, R0_: int = 0):
    p = np.power(10, p)
    sol = seir_model.seir_model(time, N=population, beta=p[0], gamma=p[1], 
                                sigma=1/latency_time_, I0=I0_, R0=R0_, E0=float(p[2]))
    y = sol.sol(time)
    S, E, I, R = y
    
    n = len(infected_)
    residual = np.empty(n*2)
    for i in range(n):
        residual[i] = I[i] - infected_[i] #np.log10(I[i]+1) - np.log10(infected[i]+1)
        residual[i+n] = R[i] - removed_[i] #np.log10(R[i]+1) - np.log10(removed[i]+1)
    return residual

def fobj_seir_e0(p: np.ndarray, time: np.ndarray, infected_: np.ndarray, 
             removed_: np.ndarray, population: float, latency_time_: float,
             gamma_: float,
             I0_: int, R0_: int = 0, E0_: int = 0):
    p = np.power(10, p)
    sol = seir_model.seir_model(time, N=population, beta=p[0], gamma=gamma_, 
                                sigma=1/latency_time_, I0=I0_, R0=R0_, E0=E0_)
    y = sol.sol(time)
    S, E, I, R = y
    
    n = len(infected_)
    residual = np.empty(n*2)
    for i in range(n):
        residual[i] = I[i] - infected_[i] #np.log10(I[i]+1) - np.log10(infected[i]+1)
        residual[i+n] = R[i] - removed_[i] #np.log10(R[i]+1) - np.log10(removed[i]+1)
    return residual

def seir(time: np.ndarray, p: np.ndarray, population_: float, latency_time_: float, I0_: int, R0_: int = 0):
    p = np.power(10, p)
    sol = seir_model.seir_model(time, N=population_, beta=p[0], gamma=p[1], 
                                sigma=1/latency_time_, I0=I0_, R0=R0_, E0=float(p[2]))
    y = sol.sol(time)
    S, E, I, R = y
    points = len(time)
    res = np.zeros((points, 2),dtype=np.float)
    for n, i, r in zip(range(points), I, R):
        res[n] = (i,r)
    return res

def seir_tot(time: np.ndarray, p: np.ndarray, population_: float, latency_time_: float, I0_: int, R0_: int = 0):
    p = np.power(10, p)
#    t = np.linspace(np.amin(time),np.amax(time),1000)
    sol = seir_model.seir_model(time, N=population_, beta=p[0], gamma=p[1], 
                                sigma=1/latency_time_, I0=I0_, R0=R0_, E0=float(p[2]))
    y = sol.sol(time)
    S, E, I, R = y
    points = len(time)
    res = np.zeros((points, 4),dtype=np.float)
    for n, s, e, i, r in zip(range(points), S, E, I, R):
        res[n] = (s, e, i, r)
    return res

def fobj_seir_wvd(p: np.ndarray, time: np.ndarray, infected_: np.ndarray, 
             removed_: np.ndarray, population_: float, latency_time: float, 
             birth_rate: float, I0_: int, R0_: int = 0):
    p = np.power(10, p)
    sol = seir_model.seir_model_wvd(time, N=population_, beta=p[0], gamma=p[1], 
                                    sigma=1/latency_time, mu=p[2], nu=p[2], 
                                    I0=I0_, R0=R0_, E0=p[3])
    y = sol.sol(time)
    S, E, I, R = y
    
    n = len(infected_)
    residual = np.empty(n*2)
    for i in range(n):
        residual[i] = I[i] - infected_[i] #np.log10(I[i]+1) - np.log10(infected[i]+1)
        residual[i+n] = R[i] - removed_[i] #np.log10(R[i]+1) - np.log10(removed[i]+1)
    return residual

def seir_wvd(time: np.ndarray, p: np.ndarray, population_: float, 
             latency_time_: float, birth_rate: float, I0_: int, R0_: int = 0):
    p = np.power(10, p)
    sol = seir_model.seir_model_wvd(time, N=population_, beta=p[0], gamma=p[1], 
                                    sigma=1/latency_time_, mu=p[2], nu=p[2], 
                                    I0=I0_, R0=R0_, E0=p[3])
    y = sol.sol(time)
    S, E, I, R = y
    points = len(time)
    res = np.zeros((points, 2),dtype=np.float)
    for n, i, r in zip(range(points), I, R):
        res[n] = (i,r)
    return res

def seir_tot_wvd(time: np.ndarray, p: np.ndarray, population_: float, 
                 latency_time_: float, birth_rate: float, I0_: int, R0_: int = 0):
    p = np.power(10, p)
#    t = np.linspace(np.amin(time),np.amax(time),1000)
    sol = seir_model.seir_model_wvd(time, N=population_, beta=p[0], gamma=p[1], 
                                    sigma=1/latency_time_, mu=p[2], nu=p[2], 
                                    I0=I0_, R0=R0_, E0=p[3])
    y = sol.sol(time)
    S, E, I, R = y
    points = len(time)
    res = np.zeros((points, 4),dtype=np.float)
    for n, s, e, i, r in zip(range(points), S, E, I, R):
        res[n] = (s, e, i, r)
    return res

def fobj_pop(p: np.ndarray, t:np.ndarray, value: np.ndarray):
    return p[0]*t + p[1] - value

def jac_pop(p: np.ndarray, t:np.ndarray, value: np.ndarray):
    rows = len(t)
    jac = np.empty((rows,2))
    for i, v in enumerate(t):
        jac[i] = (v, 1)
    return jac

def pop_model(t:np.ndarray, p: np.ndarray):
    return p[0]*t + p[1]

defaultPlotStyle = {'font.size': 12,
                     'font.family': 'Arial',
                     'font.weight': 'regular',
                    'legend.fontsize': 12,
                    'mathtext.fontset': 'stix',
#                    'mathtext.rm': 'Times New Roman',
#                    'mathtext.it': 'Times New Roman:italic',#'Arial:italic',
#                    'mathtext.cal': 'Times New Roman:italic',#'Arial:italic',
#                    'mathtext.bf': 'Times New Roman:bold',#'Arial:bold',
                    'xtick.direction' : 'in',
                    'ytick.direction' : 'in',
                    'xtick.major.size' : 4.5,
                    'xtick.major.width' : 1.75,
                    'ytick.major.size' : 4.5,
                    'ytick.major.width' : 1.75,
                    'xtick.minor.size' : 2.75,
                    'xtick.minor.width' : 1.0,
                    'ytick.minor.size' : 2.75,
                    'ytick.minor.width' : 1.0,
                    'ytick.right' : False,
                    'lines.linewidth'   : 2.5,
                    'lines.markersize'  : 10,
                    'lines.markeredgewidth'  : 0.85,
                    'axes.labelpad'  : 5.0,
                    'axes.labelsize' : 14,
                    'axes.labelweight' : 'regular',
                    'legend.handletextpad' : 0.2,
                    'legend.borderaxespad' : 0.2,
                    'axes.linewidth': 1.25,
                    'axes.titlesize' : 14,
                    'axes.titleweight' : 'bold',
                    'axes.titlepad' : 6,
                    'figure.titleweight' : 'bold',
                    'figure.dpi': 100}

def get_rsquared(res, infected_, removed_):
    residuals = res.fun
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.concatenate([infected_,removed_])-np.mean(np.concatenate([infected_,removed_])))**2)
    return 1.0 - (ss_res/ss_tot)
    
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def latex_format(x: float, digits: int = 2):
    if np.isinf(x):
        return '$\infty$'
    digits = abs(digits)
    fmt_dgts = '%%.%df' % digits
    fmt_in = '%%.%dE' % digits
    x_str = fmt_in % x
    x_sci = (np.array(x_str.split('E'))).astype(np.float)
    if abs(x_sci[1]) <= 3:
        fmt_dgts = '%%.%df' % (digits)
        return fmt_dgts % x
    if digits == 0:
        return r'$\mathregular{10^{%d}}$' % x_sci[1]
    else:
        ltx_str = fmt_dgts % x_sci[0]
        ltx_str += r'$\mathregular{\times 10^{%d}}$' % x_sci[1]
        return ltx_str
    
covid_type = np.dtype([('date', 'M8[ns]'),
                       ('confirmed', 'u8'),
                       ('recovered', 'u8'),
                       ('dead', 'u8'),
                       ('infected', 'u8')])
    
    
seir_fit_dtype = np.dtype([
        ('Country/Region', 'U50'),
        ('beta1', 'd'),
        ('beta1_ci_l', 'd'),
        ('beta1_ci_u', 'd'),
        ('gamma1', 'd'),
        ('gamma1_ci_l', 'd'),
        ('gamma1_ci_u', 'd'),
        ('E01', 'd'),
        ('E01_ci_l', 'd'),
        ('E0_ci_u', 'd'),
        ('I01', 'i8'),
        ('R01', 'i8'),
        ('reproductive number1', 'd'),
        ('ndays1', 'i'),
        ('beta2', 'd'),
        ('beta2_ci_l', 'd'),
        ('beta2_ci_u', 'd'),
        ('gamma2', 'd'),
        ('gamma2_ci_l', 'd'),
        ('gamma2_ci_u', 'd'),
        ('E02', 'd'),
        ('E02_ci_l', 'd'),
        ('E02_ci_u', 'd'),
        ('I02', 'i8'),
        ('R02', 'i8'),
        ('reproductive number2', 'd'),
        ('ndays2', 'i'),
        ('beta3', 'd'),
        ('beta3_ci_l', 'd'),
        ('beta3_ci_u', 'd'),
        ('gamma3', 'd'),
        ('gamma3_ci_l', 'd'),
        ('gamma3_ci_u', 'd'),
        ('E03', 'd'),
        ('E03_ci_l', 'd'),
        ('E03_ci_u', 'd'),
        ('I03', 'i8'),
        ('R03', 'i8'),
        ('reproductive number3', 'd'),
        ('ndays3', 'i'),
        ('latency', 'd'),
        ('start_idx', 'i8'),
        ('inflection_point1', 'i'),
        ('inflection_point2', 'i'),
        ('inflection_point3', 'i'),
        ('inflection_point4', 'i'),
        ('before_day', 'i'),
        ('fit date' , 'M8[ns]'),
        ('beta1_guess', 'd'),
        ('gamma1_guess', 'd'),
        ('E01_guess', 'd'),
        ('rsquared1', 'd'),
        ('chisquared1', 'd'),
        ('p1', 'd'),
        ('beta2_guess', 'd'),
        ('gamma2_guess', 'd'),
        ('E02_guess', 'd'),
        ('rsquared2', 'd'),
        ('chisquared2', 'd'),
        ('p2', 'd'),
        ('rsquared3', 'd'),
        ('chisquared3', 'd'),
        ('p3', 'd'),
        ('recovery_rate', 'd'),
        ('death_rate','d'),
        ('peak_date_current_beta', 'M8[ns]'),
        ('peak_value_current_beta', 'd'),
        ('peak_date_b90', 'M8[ns]'),
        ('peak_value_b90', 'd'),
        ('peak_date_b50', 'M8[ns]'),
        ('peak_value_b50', 'd'),
        ('peak_date_b10', 'M8[ns]'),
        ('peak_value_b10', 'd'),
        ])
    
seir_curvfit_type = np.dtype([
            ('time', 'i8'),
            ('infected', 'd'),
            ('infected_lpb', 'd'),
            ('infected_upb', 'd'),
            ('removed', 'd'),
            ('removed_lpb', 'd'),
            ('removed_upb', 'd')
        ])
    
seir_projection_type = np.dtype([
            ('time', 'i8'),
            ('susceptible', 'd'),
            ('exposed', 'd'),
            ('infected', 'd'),
            ('removed', 'd'),
        ])

if __name__ == '__main__':
    
    mpl.rcParams.update(defaultPlotStyle)
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if location != 'global':
        country_code = df_country_codes[df_country_codes['Country'] == country_name]['ISO3-Code'].item()
        m49 = df_country_codes[df_country_codes['Country'] == country_name]['M49'].item()
    
    # Estimate the natality rate
#    df_un_population_curve = pd.read_excel('https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/EXCEL_FILES/1_Population/WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx', sheet_name='ESTIMATES', skiprows=16).rename(columns={'Country code': 'M49'})
#    df_un_population_curve_country = df_un_population_curve[df_un_population_curve['M49'] == int(m49)]
#    del df_un_population_curve
#    df_un_population_curve_country_cols = df_un_population_curve_country[df_un_population_curve_country.columns[7::]].T.reset_index()
#    del df_un_population_curve_country
#    pop_cols = df_un_population_curve_country_cols.columns
#    df_un_population_curve_country_cols = df_un_population_curve_country_cols.rename(columns={pop_cols[0]: 'year', pop_cols[1]: 'population'})
#    pop_year = np.array(df_un_population_curve_country_cols['year'], dtype=np.float)
#    pop_x = pop_year - pop_year[0]
#    pop_value = np.array(df_un_population_curve_country_cols['population'], dtype=np.float)*1000
#    res_pop = optimize.least_squares(fobj_pop, [1,0.1], bounds=([0,0], [np.inf, np.inf]), 
#                                     args=(pop_x, pop_value), jac=jac_pop, verbose=2,
#                                     xtol=all_tol,
#                                     ftol=all_tol,
#                                     gtol=all_tol)
#    
#    popt0 = res_pop.x
#    
#    natality_rate = popt0[0]/365
    dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y')
    if location != 'global':
        df_confd_country = df_confd[df_confd['Country/Region'] == alt_loc].groupby("Country/Region").sum()
        df_confd_country = df_confd_country[df_confd_country.columns[2::]].T.reset_index()
        df_confd_country = df_confd_country.rename(columns={alt_loc: 'confirmed',
                                                            'index':'date'})
    
        df_confd_country['date'] = pd.to_datetime(df_confd_country['date'], format='%m/%d/%y')
        
        df_death_country = df_death[df_death['Country/Region'] == alt_loc].groupby("Country/Region").sum()
        df_death_country = df_death_country[df_death_country.columns[2::]].T.reset_index()
        df_death_country = df_death_country.rename(columns={alt_loc: 'dead',
                                                            'index':'date'})
        
        df_death_country['date'] = pd.to_datetime(df_death_country['date'], format='%m/%d/%y')
        
        df_recvd_country = df_recvd[df_recvd['Country/Region'] == alt_loc].groupby("Country/Region").sum()
        df_recvd_country = df_recvd_country[df_recvd_country.columns[2::]].T.reset_index()
        df_recvd_country = df_recvd_country.rename(columns={alt_loc: 'recovered',
                                                            'index':'date'})
    
        df_recvd_country['date'] = pd.to_datetime(df_recvd_country['date'], format='%m/%d/%y')
        
        df_full = pd.merge(df_confd_country, df_recvd_country,
                       how="outer").fillna(0)
    
        df_full = pd.merge(df_full, df_death_country,
                           how="outer").fillna(0).reset_index().drop(columns=['index'])
        
        df_beds_country = df_beds_wb[df_beds_wb['Country Code'] == country_code]
        df_beds_country = df_beds_country[df_beds_country.columns[4::]].T.reset_index().dropna()
        df_beds_cols = list(df_beds_country.columns)
        df_beds_new_cols = ['year', 'beds']
        df_beds_col_map = {on: nn for on, nn in zip(df_beds_cols, df_beds_new_cols)}
        df_beds_country = df_beds_country.rename(columns = df_beds_col_map)
        beds_per_k = float(df_beds_country[df_beds_country['year'] == df_beds_country['year'].max()]['beds'])
        
        found_country_acute = len(df_beds_acute[df_beds_acute['ISO3-Code'] == country_code])
        if found_country_acute:
            df_acute_beds_country = float(df_beds_acute[df_beds_acute['ISO3-Code'] == country_code]['beds per1000'])
        
        
        population = float(df_un_population[df_un_population['M49'] == m49]['population'])*1000
        
    else:
        df_confd_global = df_confd[df_confd.columns[4::]].sum()
        df_confd_global = df_confd_global.T.reset_index()
        df_confd_global = df_confd_global.rename(columns={0: 'confirmed',
                                                            'index':'date'})
    
        df_death_global = df_death[df_death.columns[4::]].sum()
        df_death_global = df_death_global.T.reset_index()
        df_death_global = df_death_global.rename(columns={0: 'dead',
                                                            'index':'date'})
    
        df_recvd_global = df_recvd[df_recvd.columns[4::]].sum()
        df_recvd_global = df_recvd_global.T.reset_index()
        df_recvd_global = df_recvd_global.rename(columns={0: 'recovered',
                                                            'index':'date'})
    
    
        df_full = pd.merge(df_confd_global, df_recvd_global,
                       how="outer").fillna(0)
    
        df_full = pd.merge(df_full, df_death_global,
                           how="outer").fillna(0).reset_index().drop(columns=['index'])
        
        population = float(df_un_population[df_un_population['M49'] == 900]['population'])*1000
    
    
    df_full = df_full.eval('infected = confirmed - recovered - dead')
    df_full['date'] = pd.to_datetime(df_full['date'], format='%m/%d/%y')
    
    df_full = df_full[df_full['infected']>0].reset_index().drop(columns=['index'])
    
#    df = pd.read_csv(csv_data) 
#    df = df[df['location'] == location]
#    covid_by_country = df.sort_values(by=['date'])
##    covid_by_country = covid_by_country[0:-1]
    cv_date = pd.to_datetime(df_full['date']).to_numpy()
    time_s = 1E-9*(cv_date - np.amin(cv_date))
    time_days = np.array([t/86400 for t in time_s], dtype=float)
    
    
    
    infected = df_full['infected'].to_numpy()
    recovered = df_full['recovered'].to_numpy()
    confirmed = df_full['confirmed'].to_numpy()
    dead = df_full['dead'].to_numpy()
    t0 = datetime.datetime.utcfromtimestamp(cv_date[0].astype(datetime.datetime)*1E-9)
    
#    I0 = infected[0]
#    id_start = np.argmin(infected<=I0) - 2
#    range_msk = np.zeros_like(confirmed, dtype=bool)
#    for i,v in enumerate(confirmed):
#        if i >= id_start:
#            range_msk[i] = True
#    
#    
#    
#    time_days = time_days[range_msk]
#    infected = infected[range_msk]
#    recovered = recovered[range_msk]
#    confirmed = confirmed[range_msk]
#    dead = dead[range_msk]
#    cv_date = cv_date[range_msk]
    removed = dead + recovered
    
    
    time_days_fit = time_days[start_idx::]
    infected_fit = infected[start_idx::]
    recovered_fit = recovered[start_idx::]
    confirmed_fit = confirmed[start_idx::]
    dead_fit = dead[start_idx::]
    cv_date_fit = cv_date[start_idx::]
    t0_fit = datetime.datetime.utcfromtimestamp(cv_date_fit[0].astype(datetime.datetime)*1E-9)
    
    removed_fit = dead_fit + recovered_fit
    
    
    idx_gtz = confirmed>0
    recovery_rate = recovered[-1]/confirmed[-1]
    death_rate = dead[-1]/confirmed[-1]
    before_day = (before_date - t0).days
    
    if before_day > 0:
        idx = time_days_fit <= before_day
        time_days_fit = time_days_fit[idx]
        infected_fit = infected_fit[idx]
        recovered_fit = recovered_fit[idx]
        confirmed_fit = confirmed_fit[idx]
        dead_fit = dead_fit[idx]
        cv_date_fit = cv_date_fit[idx]
        removed_fit = removed_fit[idx]
    
    idx1 = time_days_fit < time_days_fit[inflection_point1]
    if inflection_point3 > inflection_point2:
        idx2 = np.logical_and(time_days_fit >= time_days_fit[inflection_point2] , time_days_fit <= time_days_fit[inflection_point3])
    else:
        idx2 = time_days_fit >= time_days_fit[inflection_point2]
    time1 = time_days_fit[idx1]
    time2 = time_days_fit[idx2]
    infected1 = infected_fit[idx1]
    infected2 = infected_fit[idx2]
    removed1 = removed_fit[idx1]
    removed2 = removed_fit[idx2]
    
    
    n_days1 = len(time1)
    n_days2 = len(time2)
    
    
    res1 = optimize.least_squares(fobj_seir,np.log10(np.array(b0)), 
                                  jac='3-point',
                                  bounds=(b_lower_limit,[1,0,np.log10(population)]),
                                  args=(time1,infected1, removed1,
                                        population, latency_time,
                                        infected1[0],removed1[0]),
                                  xtol=all_tol,
                                  ftol=all_tol,
                                  gtol=all_tol,
#                                 x_scale='jac',
#                                 loss='soft_l1', f_scale=0.1,
#                                 loss='cauchy', f_scale=0.1,
                                  max_nfev=n_days1*1000,
                                  verbose=2)
   
    res2 = optimize.least_squares(fobj_seir,np.log10(np.array(b1)), 
                                  jac='3-point',
                                  bounds=(b_lower_limit,[10,10,np.log10(population)]),
                                  args=(time2,infected2, removed2,
                                        population, latency_time, 
                                        infected2[0],removed2[0]),
                                  xtol=all_tol,
                                  ftol=all_tol,
                                  gtol=all_tol,
                                  max_nfev=n_days2*1000,
                                  verbose=2)
                                  
    
    
    popt1_log = res1.x
    popt2_log = res2.x
    
    popt1 = np.power(10, popt1_log)
    popt2 = np.power(10, popt2_log)
    
    
    ysize1 = len(res1.fun)
    cost = 2 * res1.cost  # res.cost is half sum of squares!
    s_sq = cost / (ysize1 - popt1.size)
    
    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res1.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res1.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov1_log = np.dot(VT.T / s**2, VT)
    pcov1_log = pcov1_log * s_sq
    
#    print(pcov1_log)
   

    if pcov1_log is None:
        # indeterminate covariance
        print('Failed estimating pcov1')
        pcov1_log = np.zeros((len(pcov1_log), len(pcov1_log)), dtype=float)
        pcov1_log.fill(np.inf)
        
        
    ysize2 = len(res2.fun)
    cost2 = 2 * res2.cost  # res.cost is half sum of squares!
    s_sq = cost / (ysize2 - popt2.size)
    
    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res2.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res2.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov2_log = np.dot(VT.T / s**2, VT)
    pcov2_log = pcov2_log * s_sq
    
#    print(pcov1_log)
   

    if pcov2_log is None:
        # indeterminate covariance
        print('Failed estimating pcov2')
        pcov2_log = np.zeros((len(pcov2_log), len(pcov2_log)), dtype=float)
        pcov2_log.fill(np.inf)
        
    if inflection_point3 > inflection_point2:
        idx3 = time_days_fit >= time_days_fit[inflection_point4]
        time3 = time_days_fit[idx3]
        infected3 = infected_fit[idx3]
        removed3 = removed_fit[idx3]
        n_days3 = len(time3)
        
        
        time_ = np.linspace(np.amin(time2), np.amin(time3), 500)
        sol = seir_model.seir_model(time_, N=population, beta=popt1[0], 
                                    gamma=popt1[1], sigma=1/latency_time, 
                                    I0=infected1[0], R0=removed1[0], 
                                    E0=popt1[2])
        y_ = sol.sol(time_)
        s_, e_, i_, r_ = y_
        
        res3 = optimize.least_squares(fobj_seir, np.log10(np.array(b1)),#[popt2_log[0], popt2_log[1], np.log10(e_[-1])],#np.log10(np.array(b1)), 
                                      jac='3-point',
                                      bounds=(b_lower_limit,[10,10,np.log10(population)]),
                                      args=(time3,infected3, removed3,
                                            population, latency_time, 
                                            infected3[0],removed3[0]),
                                      xtol=all_tol,
                                      ftol=all_tol,
                                      gtol=all_tol,
                                      max_nfev=n_days3*1000,
                                      verbose=2)
                                      
        
        popt3_log = res3.x
        popt3 = np.power(10, popt3_log)
        ysize3 = len(res3.fun)
        cost = 2 * res3.cost  # res.cost is half sum of squares!
        s_sq = cost / (ysize3 - popt3.size)
        
        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = svd(res3.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res3.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov3_log = np.dot(VT.T / s**2, VT)
        pcov3_log = pcov3_log * s_sq
    
        if pcov3_log is None:
            # indeterminate covariance
            print('Failed estimating pcov1')
            pcov1_log = np.zeros((len(pcov3_log), len(pcov3_log)), dtype=float)
            pcov1_log.fill(np.inf)
        
        ci3 = np.power(10,cf.confint(ysize3,popt3_log,pcov3_log))
        
        time_pred3 = np.linspace(np.amin(time3), np.amax(time3), 200)
        def smodel3(x,p): return seir(x, p, population_=population, latency_time_=latency_time, I0_=infected3[0], R0_=removed3[0])
        ypred3,lpb3,upb3 = cf.predint_multi(time_pred3, time3, infected3, smodel3, res3, mode='functional')
        
        yp3 = seir_tot(time3, popt3_log, latency_time_=latency_time, population_=population, I0_=infected3[0], R0_=removed3[0])
        idx_chi31 = yp3[:,2] > 0
        idx_chi32 = yp3[:,3] > 0
        
        chisq3, p3 = chisquare(f_obs=np.concatenate([infected3[idx_chi31], removed3[idx_chi32]]), f_exp=np.concatenate([yp3[idx_chi31,2], yp3[idx_chi32,3]]), ddof=len(popt3))
        rsquared3 = get_rsquared(res3, infected3, removed3)
    else:
        popt3 = np.zeros(3)
        ci3 = np.zeros((3,2))
        chisq3, p3, rsquared3 = 0, 0, 0
        n_days3 = 0
        
#    pcov_seir = np.power(10, pcov1_log)
    
    time_pred1 = np.linspace(np.amin(time1), np.amax(time1), 500)
    time_pred2 = np.linspace(np.amin(time2), np.amax(time2), 500)
    ci1 = np.power(10,cf.confint(ysize1,popt1_log,pcov1_log))
    ci2 = np.power(10,cf.confint(ysize2,popt2_log,pcov2_log))
    
    """
     seir_tot_wvd(time: np.ndarray, p: np.ndarray, population_: float, 
                 latency_time_: float, birth_rate: float, I0_: int, R0_: int = 0)
     """
    def smodel1(x,p): return seir(x, p, population_=population, latency_time_=latency_time, I0_=infected1[0], R0_=removed1[0])
    def smodel2(x,p): return seir(x, p, population_=population, latency_time_=latency_time, I0_=infected2[0], R0_=removed2[0])
    ypred1,lpb1,upb1 = cf.predint_multi(time_pred1, time1, infected1, smodel1, res1, mode='functional')
    ypred2,lpb2,upb2 = cf.predint_multi(time_pred2, time2, infected2, smodel2, res2, mode='functional')
    
    yp1 = seir_tot(time1, popt1_log, latency_time_=latency_time, population_=population, I0_=infected1[0], R0_=removed1[0])
    yp2 = seir_tot(time2, popt2_log, latency_time_=latency_time, population_=population, I0_=infected2[0], R0_=removed2[0])
    
    idx_chi11 = yp1[:,2] > 0
    idx_chi12 = yp1[:,3] > 0
    idx_chi21 = yp2[:,2] > 0
    idx_chi22 = yp2[:,3] > 0
    chisq1, p1 = chisquare(f_obs=np.concatenate([infected1[idx_chi11], removed1[idx_chi12]]), f_exp=np.concatenate([yp1[idx_chi11,2], yp1[idx_chi12,3]]), ddof=len(popt1))
    chisq2, p2 = chisquare(f_obs=np.concatenate([infected2[idx_chi21], removed2[idx_chi22]]), f_exp=np.concatenate([yp2[idx_chi21,2], yp2[idx_chi22,3]]), ddof=len(popt2))
    
    rsquared1 = get_rsquared(res1, infected1, removed1)
    rsquared2 = get_rsquared(res2, infected2, removed2)
    
    print('Estimating projected infections...')
    
    
    
    if inflection_point3 > inflection_point2:
        time_projected = np.linspace(np.amin(time3), np.amax(time3)+add_days, 1000)
        yprojected = seir_tot(time_projected, [popt3_log[0], popt2_log[1], popt2_log[2]], latency_time_=latency_time, population_=population, I0_=infected3[0], R0_=removed3[0])
        ypred_all = seir_tot(time_pred3, [popt3_log[0], popt2_log[1], popt2_log[2]], latency_time_=latency_time, population_=population, I0_=infected3[0], R0_=removed3[0])
        beta_intervention = popt3[0]*beta_intervention_factor
        xintervened = np.linspace(np.max(time3), np.amax(time3)+add_days, 1000)
        gamma3 = popt3[1]
        E03 = popt3[2]
    else:
        time_projected = np.linspace(np.amin(time2), np.amax(time2)+add_days, 1000)
        yprojected = seir_tot(time_projected, popt2_log, latency_time_=latency_time, population_=population, I0_=infected2[0], R0_=removed2[0])
        ypred_all = seir_tot(time_pred2, popt2_log, latency_time_=latency_time, population_=population, I0_=infected2[0], R0_=removed2[0])
        beta_intervention = popt2[0]*beta_intervention_factor
        xintervened = np.linspace(np.max(time2), np.amax(time2)+add_days, 1000)
    
        gamma3 = popt2[1]
        E03 = popt2[2]
    
    intervened_max_infections = np.zeros(4)
    intervened_peak_time = np.zeros(4, dtype=np.dtype('M8[ns]'))
    infections_peak = np.amax(yprojected[:,2])
    intervened_max_infections[0] = infections_peak
    peak_date_days = np.amin(time_projected[yprojected[:,2] == infections_peak])
    peak_datetime = np.datetime64(t0_fit + datetime.timedelta(days=peak_date_days), 'ns')
    intervened_peak_time[0] = peak_datetime
    
    for i, bi in enumerate(beta_intervention):
        sol = seir_model.seir_model(xintervened, N=population, beta=bi, 
                                    gamma=gamma3, sigma=1/latency_time, 
                                    I0=ypred_all[-1,2], R0=ypred_all[-1,3], 
                                    E0=E03)
        yi = sol.sol(xintervened)
        _, _, infected_i, _ = yi
        infections_peak = np.amax(infected_i)
        intervened_max_infections[i+1] = infections_peak
        peak_date_days = np.amin(xintervened[infected_i == infections_peak])
        peak_datetime = np.datetime64(t0_fit + datetime.timedelta(days=peak_date_days), 'ns')
        intervened_peak_time[i+1] = peak_datetime
        
        
    yintervened = sol.sol(xintervened)
    _, _, I_intervened, R_intervened = yintervened
    
    dates_proj = np.array([cv_date_fit[0] + np.timedelta64(int(x), 'D') for x in time_projected])
    
    
    
    
    
    R01_cal = popt1[0]/popt1[1]
    R02_cal = popt2[0]/popt2[1]
    if inflection_point3 > inflection_point2:
        R03_cal = popt3[0]/popt3[1]
    else:
        R03_cal = 0
    
    
#    fig_pop = plt.figure()
#    fig_pop.set_size_inches(4.5,4.0,forward=True)
#    fig_pop.subplots_adjust(hspace=0.1, wspace=0.1)
#    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_pop, width_ratios=[1])
#    
#    ax0 = fig_pop.add_subplot(gs0[0, 0]) 
#    ax0.plot(pop_x, pop_value, 'o', label='UN data', fillstyle='none',)
#    ax0.plot(pop_x, pop_model(pop_x, res_pop.x), '-', label='Fit')
#    ax0.set_xlabel('$t$ (years)')
#    ax0.set_ylabel('Population')
#    ax0_y_lim = (pop_year[0] + ax0.get_xlim()[0],
#                 pop_year[0] + ax0.get_xlim()[1])
#    ax0_y = ax0.twiny()
#    ax0_y.xaxis.tick_top()
#    ax0_y.set_xlim(ax0_y_lim)
#    ax0.yaxis.set_major_formatter(engfmt)
#    ax0.set_title('Population in {0}'.format(location))
#    ax0.text(0.05, 0.95, '$N=a_0 t + a_1$\n$a_0 =${0} (#/year)\n$a_1 =$ {1} (#)'.format(latex_format(popt0[0],2),latex_format(popt0[1],2)), 
#         horizontalalignment='left',
#         verticalalignment='top', 
#         transform=ax0.transAxes,
#         fontsize=12,
#         color='k',
#         zorder=6)
#    plt.tight_layout()
#    plt.show()
    
    
    fig = plt.figure()
    fig.set_size_inches(6.5,6.5,forward=True)
    fig.subplots_adjust(hspace=0.75, wspace=0.5)
    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1])
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=2, 
                                            subplot_spec = gs0[0])
    
    
    ax1 = fig.add_subplot(gs00[0,0])
    ax2 = fig.add_subplot(gs00[0,1])
    ax3 = fig.add_subplot(gs00[1,0])
    ax4 = fig.add_subplot(gs00[1,1])
    
    
    
    
    if isinstance(data_color, str):
        model2_color = mpl.colors.to_rgb('C0')
        data_color = mpl.colors.to_rgb(data_color)
        removed_color = mpl.colors.to_rgb(removed_color)
        
    
    pband_color = lighten_color(data_color, 0.25)
    pband_color2 = lighten_color(removed_color, 0.25)
#    ax1.fill_between(time1,lpb,upb, color=pband_color)
#    ax1.plot(time_days, infected,'o', color=data_color, fillstyle='none')
#    ax1.plot(time1, ypred, color=data_color)
    
    ax1.plot(time_days, confirmed,'o', color='tab:orange', fillstyle='full', label='total')
    ax1.plot(time_days, recovered,'s', color='tab:green', fillstyle='none', label='recovered')
    ax1.plot(time_days, dead,'^', color='tab:red', fillstyle='none', label='dead')
    ax1.plot(time_days, infected,'o', color=data_color, fillstyle='none', label='infected')
    
    per1000_factor = 1000/population
    
    if plot_pbands:
        ax2.fill_between(time_pred1,lpb1[:,0]*per1000_factor,upb1[:,0]*per1000_factor, color=pband_color)
        ax2.fill_between(time_pred1,lpb1[:,1]*per1000_factor,upb1[:,1]*per1000_factor, color=pband_color2)
        ax2.fill_between(time_pred2,lpb2[:,0]*per1000_factor,upb2[:,0]*per1000_factor, color=pband_color)
        ax2.fill_between(time_pred2,lpb2[:,1]*per1000_factor,upb2[:,1]*per1000_factor, color=pband_color2)
    ax2.plot(time_days_fit, infected_fit*per1000_factor,'o', color=data_color, fillstyle='none', label='infected')
    ax2.plot(time_days_fit, removed_fit*per1000_factor,'v', color=removed_color, fillstyle='none', label='removed')
    ax2.plot(time_pred1, ypred1[:,0]*per1000_factor, ':', color=model2_color)
    ax2.plot(time_pred1, ypred1[:,1]*per1000_factor, ':', color=removed_color)
    ax2.plot(time_pred2, ypred2[:,0]*per1000_factor, color=model2_color)
    ax2.plot(time_pred2, ypred2[:,1]*per1000_factor, color=removed_color)
    if inflection_point3 > inflection_point2:
        if plot_pbands:
            ax2.fill_between(time_pred3,lpb3[:,0]*per1000_factor,upb3[:,0]*per1000_factor, color=pband_color)
            ax2.fill_between(time_pred3,lpb3[:,1]*per1000_factor,upb3[:,1]*per1000_factor, color=pband_color2)
        ax2.plot(time_pred3, ypred3[:,0]*per1000_factor,'--', color=model2_color)
        ax2.plot(time_pred3, ypred3[:,1]*per1000_factor, '--',color=removed_color)
#    ax2_b = ax2.twinx()
#    ax3.tick_params(axis='y', labelcolor='C1')
    
    ax2.axvline(x=inflection_point1, ls='--', lw=1.0, color='tab:grey')
    ax2.axvline(x=inflection_point2, ls='--', lw=1.0, color='tab:grey')
    ax2.axvline(x=inflection_point3, ls='--', lw=1.0, color='tab:grey')
    ax2.axvline(x=inflection_point4, ls='--', lw=1.0, color='tab:grey')
    
#    ax3.plot(time_projected, yprojected[:,0], ls=':', color='C6', label='suceptible')
#    ax3.plot(time_projected, yprojected[:,1], ls='-.', color='C7', label='exposed')
    infected_res =  yprojected[:,2]
    removed_res = yprojected[:,3]
    imax_seir = np.amax(infected_res)
    rmax_seir = np.amax(removed_res)
    confirmed_seir = infected_res + removed_res
     
    tmax_seir = int(time_projected[infected_res == imax_seir])
    peak_date = t0_fit + datetime.timedelta(days=tmax_seir)
    
#    peak_str = 'Date = {0}\nCases: {1}'.format(peak_date.strftime('%Y/%m/%d'),
#                       engfmt.format_data(imax_seir))
    peak_str = 'Date = {0}'.format(peak_date.strftime('%Y/%m/%d'))
    
    ax3.plot(time_projected, yprojected[:,2]*per1000_factor, ls='-', color=data_color, label='infected')
    ax3.plot(time_projected, yprojected[:,3]*per1000_factor, ls='--', color=removed_color, label='removed')
    ax3.plot(time_projected, confirmed_seir*recovery_rate*per1000_factor, color='tab:green', label='recovered')
#    ax3.plot(time_projected, confirmed_seir*death_rate*per1000_factor, color='tab:red', label='dead')
    
    # Projected cases
    cmap = mpl.cm.get_cmap('cool')
    normalize = mpl.colors.Normalize(vmin=0, vmax=3)
    ax4_colors = [cmap(normalize(t)) for t in range(1,4)]
    
    projected_infections = infected_res*per1000_factor
    mild_infections = projected_infections*0.8
    severe_infections = projected_infections*0.15
    critical_infections = projected_infections*0.05
    
    # Intervention
    projected_infections_intervened = 1000*I_intervened/population
    mild_infections_intervened = projected_infections_intervened*mild_percent
    severe_infections_intervened = projected_infections_intervened*severe_percent
    critical_infections_intervened = projected_infections_intervened*critical_percent
    
#    ax4.plot(time_projected, projected_infections, ls='-', color=data_color, label='Total')
    ax4.plot(time_projected, mild_infections, ls='-', color=ax4_colors[0], label='Mild')
    ax4.plot(time_projected, severe_infections, ls='-', color=ax4_colors[1], label='Severe')
    ax4.plot(time_projected, critical_infections, ls='-', color=ax4_colors[2], label='Acute')
    
    ax4.plot(xintervened, mild_infections_intervened, ls='--', color=ax4_colors[0], label='Mild')
    ax4.plot(xintervened, severe_infections_intervened, ls='--', color=ax4_colors[1], label='Severe')
    ax4.plot(xintervened, critical_infections_intervened, ls='--', color=ax4_colors[2], label='Acute')
    
        
    
    if location != 'global':
        if beds_per_k < np.amax(mild_infections)*2:
            ax4.axhline(y=beds_per_k, color='k', ls='--', lw=1.75)
        #    if found_country_acute:
        #        ax4.axhline(y=df_acute_beds_country, color=lighten_color(ax4_colors[2], 0.75), ls='-', lw=1.75)
        #    ax4.yaxis.set_major_locator(locmaj2)
        #    ax4.yaxis.set_minor_locator(locmin2)
        #    ax4.yaxis.set_major_formatter(engfmt_ax)
            ax4.text(np.amax(time_projected), beds_per_k*0.75, 'Hospital beds',
                     horizontalalignment='right',
                     verticalalignment='top', 
                     fontsize=11,
                     color='k',#lighten_color(ax4_colors[1], 1.0),
                     zorder=6)
    #    if found_country_acute:
    #        ax4.text(np.amin(time_projected), df_acute_beds_country*0.25, 'Acute care beds',
    #                 horizontalalignment='left',
    #                 verticalalignment='bottom', 
    #                 fontsize=11,
    #                 color=lighten_color(ax4_colors[2], 0.75),
    #                 zorder=6)
    
    
    ax4.text(0.05, 0.95, 'Mild',
             horizontalalignment='left',
             verticalalignment='top', 
             transform=ax4.transAxes,
             fontsize=12,
             color=ax4_colors[0])
    
    ax4.text(0.05, 0.85, 'Severe',
             horizontalalignment='left',
             verticalalignment='top',  
             transform=ax4.transAxes,
             fontsize=12,
             color=ax4_colors[1])
    
    ax4.text(0.05, 0.75, 'Critical',
             horizontalalignment='left',
             verticalalignment='top',  
             transform=ax4.transAxes,
             fontsize=12,
             color=ax4_colors[2])
    
    
#    ax4.text(0.95,0.95, '$\\beta =$ {0} (1/day)'.format(latex_format(popt1[0],3)),
#             horizontalalignment='right',
#             verticalalignment='top',  
#             transform=ax4.transAxes,
#             fontsize=11,
#             color='k')
#    
#    
#    ax4.text(0.95,0.05, '$\\beta =$ {0} (1/day)'.format(latex_format(beta_intervention,3)),
#             horizontalalignment='right',
#             verticalalignment='bottom',  
#             transform=ax4.transAxes,
#             fontsize=11,
#             color='k')
    
    
    
    ax3.text(0.05, 0.95, 'Infections',
             horizontalalignment='left',
             verticalalignment='top', 
             transform=ax3.transAxes,
             fontsize=12,
             color='tab:blue')
    
    ax3.text(0.05, 0.85, 'Removed',
             horizontalalignment='left',
             verticalalignment='top',  
             transform=ax3.transAxes,
             fontsize=12,
             color='tab:purple')
    
    ax3.text(0.05, 0.75, 'Recoveries',
             horizontalalignment='left',
             verticalalignment='top',  
             transform=ax3.transAxes,
             fontsize=12,
             color='tab:green')
    
    
    
#    ax3.annotate(peak_str,
#            xy=(tmax_seir, imax_seir*per1000_factor), xycoords='data',
#            xytext=(xpos, ypos), textcoords='offset points',
#            color='tab:blue',
#            fontsize=11,
#            arrowprops=dict(arrowstyle="->",
#                            color='k',
#                            connectionstyle="angle3,angleA=0,angleB=90"))
    
    
    leg_colors = ['tab:orange','tab:green','tab:red','C0']
    leg1 = ax1.legend(loc='upper left',frameon=False)
    for i, text in enumerate(leg1.get_texts()):
        text.set_color(leg_colors[i])
        
    leg_colors2 = [data_color,removed_color]
    leg2 = ax2.legend(loc='upper left',frameon=False)
    for i, text in enumerate(leg2.get_texts()):
        text.set_color(leg_colors2[i])
    
    
    
    ax1.set_xlabel('Days since {0}'.format(t0.strftime('%Y/%m/%d')), fontsize=12, color='tab:grey')
    ax1.xaxis.set_label_position('top') 
    ax1.set_ylabel('Total Cases')
            
    ax2.set_xlabel('Days since {0}'.format(t0_fit.strftime('%Y/%m/%d')), fontsize=12, color='tab:grey')
    ax2.xaxis.set_label_position('top') 
    ax2.set_ylabel('Cases per 1000')                    
            
    ax3.set_xlabel('Days since {0}'.format(t0_fit.strftime('%Y/%m/%d')), fontsize=12, color='tab:grey')
    ax3.xaxis.set_label_position('top') 
    ax3.set_ylabel('Cases per 1000')#, color='C1'
            
    ax4.set_xlabel('Days since {0}'.format(t0_fit.strftime('%Y/%m/%d')), fontsize=12, color='tab:grey')
    ax4.xaxis.set_label_position('top') 
    ax4.set_ylabel('Cases per 1000')#, color='C1'
            
        
    if location != 'global':
        ax1.set_title('Cases in {0}'.format(location))
    else:
        ax1.set_title('Global Cases'.format(location))

    
#    res_str = '$\\beta$ = {0} (days)$^{{-1}}$\n95% CI: [{1},{2}]\n'.format(latex_format(popt1[0],3),
#                  latex_format(ci1[0][0],3), latex_format(ci1[0][1],3))
#    res_str += '$\\gamma$ = {0} (days)$^{{-1}}$\n95% CI: [{1},{2}]\n'.format(latex_format(popt1[1],3),
#                    latex_format(ci1[1][0],3), latex_format(ci1[1][1],3))
#    res_str += '$\\sigma$ = {0} (days)$^{{-1}}$\n95% CI: [{1},{2}]\n'.format(latex_format(popt1[2],3),
#                    latex_format(ci1[2][0],3), latex_format(ci1[2][1],3))
##    res_str += '$I_0 =$  {0:.3g}\n95% CI: [{1:.3g},{2:.3g}]'.format(popt1[2],
##                    ci1[2][0], ci1[2][1])
#    res_str += '$I_0 =$  {0:.3g}, $R_0$ = {1:.0f}, $E_0 =$ {2:.2g}, $t_0 =$ {3:.0f}'.format(infected1[0], 
#                         removed1[0], popt1[3], time_days_fit[0])
    
#    ax2.text(0.05, 0.1, res_str, 
#             horizontalalignment='left',
#             verticalalignment='bottom', 
#             transform=ax2.transAxes,
#             fontsize=10,
#             color='k',
#             zorder=6)
    
    res_str = '$\\beta$ = {0} (days)$^{{-1}}$\n'.format(latex_format(popt1[0],3))
    res_str += '$\\gamma$ = {0} (days)$^{{-1}}$\n'.format(latex_format(popt1[1],4))
#    res_str += '$\\sigma$ = {0} (days)$^{{-1}}$\n'.format(latex_format(popt1[2],4))
    
    res_str += '$I_0 =$  {0:.3g}, $R_0$ = {1:.0f}, $E_0 =$ {2}, $t_0 =$ {3:.0f}'.format(infected1[0], 
                         removed1[0], latex_format(popt1[2],0), time_days_fit[0])
    
#    ax2.text(0.95, 0.05, res_str, 
#         horizontalalignment='right',
#         verticalalignment='bottom', 
#         transform=ax2.transAxes,
#         fontsize=9,
#         color='k',
#         zorder=6)
    
    ax3_info_str = 'Population: {0}\n$\\mathcal{{R}}_0 = $ {1}'.format(
            engfmt(population),
            latex_format(R01_cal,2))
    ax3_info_str += "\nDeath rate = {0:.3f} %".format(death_rate*100)
    
#    if location == 'China':
#        ax3_info_str = 'Population: {0}'.format(
#            engfmt(population))
#        ax3_info_str += "\nDeath rate = {0:.3f} %".format(death_rate*100)
#        ax3_info_str += "\nRecovery rate = {0:.3f} %".format(recovery_rate*100)
    
    ax3.text(0.95,0.05, ax3_info_str,
            horizontalalignment='right',
            verticalalignment='bottom', 
            transform=ax3.transAxes,
            fontsize=10,
            color='k',
            zorder=6)
    
        
    
    ax2.set_title('SEIR model fit')

    
    ax3.set_title('Projected Infections')
    ax4.set_title('Severity')
    
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    
    
    
#    ax4.set_yscale('log')
    
    if location == 'China' or location == 'global':
        ax1.set_ylim(
                top=np.amax(confirmed)*100, 
                bottom=np.amax([0,np.amin(confirmed),np.amin(recovered), np.amin(dead)])
                )
    

    
    ax2.set_ylim(
            top=np.amax(ypred2[:]*per1000_factor)*10, 
            bottom=max(1E-5,np.amin(ypred1[:]*per1000_factor))
            )
    
    if inflection_point3 > inflection_point2:
        ax2.set_ylim(
            top=np.amax(ypred3[:]*per1000_factor)*10, 
            bottom=max(1E-5,np.amin(ypred1[:]*per1000_factor))
            )
    
    

    locmaj = mpl.ticker.LogLocator(base=10.0,numticks=5) 
    locmin = mpl.ticker.LogLocator(base=10.0,numticks=40, subs=np.arange(0.2,1.0,0.1)) 
    
    
#    ax1.yaxis.set_major_formatter(engfmt_ax)
#    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax1.yaxis.set_major_locator(locmaj)
    ax1.yaxis.set_minor_locator(locmin)
#    ax1.yaxis.set_ticks_position('both')
    
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(5,prune=None))
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    
#    ax2.yaxis.set_major_formatter(engfmt_ax)
#    ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax2.yaxis.set_ticks_position('both')
    
#    ax2.xaxis.set_major_locator(mticker.MaxNLocator(6,prune=None))
#    ax2.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    ax2.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0,numticks=5))
    ax2.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0,numticks=40, subs=np.arange(2, 10) * .1) )
    
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(6,prune=None))
    ax2.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    
    
    locmaj2 = mpl.ticker.LogLocator(base=10.0,numticks=5,subs=(1.0,)) 
    locmin2 = mpl.ticker.LogLocator(base=10.0,numticks=40, subs=np.arange(2, 10) * .1) 
    
    if np.amax(np.log10(yprojected[:,2])) - np.amin(np.log10(yprojected[:,2])) > 2:
        ax3.set_ylim(bottom=max(1E-4,np.amin(yprojected[:,2]*per1000_factor*death_rate)),
                     top=np.amax(yprojected[:,2])*per1000_factor*5)
        ax3.set_yscale('log')
        ax3.yaxis.set_major_locator(locmaj2)
        ax3.yaxis.set_minor_locator(locmin2)
    else:
        ax3.set_ylim(bottom=max(-1E-4,np.amin(yprojected[:,2]*per1000_factor*death_rate)*0.75),
                     top=np.amax(yprojected[:,2])*per1000_factor*1.5)
        ax3.yaxis.set_major_locator(mticker.MaxNLocator(5,prune=None))
        ax3.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
#        ax3.yaxis.set_major_formatter(xfmt)
    
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(6,prune=None))
    ax3.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    
#    ax3.yaxis.set_major_formatter(engfmt_ax)
#    ax3.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax3.yaxis.set_ticks_position('both')
    
    
    locmaj3 = mpl.ticker.LogLocator(base=10.0,numticks=6,subs=(1.0,)) 
    locmin3 = mpl.ticker.LogLocator(base=10.0,numticks=48, subs=np.arange(2, 10) * .1) 
    
    if np.amax(np.log10(mild_infections)) - np.amin(np.log10(mild_infections)) > 2:
        ax4.set_ylim(bottom=max(0.001,np.amin(critical_infections)),top=np.amax(mild_infections)*5)
        ax4.set_yscale('log')
        ax4.yaxis.set_major_locator(locmaj3)
        ax4.yaxis.set_minor_locator(locmin3)
    else:
        ax4.set_ylim(bottom=max(-0.01,np.amin(critical_infections)*0.5),top=np.amax(mild_infections)*1.5)
        ax4.yaxis.set_major_locator(mticker.MaxNLocator(5,prune=None))
        ax4.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
#        ax4.yaxis.set_major_formatter(xfmt)
    
    
    ax4.xaxis.set_major_locator(mticker.MaxNLocator(6,prune=None))
    ax4.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
#    ax4.yaxis.set_major_formatter(engfmt_ax)
    ax4.yaxis.set_ticks_position('both')
#    ax4.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    
    day_locator = mpl.dates.DayLocator(interval=7)
    month_locator = mpl.dates.MonthLocator(interval=1)
    
    
    ax1_locs = ax1.get_xticks()  
    
    
    ax1_d_lim = (
                t0 + datetime.timedelta(days = ax1.get_xlim()[0]),
                t0 + datetime.timedelta(days = ax1.get_xlim()[1])
            )
    
    ax1_d = ax1.twiny()
    ax1_d.xaxis.tick_bottom()
    ax1_d.xaxis.set_major_formatter(datefmt)
    ax1.xaxis.tick_top()
    ax1_d.set_xlim(ax1_d_lim)
    ax1_d.tick_params(rotation=90, labelsize='11')
    
    # Date axis2
    ax2_locs = ax2.get_xticks()  
    
    
    ax2_d_lim = (
                t0_fit + datetime.timedelta(days = ax2.get_xlim()[0]),
                t0_fit + datetime.timedelta(days = ax2.get_xlim()[1])
            )
    
    ax2_d = ax2.twiny()
    ax2_d.xaxis.tick_bottom()
    ax2.xaxis.tick_top()
    ax2_d.set_xlim(ax2_d_lim)
    ax2_d.xaxis.set_major_formatter(datefmt)
    ax2_d.xaxis.set_major_locator(day_locator)
    ax2_d.tick_params(rotation=90, labelsize='11')
    
    # Date axis3
    ax3_locs = ax3.get_xticks()  
    
    
    ax3_d_lim = (
                t0_fit + datetime.timedelta(days = ax3.get_xlim()[0]),
                t0_fit + datetime.timedelta(days = ax3.get_xlim()[1])
            )
    
    ax3_d = ax3.twiny()
    ax3_d.xaxis.tick_bottom()
    ax3.xaxis.tick_top()
    ax3_d.set_xlim(ax3_d_lim)
    ax3_d.xaxis.set_major_formatter(datefmt)
    ax3_d.xaxis.set_major_locator(mpl.dates.DayLocator(interval=30))
    ax3_d.tick_params(rotation=90, labelsize='11')
    
    
    # Date axis3
    ax4_locs = ax4.get_xticks()  
    
    
    ax4_d_lim = (
                t0_fit + datetime.timedelta(days = ax4.get_xlim()[0]),
                t0_fit + datetime.timedelta(days = ax4.get_xlim()[1])
            )
    
    ax4_d = ax4.twiny()
    ax4_d.xaxis.tick_bottom()
    ax4.xaxis.tick_top()
    ax4_d.set_xlim(ax4_d_lim)
    ax4_d.xaxis.set_major_formatter(datefmt)
    ax4_d.xaxis.set_major_locator(mpl.dates.DayLocator(interval=30))
    ax4_d.tick_params(rotation=90, labelsize='11')
    
    ax1.yaxis.tick_left()
    ax1_k_lim = (
                ax1.get_ylim()[0]*1E3/population,
                ax1.get_ylim()[1]*1E3/population
            )
    ax1_k = ax1.twinx()
    ax1_k.set_yscale('log')
    ax1_k.yaxis.tick_right()
    ax1_k.set_ylim(ax1_k_lim)
#    ax1_k.set_ylabel('per 1000 people')
    
    ax1_k.yaxis.set_major_locator( mpl.ticker.LogLocator(base=10.0,numticks=5,subs=(1.0,)) )
    ax1_k.yaxis.set_minor_locator( mpl.ticker.LogLocator(base=10.0,numticks=40, subs=np.arange(2, 10) * .1) )
#    ax1_k.yaxis.set_major_formatter(engfmt_ax)
    
    
    ax4_d_lim = (
                t0_fit + datetime.timedelta(days = ax4.get_xlim()[0]),
                t0_fit + datetime.timedelta(days = ax4.get_xlim()[1])
            )
    
#    ax4_d = ax4.twiny()
#    ax4_d.xaxis.tick_top()
#    ax4_d.set_xlim(ax4_d_lim)
#    ax4_d.xaxis.set_major_formatter(datefmt)
#    ax4_d.xaxis.set_major_locator(mpl.dates.DayLocator(interval=30))
#    ax4_d.tick_params(rotation=90, labelsize='11')
    
#    ax3_d = ax3.twiny()
#    ax3_d.xaxis.tick_top()
#    ax3_d.set_xlim(np.amin(dates_proj),np.amax(dates_proj))
#    ax3_d.xaxis.set_major_formatter(datefmt_yr)
#    ax3_d.xaxis.set_major_locator(month_locator)
##    ax3_d.xaxis.set_major_locator(mticker.MultipleLocator(2))
#    ax3_d.tick_params(rotation=90, labelsize='11')
    
#    ax3_k_lim = (
#                ax3.get_ylim()[0]*1E3/population,
#                ax3.get_ylim()[1]*1E3/population
#            )
#    ax3_k = ax3.twinx()
#    ax3_k.set_yscale('log')
#    ax3_k.yaxis.tick_right()
#    ax3_k.set_ylim(ax3_k_lim)
#    ax3_k.set_ylabel('per 1000 people')
#    
    
    
    
    
#    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.tight_layout()
    plt.show()
    
    filetag = 'covid19_model_{0}_ir'.format(location)
    filetag_pop = '{0}_population'.format(location)
    if before_day > 0:
        filetag += '_before_{0}d'.format(before_day)
   
    print('-------------------------------------------------------------------')
    print('beta1\t=\t{0:.3E} 1/days,\t95% CI: [{1:.3E}, {2:.3E}] 1/days'.format(
        popt1[0], ci1[0][0], ci1[0][1]))
    print('gamma1\t=\t{0:.3E} 1/days,\t95% CI: [{1:.3E}, {2:.3E}] 1/days'.format(
        popt1[1], ci1[1][0], ci1[1][1]))
#    print('mu1 =\t{0:.3E} 1/days, 95% CI: [{1:.3E}, {2:.3E}] 1/days'.format(
#        popt1[2], ci1[2][0], ci1[2][1]))
    print('E0\t\t=\t{0:.3E} people,\t95% CI: [{1:.3E}, {2:.3E}] people'.format(
        popt1[2], ci1[2][0], ci1[2][1]))
    print('rsquared = {0:.5f},\tchisquare = {1:.3E}, p = {2:.5f}'.format(
        rsquared1, chisq1, p1))
    print('-------------------------------------------------------------------')
    print('beta2\t=\t{0:.3E} 1/days,\t95% CI: [{1:.3E}, {2:.3E}] 1/days'.format(
        popt2[0], ci2[0][0], ci2[0][1]))
    print('gamma2\t=\t{0:.3E} 1/days,\t95% CI: [{1:.3E}, {2:.3E}] 1/days'.format(
        popt2[1], ci2[1][0], ci2[1][1]))
#    print('mu2 =\t{0:.3E} 1/days, 95% CI: [{1:.3E}, {2:.3E}] 1/days'.format(
#        popt2[2], ci2[2][0], ci2[2][1]))
    print('E0\t\t=\t{0:.3E} people,\t95% CI: [{1:.3E}, {2:.3E}] people'.format(
        popt2[2], ci2[2][0], ci2[2][1]))
    
    print('rsquared = {0:.5f}, chisquare = {1:.3E}, p = {2:.5f}'.format(
        rsquared2, chisq2, p2))
    print('-------------------------------------------------------------------')
    if inflection_point3 > inflection_point2:
        print('beta2\t=\t{0:.3E} 1/days,\t95% CI: [{1:.3E}, {2:.3E}] 1/days'.format(
                popt3[0], ci3[0][0], ci3[0][1]))
        print('gamma3\t=\t{0:.3E} 1/days,\t95% CI: [{1:.3E}, {2:.3E}] 1/days'.format(
                popt3[1], ci3[1][0], ci3[1][1]))
        print('E0\t\t=\t{0:.3E} people,\t95% CI: [{1:.3E}, {2:.3E}] people'.format(
                popt3[2], ci3[2][0], ci3[2][1]))
        print('rsquared = {0:.5f}, chisquare = {1:.3E}, p = {2:.5f}'.format(
                rsquared3, chisq3, p3))
        print('-------------------------------------------------------------------')
    
    
    if save_results:
        fig.savefig(os.path.join(results_folder,filetag+'.png'), dpi=600)
#        fig_pop.savefig(os.path.join(results_folder,filetag_pop+'.png'), dpi=600)
        results_path = os.path.join(results_folder, csv_results)
        fitting_results = np.array([(
                    location,
                    popt1[0], ci1[0][0], ci1[0][1],
                    popt1[1], ci1[1][0], ci1[1][1],
                    popt1[2], ci1[2][0], ci1[2][1],
                    infected1[0], removed1[0], R01_cal, n_days1,
                    popt2[0], ci2[0][0], ci2[0][1],
                    popt2[1], ci2[1][0], ci2[1][1],
                    popt2[2], ci2[2][0], ci2[2][1],
                    infected2[0], removed2[0], R02_cal, n_days2,
                    popt3[0], ci3[0][0], ci3[0][1],
                    popt3[1], ci3[1][0], ci3[1][1],
                    popt3[2], ci3[2][0], ci3[2][1],
                    infected3[0], removed3[0], R03_cal, n_days3,
                    latency_time,
                    start_idx, 
                    inflection_point1, inflection_point2, 
                    inflection_point3, inflection_point4,
                    before_day,
                    np.datetime64(datetime.datetime.now(), 'ns'),
                    b0[0], b0[1], b0[2], rsquared1, chisq1, p1,
                    b1[0], b1[1], b1[2], rsquared2, chisq2, p2,
                    rsquared3, chisq3, p3,
                    recovery_rate, death_rate,
                    intervened_peak_time[0],
                    intervened_max_infections[0],
                    intervened_peak_time[1],
                    intervened_max_infections[1],
                    intervened_peak_time[2],
                    intervened_max_infections[2],
                    intervened_peak_time[3],
                    intervened_max_infections[3]
                )], dtype=seir_fit_dtype)
        new_results_df = pd.DataFrame(data=fitting_results)
        if not os.path.exists(results_path):
            new_results_df.to_csv(path_or_buf=results_path, index=False, 
                                  date_format='%Y-%m-%d %H:%M:%S')
        else:
            results_df = pd.read_csv(results_path, index_col=False)
            results_df = results_df.append(new_results_df, ignore_index=True)
            results_df.to_csv(path_or_buf=results_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
        filetag = location + datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Prepare the fitted and projected curves
        fitted_curves1 = np.empty(len(time1), dtype=seir_curvfit_type)
        for i, x, y, l, u in zip(range(len(time1)), time1, ypred1, lpb1, upb1):
            fitted_curves1[i] = (x, y[0], l[0], u[0], y[1], l[1], u[1])
            
        fitted_curves2 = np.empty(len(time2), dtype=seir_curvfit_type)
        for i, x, y, l, u in zip(range(len(time2)), time2, ypred2, lpb2, upb2):
            fitted_curves2[i] = (x, y[0], l[0], u[0], y[1], l[1], u[1])
            
        projected_curves = np.empty(len(time_projected), dtype=seir_projection_type)
        for i, x, y in zip(range(len(time_projected)), time_projected, yprojected):
            projected_curves[i] = (x, y[0], y[1], y[2], y[3])
        
        h5File = os.path.join(results_folder, filetag + '.h5')
        with h5py.File(h5File, 'w') as hf:
            ds_fc = hf.create_dataset('fitted_curve1', (len(time1),), dtype=seir_curvfit_type)
            ds_fc[...] = fitted_curves1
            ds_fc.attrs['beta_fit'] = popt1[0]
            ds_fc.attrs['beta_ci_l'] = ci1[0][0]
            ds_fc.attrs['beta_ci_u'] = ci1[0][1]
            ds_fc.attrs['gamma_fit'] = popt1[1]
            ds_fc.attrs['gamma_ci_l'] = ci1[1][0]
            ds_fc.attrs['gamma_ci_u'] = ci1[1][1]
            ds_fc.attrs['E0_fit'] = popt1[2]
            ds_fc.attrs['E0_ci_l'] = ci1[2][0]
            ds_fc.attrs['E0_ci_u'] = ci1[2][1]
            ds_fc.attrs['R0'] = removed1[0]
            ds_fc.attrs['I0'] = infected1[0]
            ds_fc.attrs['R01_cal'] = R01_cal
            ds_fc.attrs['latency_time'] = latency_time
            ds_fc.attrs['start_idx'] = start_idx
            ds_fc.attrs['before_day'] = before_day
            ds_fc.attrs['inflection_point'] = inflection_point1
            ds_fc.attrs['beta_guess'] = b0[0]
            ds_fc.attrs['gamma_guess'] = b0[1]
            ds_fc.attrs['E0_guess'] = b0[2]
            ds_fc.attrs['death_rate'] = death_rate
            ds_fc.attrs['recovery_rate'] = recovery_rate
            ds_fc.attrs['rsquared'] = rsquared1
            ds_fc.attrs['chisquare'] = chisq1
            ds_fc.attrs['p'] = p1
            
            ds_fc = hf.create_dataset('fitted_curve2', (len(time2),), dtype=seir_curvfit_type)
            ds_fc[...] = fitted_curves2
            ds_fc.attrs['beta_fit'] = popt2[0]
            ds_fc.attrs['beta_ci_l'] = ci2[0][0]
            ds_fc.attrs['beta_ci_u'] = ci2[0][1]
            ds_fc.attrs['gamma_fit'] = popt2[1]
            ds_fc.attrs['gamma_ci_l'] = ci2[1][0]
            ds_fc.attrs['gamma_ci_u'] = ci2[1][1]
            ds_fc.attrs['E0_fit'] = popt2[2]
            ds_fc.attrs['E0_ci_l'] = ci2[2][0]
            ds_fc.attrs['E0_ci_u'] = ci2[2][1]
            ds_fc.attrs['R0'] = removed2[0]
            ds_fc.attrs['I0'] = infected2[0]
            ds_fc.attrs['latency_time'] = latency_time
            ds_fc.attrs['start_idx'] = start_idx
            ds_fc.attrs['before_day'] = before_day
            ds_fc.attrs['inflection_point'] = inflection_point2
            ds_fc.attrs['beta_guess'] = b1[0]
            ds_fc.attrs['gamma_guess'] = b1[1]
            ds_fc.attrs['E0_guess'] = b1[2]
            ds_fc.attrs['death_rate'] = death_rate
            ds_fc.attrs['recovery_rate'] = recovery_rate
            ds_fc.attrs['rsquared'] = rsquared2
            ds_fc.attrs['chisquare'] = chisq2
            ds_fc.attrs['p'] = p2
            
            if inflection_point3 > inflection_point2:
                fitted_curves3 = np.empty(len(time3), dtype=seir_curvfit_type)
                for i, x, y, l, u in zip(range(len(time3)), time3, ypred3, lpb3, upb3):
                    fitted_curves3[i] = (x, y[0], l[0], u[0], y[1], l[1], u[1])
                
                ds_fc = hf.create_dataset('fitted_curve3', (len(time3),), dtype=seir_curvfit_type)
                ds_fc[...] = fitted_curves3
                ds_fc.attrs['beta_fit'] = popt3[0]
                ds_fc.attrs['beta_ci_l'] = ci3[0][0]
                ds_fc.attrs['beta_ci_u'] = ci3[0][1]
                ds_fc.attrs['R0'] = removed3[0]
                ds_fc.attrs['I0'] = infected3[0]
                ds_fc.attrs['latency_time'] = latency_time
                ds_fc.attrs['start_idx'] = start_idx
                ds_fc.attrs['before_day'] = before_day
                ds_fc.attrs['inflection_point'] = inflection_point3
                ds_fc.attrs['beta_guess'] = b1[0]
                ds_fc.attrs['death_rate'] = death_rate
                ds_fc.attrs['recovery_rate'] = recovery_rate
                ds_fc.attrs['rsquared'] = rsquared3
                ds_fc.attrs['chisquare'] = chisq3
                ds_fc.attrs['p'] = p3
                    
            ds_pc = hf.create_dataset('projection', (len(time_projected),), dtype=seir_projection_type)
            ds_pc[...] = projected_curves
            
    
