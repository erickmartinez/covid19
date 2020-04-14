# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 06:48:20 2020

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
import datetime


data_folder = r'./SEIR_v3_batch_20200412'
results_folder = 'fit_analysis'
csv_results = 'fitting_results_seir.csv'
geodata =  r'./countries_coordinates.csv'

t0 = datetime.date(year=2020, month=4, day=12)

exclusions = ['global', 'Chile', 'India', 'Netherlands', 'Ireland']

filetag = 'fit_analysis_v0'
data_color = 'C0'

xfmt = ScalarFormatter(useMathText=True)
xfmt.set_powerlimits((-3,3))
engfmt = EngFormatter(places=1, sep=u"\N{THIN SPACE}")  # U+2009
datefmt = mpl.dates.DateFormatter("%m/%d")
datefmt_yr = mpl.dates.DateFormatter("%Y/%m/%d")
engfmt_ax = EngFormatter(places=0, sep=u"\N{THIN SPACE}")  # U+2009

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
                    'axes.labelsize' : 12,
                    'axes.labelweight' : 'regular',
                    'legend.handletextpad' : 0.2,
                    'legend.borderaxespad' : 0.2,
                    'axes.linewidth': 1.25,
                    'axes.titlesize' : 14,
                    'axes.titleweight' : 'bold',
                    'axes.titlepad' : 6,
                    'figure.titleweight' : 'bold',
                    'figure.dpi': 100}


if __name__ == '__main__':
    
    results_path = os.path.join(data_folder, results_folder)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        results_path = os.path.abspath(results_path)
    
    # read resutls csv
    df_fit = pd.read_csv(os.path.join(data_folder, csv_results))
    
    full_country_names = []
    for i, r in df_fit.iterrows():
        location = r['Country/Region']
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
        full_country_names.append(country_name)
        
    
    df_fit['Country'] = full_country_names
        
    df_country_names = pd.DataFrame(data=full_country_names, columns=['Country'])
    
    df_country_codes = pd.read_csv('./UN_country_Codes.csv', usecols=['Country or Area', 'M49 Code','ISO-alpha3 Code']).rename(columns={'Country or Area': 'Country', 'M49 Code': 'M49','ISO-alpha3 Code': 'ISO3-Code'})
    df_country_codes = df_country_codes[df_country_codes['M49'].str.isnumeric()].astype({'M49': 'int16'})
    
    df_country_codes = pd.merge(df_country_names, df_country_codes, on='Country', how='inner')
    coords_df = pd.read_csv(geodata)
    coords_df = pd.merge(coords_df, df_fit[['Country/Region', 'Country']], on='Country/Region', how='inner')
    df_country_geo = pd.merge(df_country_codes, coords_df, on='Country', how='inner')
    fit_df = pd.merge(df_country_geo, df_fit, on='Country', how='left').rename(columns={'Country/Region_x': 'Country/Region'})
    
    for e in exclusions:
        fit_df = fit_df[fit_df['Country'] != e]
    fit_df = fit_df.reset_index(drop=True)
#    fit_df = fit_df[fit_df['Country/Region'] != 'global'].reset_index()
    fit_df = fit_df.rename(columns={'wb_beds per1000': 'hospital_beds'})
    fit_df['beta_latest'] = np.where(fit_df['inflection_point3'] > 0, fit_df['beta3'], fit_df['beta2'])
    fit_df['beta_latest_ci_l'] = np.where(fit_df['inflection_point3'] > 0, fit_df['beta3_ci_l'], fit_df['beta2_ci_l'])
    fit_df['beta_latest_ci_u'] = np.where(fit_df['inflection_point3'] > 0, fit_df['beta3_ci_u'], fit_df['beta2_ci_u'])
    fit_df['gamma_latest'] = np.where(fit_df['inflection_point3'] > 0, fit_df['gamma3'], fit_df['beta2'])
    fit_df['gamma_latest_ci_l'] = np.where(fit_df['inflection_point3'] > 0, fit_df['gamma3_ci_l'], fit_df['gamma2_ci_l'])
    fit_df['gamma_latest_ci_u'] = np.where(fit_df['inflection_point3'] > 0, fit_df['gamma3_ci_u'], fit_df['gamma2_ci_u'])
    fit_df = fit_df.eval('beta_change = beta3 - beta2')
    fit_df = fit_df.eval('beta_change_percent = beta_change/beta2')
    fit_df = fit_df.eval('beta_ratio1 = beta2/beta1')
    fit_df = fit_df.eval('beta_ratio2 = beta3/beta2')
    fit_df = fit_df.eval('confirmed_per100k = 1E5*confirmed/population')
    fit_df = fit_df.eval('recovered_per100k = 1E5*recovered/population')
    fit_df = fit_df.eval('dead_per100k = 1000*dead/population')
    fit_df = fit_df.eval('peak_value_current_beta_per100k = 1E5*peak_value_current_beta/population')
    fit_df = fit_df.eval('peak_value_b90_per100k = 1E5*peak_value_b90/population')
    fit_df = fit_df.eval('peak_value_b50_per100k = 1E5*peak_value_b50/population')
    fit_df = fit_df.eval('peak_value_b10_per100k = 1E5*peak_value_b10/population')
    fit_df = fit_df.eval('health_care_stress_current = 0.2*peak_value_current_beta_per100k/hospital_beds')
    fit_df = fit_df.eval('health_care_stress_b90 = 0.2*peak_value_b90_per100k/hospital_beds')
    fit_df = fit_df.eval('health_care_stress_b50 = 0.2*peak_value_b50_per100k/hospital_beds')
    fit_df = fit_df.eval('health_care_stress_b10 = 0.2*peak_value_b10_per100k/hospital_beds')
    
    
    fit_df = fit_df.sort_values(by=['confirmed_per100k'], ascending=False)
    fit_df = fit_df.reset_index(drop=True)
    
    confirmed = fit_df['confirmed']
    infections_peak = np.array([
            fit_df['peak_value_current_beta_per100k'],
            fit_df['peak_value_b90_per100k'],
            fit_df['peak_value_b50_per100k'],
            fit_df['peak_value_b10_per100k'],
    ]).T
    peak_datetime = np.array([
            fit_df['peak_date_current_beta'],
            fit_df['peak_date_b90'],
            fit_df['peak_date_b50'],
            fit_df['peak_date_b10'],
    ], dtype=np.datetime64).T
    
    hospital_occupancy = np.array(
                fit_df[['health_care_stress_current',
                'health_care_stress_b90',
                'health_care_stress_b50',
                'health_care_stress_b10']]
            )
    
    peak_delta_date = (peak_datetime - np.datetime64(t0, 'ns')).astype('timedelta64[D]').astype(np.int)
    
    
    cmap_blues = mpl.cm.get_cmap('Blues')
    cmap_purples = mpl.cm.get_cmap('Purples')
    cmap_greens = mpl.cm.get_cmap('Greens')
    normalize = mpl.colors.Normalize(vmin=0, vmax=3)
    beta_colors = [cmap_blues(normalize(t)) for t in range(1,4)]
    gamma_colors = [cmap_purples(normalize(t)) for t in range(1,4)]
    qs_colors = [cmap_greens(normalize(t)) for t in range(1,4)]
    
    # Plot beta3 only if available
    df_beta3 = fit_df[fit_df['inflection_point4'] != 0]
#    df_beta3 = df_beta3.reset_index(drop=True)
    
    x12 = list(fit_df.index)
    x3 = list(df_beta3.index)
    
    x_labels_bottom = list(fit_df['Country/Region'])
    x_labels_top = [engfmt.format_eng(x) for x in list(fit_df['confirmed'])]
    

    
    mpl.rcParams.update(defaultPlotStyle)
    
    fig = plt.figure()
    fig.set_size_inches(6.5,8.5,forward=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.4)
    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1])
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=4, ncols=1, 
                                            subplot_spec = gs0[0])
    
    
    ax1 = fig.add_subplot(gs00[0,0])
    ax2 = fig.add_subplot(gs00[1,0])
    ax3 = fig.add_subplot(gs00[2,0])
    ax4 = fig.add_subplot(gs00[3,0])

    
    ax1.bar(x12, fit_df['confirmed_per100k'], color='tab:orange', label='Confirmed Cases')
    ax1.bar(x12, fit_df['recovered_per100k'], color='tab:green', label='Recoveries')
    ax1.bar(x12, fit_df['dead_per100k'], color='tab:red', label='Deaths')
    
    bar_width = 0.25
    
#    ax1_lr = ax1.twinx()
#    ax1_lr.plot(x12, fit_df['death_rate']*100, color='tab:red')
    
    
    ax2.bar(x12,fit_df['beta1'], width=bar_width,
                color=beta_colors[0], label='$\\beta_1$',
                zorder=1)
    
    ax2.bar([x +bar_width for x in x12],fit_df['beta2'],width=bar_width,
                color=beta_colors[1], label='$\\beta_2$', 
                zorder=2)
   
    ax2.bar([x + 2*bar_width for x in x3], df_beta3['beta3'],width=bar_width, 
                label='$\\beta_3$',
                color=beta_colors[2],
                zorder=3)
    
    

    ax3.bar(x12,fit_df['gamma1'], width=bar_width,
                color=gamma_colors[0], label='$\\gamma_1$',
                zorder=1)
    
    ax3.bar([x +bar_width for x in x12],fit_df['gamma2'],width=bar_width,
                color=gamma_colors[1], label='$\\gamma_2$', 
                zorder=2)
   
    ax3.bar([x + 2*bar_width for x in x3], df_beta3['gamma3'],width=bar_width, 
                label='$\\gamma_3$',
                color=gamma_colors[2],
                zorder=3)
    
    
    ax4.bar(x12,fit_df['beta_ratio1'], width=bar_width,
                color=qs_colors[0],
                label='$Q_1$',
                zorder=1)
    
    
    
    ax4.bar([x +bar_width for x in x3],df_beta3['beta_ratio2'], width=bar_width,
                color=qs_colors[1],
                label='$Q_2$',
                zorder=1)
    
    q1_over = fit_df['beta_ratio1'] - np.ones_like(fit_df['beta_ratio1'])
    q2_over = df_beta3['beta_ratio2'] - np.ones_like(df_beta3['beta_ratio2'])
    
    idx1 = q1_over > 0
    idx2 = q2_over > 0
    
#    ax4.bar(np.array(x12)[idx1],q1_over[idx1], width=bar_width, bottom=np.ones_like(q1_over)[idx1],
#                color='tab:red',
#                zorder=1)
#    
#    
#    
#    ax4.bar([x +bar_width for x in np.array(x3)[idx2]], q2_over[idx2], width=bar_width, bottom=np.ones_like(q2_over)[idx2],
#            color='tab:red',
#            zorder=1)
    
    
    ax4.axhline(y=1, ls='--', lw=1.0, color='k')
    
    
    
    
    ax1.set_xticks(x12)
    ax1.set_xticklabels(x_labels_bottom)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params(axis='x', rotation=90, labelsize='11')
    ax1.set_ylabel('Cases (/100 k)')
    
    ax1.set_ylim(top=fit_df['confirmed_per100k'].max()*1.25)
    ax1.legend(loc='upper right', frameon=False, ncol=3, fontsize=11)
    
    ax1.tick_params(zorder=10, labelbottom=False, bottom=True, top=True, right=True, which='both', labelright=True)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(5,prune=None))
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    locs_ax1 = ax1.get_xticks()
    
    
    ax2.set_ylabel('$\\beta$ (1/days)')
    ax2.set_xticks(x12)
    ax2.set_xticklabels(x12)
    ax2.set_xbound(ax1.get_xbound())
    ax2.tick_params(zorder=10, labelbottom=False, bottom=True, top=True, right=True, which='both', labelright=True)
    ax2.set_yscale('log')
    ax2.set_ylim(
            top=max(fit_df['beta1'].max(),fit_df['beta2'].max(),fit_df['beta3'].max())*100,
            bottom=max(fit_df['beta1'].min(),fit_df['beta2'].min(),1E-5)
    )
    
#    ax2.grid(True, axis='x', linestyle='--', linewidth=1.0, zorder=1)
    
      
    ax2.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0,numticks=5))
    ax2.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0,numticks=50, subs=np.arange(2, 10) * .1))
    ax2.legend(loc='upper right', frameon=False, ncol=3)
    
    
    
   

    ax3.set_ylabel('$\\gamma$ (1/days)')
    ax3.set_yscale('log')
    ax3.set_xticks(x12)
    ax3.set_xticklabels(x12)
    ax3.tick_params(zorder=10, labelbottom=False, bottom=True, top=True, right=True, which='both', labelright=True)
    ax3.set_ylim(
            top=max(fit_df['gamma1'].max(),fit_df['gamma2'].max(),fit_df['gamma3'].max())*10,
            bottom=min(fit_df['gamma1'].min(),fit_df['gamma2'].min(),1E-3)
    )
    ax3.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0,numticks=4,subs=(1.0,)) )
    ax3.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0,numticks=40, subs=np.arange(2, 10) * .1))
    ax3.legend(loc='upper right', frameon=False, ncol=3)
    
    # AX4
    ax4.set_ylabel('$Q_n=\\beta_{n+1}/\\beta_{n}$ ')
    ax4.set_yscale('log')
    ax4.tick_params(zorder=10, labelbottom=True, bottom=True, top=True, right=True, which='both', labelright=True)
    ax4.set_ylim(
            top=1E2,#max(fit_df['beta_ratio1'].max(),df_beta3['beta_ratio2'].max())*100,
            bottom=1E-4
    )
    ax4.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0,numticks=4,subs=(1.0,)) )
    ax4.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0,numticks=40, subs=np.arange(2, 10) * .1))
    ax4.legend(loc='upper right', frameon=False, ncol=3)
    
    
    ax4.set_xticks(x12)
    ax4.set_xticklabels(x_labels_bottom)
    ax4.tick_params(rotation=90, labelsize='11', axis='x')
    
    
    ax1.text(0.01, 0.95,
             '(a)',
             horizontalalignment='left',
             verticalalignment='top', 
             fontsize=13,
             transform=ax1.transAxes,
             color='k')
    
    ax2.text(0.01, 0.95,
             '(b)',
             horizontalalignment='left',
             verticalalignment='top', 
             transform=ax2.transAxes,
             fontsize=13,
             color='k')
    
    ax3.text(0.01, 0.95,
             '(c)',
             horizontalalignment='left',
             verticalalignment='top', 
             transform=ax3.transAxes,
             fontsize=13,
             color='k')
    
    
    ax4.text(0.01, 0.95,
             '(d)',
             horizontalalignment='left',
             verticalalignment='top', 
             transform=ax4.transAxes,
             fontsize=13,
             color='k')
    
   
#    ax4.fill_between(ax4.get_xlim(), [1,1], ax4.get_ylim()[1], color='r', alpha=0.5, zorder=0)
    
    plt.tight_layout()
    plt.show()
    
    fig.savefig(os.path.join(results_path,filetag+'.png'), dpi=600)
    fig.savefig(os.path.join(results_path,filetag+'.eps'), dpi=600, format='eps')
    
    fit_df.to_csv(path_or_buf=os.path.join(results_path,filetag+'.csv'), index=False)
    
    # Intervention
    
    
    fig_peaks = plt.figure()
    fig_peaks.set_size_inches(6.5,7.5,forward=True)
    fig_peaks.subplots_adjust(hspace=0.1, wspace=0.4)
    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_peaks, width_ratios=[1])
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1, 
                                            subplot_spec = gs0[0])
    
    
    cmap = mpl.cm.get_cmap('cool')
    normalize = mpl.colors.Normalize(vmin=0, vmax=3)
    beta_colors = [cmap(normalize(t)) for t in range(0,4)]
    
    ax1 = fig_peaks.add_subplot(gs00[0,0])
    ax2 = fig_peaks.add_subplot(gs00[1,0])
    ax3 = fig_peaks.add_subplot(gs00[2,0])
    
    bar_width2 = 0.2
    
    ax1.bar(x12,peak_delta_date[:,0],width=bar_width2, 
                color=beta_colors[0], label='$\\beta_{\\mathrm{latest}}$',
                zorder=1)
    ax1.bar([x+bar_width2 for x in x12],peak_delta_date[:,1],width=bar_width2, 
                color=beta_colors[1], label='$0.9\\beta_{\\mathrm{latest}}$',
                zorder=1)
    ax1.bar([x+2*bar_width2 for x in x12],peak_delta_date[:,2],width=bar_width2, 
                color=beta_colors[2], label='$0.5\\beta_{\\mathrm{latest}}$',
                zorder=1)
    ax1.bar([x+3*bar_width2 for x in x12],peak_delta_date[:,3],width=bar_width2, 
                color=beta_colors[3], label='$0.1\\beta_{\\mathrm{latest}}$',
                zorder=1)
    
    ax1.axhline(y=365, ls='--', color='k', lw=1.0)
    
    ax1.text(x12[-1]+1,355,
             'Simulation Limit: {0} '.format((t0+datetime.timedelta(days=365)).strftime('%Y/%m/%d')),
             horizontalalignment='right',
             verticalalignment='top', 
             fontsize=10,
             color='k')
    
    ax1.set_title('Projected Infections')
    
    
    ax2.bar(x12,infections_peak[:,0],width=bar_width2, 
                color=beta_colors[0], label='$\\beta_{\\mathrm{latest}}$',
                zorder=1)
    ax2.bar([x+bar_width2 for x in x12],infections_peak[:,1],width=bar_width2, 
                color=beta_colors[1], label='$0.9\\beta_{\\mathrm{latest}}$',
                zorder=1)
    ax2.bar([x+2*bar_width2 for x in x12],infections_peak[:,2],width=bar_width2, 
                color=beta_colors[2], label='$0.5\\beta_{\\mathrm{latest}}$',
                zorder=1)
    ax2.bar([x+3*bar_width2 for x in x12],infections_peak[:,3],width=bar_width2, 
                color=beta_colors[3], label='$0.1\\beta_{\\mathrm{latest}}$',
                zorder=1)
    
    ## Hospital beds
    ax3.bar(x12,hospital_occupancy[:,0],width=bar_width2, 
                color=beta_colors[0], label='$\\beta_{\\mathrm{latest}}$',
                zorder=1)
    ax3.bar([x+bar_width2 for x in x12],hospital_occupancy[:,1],width=bar_width2, 
                color=beta_colors[1], label='$0.9\\beta_{\\mathrm{latest}}$',
                zorder=1)
    ax3.bar([x+2*bar_width2 for x in x12],hospital_occupancy[:,2],width=bar_width2, 
                color=beta_colors[2], label='$0.5\\beta_{\\mathrm{latest}}$',
                zorder=1)
    ax3.bar([x+3*bar_width2 for x in x12],hospital_occupancy[:,3],width=bar_width2, 
                color=beta_colors[3], label='$0.1\\beta_{\\mathrm{latest}}$',
                zorder=1)

#    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#           ncol=4, mode="expand", borderaxespad=0.)

    
    
    
    ####
    ax1.set_xticks(x12)
    ax1.set_xticklabels(x_labels_bottom)
    ax1.xaxis.tick_top()
#    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params(axis='x', rotation=90, labelsize='11')
    ax1.set_ylabel('$\Delta t_{\mathrm{peak}}$ (days)')
    
    
#    ax1.set_yscale('log')
#    ax1.set_ylim(bottom=max(np.amin(peak_delta_date[:]),1), top=np.amax(peak_delta_date[:])*10)
    ax1.set_ylim(bottom=max(np.amin(peak_delta_date[:]),1), top=np.amax(peak_delta_date[:])*1.3)
    ax1.tick_params(zorder=10, labelbottom=False, bottom=True, top=True, right=True, which='both', labelright=True)
    ax1.yaxis.set_major_formatter(xfmt)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(5,prune=None))
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    
    ax1_ylim = ax1.get_xlim()
    ax1_d = ax1.twinx()
    locs = ax1.get_yticks()            # Get locations and labels
    lbls = ax1.get_yticklabels()
    
    ax1_d.set_yticks(locs)
#    ax1_d.set_yscale('log')
    ax1_d.set_ybound(ax1.get_ybound())
    ax1_d.set_yticklabels([(t0 + datetime.timedelta(days=x)).strftime('%b %Y') for x in locs])
    ax1_d.tick_params(axis='y', labelsize=9)
    
#    ax1_d_yticks = ax1_d.yaxis.get_major_ticks()
#    ax1_d_yticks[-1].label1.set_visible(False)
    
    ax1.legend(loc='upper center', frameon=False, ncol=4, fontsize=11)
    
    ax2.set_ylabel('$I_{\\mathrm{peak}}$ (/100 k)')
    ax2.set_xticks(x12)
    ax2.set_xticklabels(x12)
    ax2.set_xbound(ax1.get_xbound())
#    ax2.tick_params(zorder=10, labelbottom=True, bottom=True, top=False, right=True, which='both', labelright=True)
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1E-3, top=np.amax(infections_peak)*10)
     
    ax2.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0,numticks=5))
    ax2.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0,numticks=50, subs=np.arange(2, 10) * .1))
    ax2.tick_params(zorder=10, labelbottom=False, bottom=True, top=True, right=True, which='both', labelright=True)
#    ax2.legend(loc='upper center', frameon=False, ncol=4, fontsize=11)
#    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#           ncol=4, mode="expand", borderaxespad=0.)
    
   
    ax3.set_ylabel('Healthcare\nstress (%)')
    ax3.set_xticks(x12)
    ax3.set_yscale('log')  
    ax3.set_xticklabels(x_labels_bottom)
    ax3.tick_params(rotation=90, labelsize='11', axis='x')
    ax3.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0,numticks=5))
    ax3.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0,numticks=50, subs=np.arange(2, 10) * .1))
    ax3.axhline(y=100, ls='--', lw=1, color='tab:red')
    ax3.tick_params(zorder=10, right=True, which='y', labelright=True)
    ax3.text(x12[-3],100,
             '100 %',
             horizontalalignment='center',
             verticalalignment='bottom', 
             fontsize=11,
             fontweight='bold',
             color='tab:red')
    
    
    ax1.text(0.015, 0.95,
             '(a)',
             horizontalalignment='left',
             verticalalignment='top', 
             transform=ax1.transAxes,
             fontsize=13,
             zorder=10,
             color='k')
    
    ax2.text(0.015, 0.95,
             '(b)',
             horizontalalignment='left',
             verticalalignment='top', 
             transform=ax2.transAxes,
             fontsize=13,
             zorder=10,
             color='k')
    
    ax3.text(0.015, 0.95,
             '(c)',
             horizontalalignment='left',
             verticalalignment='top', 
             transform=ax3.transAxes,
             fontsize=13,
             zorder=10,
             color='k')
#    
    plt.tight_layout()
    plt.show()
    
    fig_peaks.savefig(os.path.join(results_path,filetag+'_intervention.png'), dpi=600)
