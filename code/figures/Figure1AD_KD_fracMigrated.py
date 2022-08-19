import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import numpy as np
import glob
import pandas as pd
import math

import FlowCytometryTools
from FlowCytometryTools import FCMeasurement

import skimage
import skimage.io
from skimage.morphology import disk
from skimage.filters import rank

import seaborn as sns

# plt.style.use('styleNB')
plt.style.use('styleNB.mplstyle')

colors = ['#738FC1', '#7AA974', '#D56C55', '#EAC264', '#AB85AC', '#C9D7EE', '#E8B19D', '#DCECCB', '#D4C2D9']

fig = plt.figure(figsize=(3*0.75*1.14,6*0.75))
gs = GridSpec(nrows=2, ncols=1)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

#############################################
#  Figure 1: Summary of screen and overview of data
# (A-D) Experimental schematic;  plot to show knockdown in HL-60 cells; bulk migration results - fraction of cells migrated
# (E) Volcano plots to show the performance of the screens - showing the results/ statistical
# significance  of each of the proliferation, differentiation, and combined cell migration data (in a different .py file)
# (F) Upset plot to show overlap of identified genes across the screens (in a different .py file)
#############################################

#############################################
# part A - functionality of knockdown line
#############################################

filelist = \
['../../data/flow_cytometry/20200309_CD4_knockdown/KW-SC575_antiCD4.fcs',
'../../data/flow_cytometry/20200309_CD4_knockdown/KW-SC575_gCD4_antiCD4.fcs',
'../../data/flow_cytometry/20200309_CD4_knockdown/KW-SC575_gCD4_antiCTRL.fcs',
]

filelist_dict = dict(zip(filelist,
                        [ 'KW-SC575',  'KW-SC575',  'KW-SC575',
                        ]))

color_dict = dict(zip(['KW-SC575_antiCD4', 'KW-SC575_gCD4_antiCD4', 'KW-SC575_gCD4_antiCTRL'],
                    ['#E28027', '#078ED1', 'grey']))

label_dict = dict(zip(['KW-SC575_antiCD4', 'KW-SC575_gCD4_antiCD4', 'KW-SC575_gCD4_antiCTRL',
                      ],
                      ['dCas9-KRAB only', 'dCas9-KRAB +\nCD4 sgRNA', 'isotype control'
                      ]))

for file in filelist:
    color_i = file.split(filelist_dict[file])[-1].split('.fcs')[0]
    sample = FCMeasurement(ID='HL60', datafile=file)
    sample = sample[sample['FSC-A'] >= 2.5E5]
    sample = sample[sample['SSC-A'] >= 2E5]
    sample['APC-Cy7-A_l'] = np.log10(sample['APC-Cy7-A'][sample['APC-Cy7-A'] > 0])

    if 'antiCTRL' in file:
        sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax1, color = 'grey',
                lw = 0, alpha = 0.3, legend = False, shade = True, zorder = 10)
    else:
        g1 = sns.kdeplot(sample["APC-Cy7-A_l"], ax = ax1,
        color = color_dict[file.split('/')[-1][:-4]],
        lw = 1.5, ls = '-', alpha = 1, legend = False, zorder = 0)

    g1.set(xlabel=None, ylabel=None)


ax1.set_xticks(np.log10([200,300,400,500,600,700,800,900,
                        2000,3000,4000,5000,6000,7000,8000,9000,
                        20000,30000,40000,50000,60000,70000,80000,90000,
                        200000,300000]), minor = True)
ax1.set_xticks(np.log10([100,1000, 10000,100000]))
ax1.set_xlim(2,4.5)
ax1.set_xticklabels([])
ax1.set_yticks([])
ax1.set_xlabel('CD4 expression\n[APC-Cy7,a.u.]', fontsize=11)
ax1.set_ylabel('frequency', fontsize=11)
ax1.tick_params(width=0.6)

for axis in ['bottom','left']:
    ax1.spines[axis].set_linewidth(0.6)

# increase major tick length
ax1.tick_params(length=8)
##################################

df_mig = pd.read_csv('../../data/migration_assays/migration_assays_summary.csv')

df_temp = df_mig[df_mig.format == 'ECM']
ax2.bar(7, df_temp.frac_migrated.mean(),
        yerr = df_temp.frac_migrated.std(),  color = '#EAC264')

df_temp = df_mig[df_mig.FBS_grad == False]
df_temp = df_temp[df_temp.fMLP_nM == 0]
df_temp = df_temp[df_temp.time_point_hr <= 2]
df_temp = df_temp[df_temp.time_point_hr > 1.0]
ax2.bar(1, df_temp.frac_migrated.mean(),
        yerr = df_temp.frac_migrated.std(), color = '#C1D1BB')


df_temp = df_mig[df_mig.FBS_grad == False]
df_temp = df_temp[df_temp.fMLP_nM == 0]
df_temp = df_temp[df_temp.time_point_hr > 2]
ax2.bar(2, df_temp.frac_migrated.mean(),
        yerr = df_temp.frac_migrated.std(), color = '#C1D1BB')

df_temp = df_mig[df_mig['FBS_%'] == 10]
df_temp = df_temp[df_temp.FBS_grad == True]
df_temp = df_temp[df_temp.time_point_hr == 1.5]
df_temp = df_temp[df_temp.date >= 20201120]

ax2.bar(4, df_temp.frac_migrated.mean(),
        yerr = df_temp.frac_migrated.std(), color = '#85A879')

df_temp = df_mig[df_mig['FBS_%'] == 10]
df_temp = df_temp[df_temp.FBS_grad == True]
df_temp = df_temp[df_temp.time_point_hr == 6]
df_temp = df_temp[df_temp.date >= 20201120]

ax2.bar(5, df_temp.frac_migrated.mean(),
        yerr = df_temp.frac_migrated.std(), color = '#85A879')


ax2.set_xlabel('time [hour]', fontsize=11)
ax2.set_ylabel('fraction of cells that\nsuccessfully migrate', fontsize=11)
ax2.set_yticks([0,0.2,0.4,0.6])
ax2.set_xticks([])
ax2.set_xticklabels([])
ax2.tick_params(width=0.6)

for axis in ['bottom','left']:
    ax2.spines[axis].set_linewidth(0.6)


####################################


plt.tight_layout()
fig.savefig('../../figures/Fig1AD_KD_fractionMigration.pdf')
