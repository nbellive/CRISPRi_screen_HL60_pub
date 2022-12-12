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

plt.style.use('styleNB.mplstyle')

def fdr(p_vals):

    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr

#############################################
#  Figure 2: Summary of screen results - focus on differentiation

#############################################
genes = ['RHEB',
         'MTOR', 'DEPTOR', 'RPTOR', 'AKT1S1', 'MLST8', # mTORC1
         'MAPKAP1', 'RICTOR', # mTORC2
         'RRAGA', 'RRAGB', 'RRAGC', 'RRAGD',
         'LAMTOR1', 'LAMTOR2', 'LAMTOR3', 'LAMTOR4', 'LAMTOR5',
         'FLCN', 'FNIP1', 'FNIP2',
         'SZT2', 'ITFG2', 'KPTN', # KICSTOR
         'NPRL2', 'NPRL3', 'DEPDC5', # GATOR1
         'MIOS', 'SEH1L', 'SEC13', 'WDR24', 'WDR59', # GATOR2
         'AKT1',
         'AKT2',
         'PDPK1',
         'PIK3CD',
         'TSC1', 'TSC2']


fig, ax = plt.subplots(figsize=(2*len(genes), 2))

#############################

df = pd.read_csv('../../data/screen_summary/stats/gene_avg/20220516_screen_log2fold_diffs_differentiation_means_pvalue.csv')
df = df[~df.gene.str.contains('CONTROL')]
df['fdr'] = fdr(df['pvalue'])


df = df[df.gene.isin(genes)]
# data = df.fdr.values.T
data = []
for g in genes:
    val = df[df.gene == g].fdr.values[0]
    val = (-np.log10(val))*np.sign(df[df.gene == g].log2fold_diff_mean.values[0])
    data = np.append(data, val)
    print(g, val)
data = np.expand_dims(data, axis=1)

fig, ax = plt.subplots()
im = ax.imshow(data, cmap='RdBu', vmin = -5, vmax=5, origin = 0)
cbar = fig.colorbar(im, ax=ax)#, extend='both')




#
ax.set_xlabel(None)
# ax.set_ylabel(genes)
ax.set_yticks(np.arange(len(genes)))
ax.set_yticklabels(genes)


plt.tight_layout()
fig.savefig('../../figures/Fig2C_mTORC1_diffScreen.pdf')
