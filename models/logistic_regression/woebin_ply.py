from pandas.api.types import is_string_dtype
import multiprocessing as mp
from .credit_scoring_utils.info_value import *


def woepoints_ply1(dtx, binx, x_i, woe_points):
    '''
    Transform original values into woe or porints for one variable.

    Params
    ------

    Returns
    ------

    '''
    # woe_points: "woe" "points"
    binx = pd.merge(
      pd.DataFrame(binx['bin'].str.split('%,%').tolist(), index=binx['bin'])\
        .stack().reset_index().drop('level_1', axis=1),
      binx[['bin', woe_points]],
      how='left', on='bin'
    ).rename(columns={0:'V1',woe_points:'V2'})

    # dtx
    ## cut numeric variable
    if is_numeric_dtype(dtx[x_i]):
        is_sv = pd.Series(not bool(re.search(r'\[', str(i))) for i in binx.V1)
        binx_sv = binx.loc[is_sv]
        binx_other = binx.loc[~is_sv]
        # create bin column
        breaks_binx_other = np.unique(list(map(float, ['-inf']+[re.match(r'.*\[(.*),.+\).*', str(i)).group(1) for i in binx_other['bin']]+['inf'])))
        labels = ['[{},{})'.format(breaks_binx_other[i], breaks_binx_other[i+1]) for i in range(len(breaks_binx_other)-1)]

        dtx = dtx.assign(xi_bin = lambda x: pd.cut(x[x_i], breaks_binx_other, right=False, labels=labels))\
          .assign(xi_bin = lambda x: [i if (i != i) else str(i) for i in x['xi_bin']])

        mask = dtx[x_i].isin(binx_sv['V1'])
        dtx.loc[mask,'xi_bin'] = dtx.loc[mask, x_i].astype(str)
        dtx = dtx[['xi_bin']].rename(columns={'xi_bin':x_i})
    ## to charcarter, na to missing
    if not is_string_dtype(dtx[x_i]):
        dtx.loc[:,x_i] = dtx.loc[:,x_i].astype(str).replace('nan', 'missing')

    dtx = dtx.fillna('missing').assign(rowid = dtx.index)
    # rename binx
    binx.columns = ['bin', x_i, '_'.join([x_i,woe_points])]
    # merge
    dtx_suffix = pd.merge(dtx, binx, how='left', on=x_i).sort_values('rowid')\
      .set_index(dtx.index)[['_'.join([x_i,woe_points])]]
    return dtx_suffix

    
def woebin_ply(dt, bins, no_cores=None, print_step=0):
    '''
    WOE Transformation
    ------
    `woebin_ply` converts original input data into woe values 
    based on the binning information generated from `woebin`.
    
    Params
    ------
    dt: A data frame.
    bins: Binning information generated from `woebin`.
    no_cores: Number of CPU cores for parallel computation. 
      Defaults NULL. If no_cores is NULL, the no_cores will 
      set as 1 if length of x variables less than 10, and will 
      set as the number of all CPU cores if the length of x 
      variables greater than or equal to 10.
    print_step: A non-negative integer. Default is 1. If 
      print_step>0, print variable names by each print_step-th 
      iteration. If print_step=0 or no_cores>1, no message is print.
    
    Returns
    -------
    DataFrame
        a dataframe of woe values for each variables 
    
    Examples
    -------
    import scorecardpy as sc
    import pandas as pd
    
    # load data
    
    # Example I
    dt = dat[["creditability", "credit.amount", "purpose"]]
    # binning for dt
    bins = woebin(dt, y = "creditability")
    
    # converting original value to woe
    dt_woe = woebin_ply(dt, bins=bins)
    
    # Example II
    # binning for germancredit dataset
    bins_germancredit = woebin(dat, y="creditability")
    
    # converting the values in germancredit to woe
    ## bins is a dict
    germancredit_woe = woebin_ply(dat, bins=bins_germancredit) 
    ## bins is a dataframe
    germancredit_woe = woebin_ply(dat, bins=pd.concat(bins_germancredit))
    '''
    # remove date/time col
    dt = rmcol_datetime_unique1(dt)
    # replace "" by NA
    dt = rep_blank_na(dt)

    print_step = check_print_step(print_step)
    
    # bins # if (is.list(bins)) rbindlist(bins)
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    # x variables
    xs_bin = bins['variable'].unique()
    xs_dt = list(dt.columns)
    xs = [el for el in xs_bin if el in xs_dt]
    # length of x variables
    xs_len = len(xs)
    # initial data set
    dat = dt.loc[:, [el for el in xs_dt if el not in xs]]
    
    # loop on xs
    if (no_cores is None) or (no_cores < 1):
        no_cores = 1 if xs_len<10 else mp.cpu_count()
    # 
    if no_cores == 1:
        for i in np.arange(xs_len):
            x_i = xs[i]
            if print_step>0 and bool((i+1) % print_step): 
                print(('{:'+str(len(str(xs_len)))+'.0f}/{} {}').format(i, xs_len, x_i), flush=True)
            #
            binx = bins[bins['variable'] == x_i].reset_index()

            dtx = dt[[x_i]]
            dat = pd.concat([dat, woepoints_ply1(dtx, binx, x_i, woe_points="woe")], axis=1)
    else:
        pool = mp.Pool(processes=no_cores)
        # arguments
        args = zip(
          [dt[[i]] for i in xs], 
          [bins[bins['variable'] == i] for i in xs], 
          [i for i in xs], 
          ["woe"]*xs_len
        )
        # bins in dictionary
        dat_suffix = pool.starmap(woepoints_ply1, args)
        dat = pd.concat([dat]+dat_suffix, axis=1)
        pool.close()
    return dat
