# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 19:49:53 2021

@author: chanho
"""



def main():
    pd.set_option('display.max_columns', None)    
    columns_char_identity = ['permno','date','y','m','ym','siccd','sic2','ticker','comnam','permco','hsiccd','cusip']
    columns_char_float = ['mvel1','beta','betasq','chmom','dolvol','idiovol','indmom','mom1m','mom6m','mom12m','mom36m','pricedelay','turn','absacc','acc','age','agr','bm','bm_ia','cashdebt','cashpr','cfp','cfp_ia','chatoia','chcsho','chempia','chinv','chpmia','convind','currat','depr','divi','divo','dy','egr','ep','gma','grcapx','grltnoa','herf','hire','invest','lev','lgr','mve_ia','operprof','orgcap','pchcapx_ia','pchcurrat','pchdepr','pchgm_pchsale','pchquick','pchsale_pchinvt','pchsale_pchrect','pchsale_pchxsga','pchsaleinv','pctacc','ps','quick','rd','rd_mve','rd_sale','realestate','roic','salecash','saleinv','salerec','secured','securedind','sgr','sin','sp','tang','tb','aeavol','cash','chtx','cinvest','ear','nincr','roaq','roavol','roeq','rsup','stdacc','stdcf','ms','baspread','ill','maxret','retvol','std_dolvol','std_turn','zerotrade',]
    columns_proxies = ['cefd', 'nipo', 'ripo', 'pdnd', 's']
    columns_macros = ['dp','ep','bm','ntis','tbl','tms','dfy','svar']
    gc.disable()
    
    # load data
    df, columns_char_industry, columns_inter_char_proxies, columns_inter_char_macros = get_data()
    gc.disable()

    # Training the model
    # char, proxies only    
    xlist = columns_char_float + columns_char_industry + columns_proxies + columns_inter_char_proxies
    xlist_name = 'char_proxies'
    df = linear_test(df, xlist, xlist_name)
    xlist = columns_char_float + columns_char_industry + columns_proxies
    xlist_name = 'char_proxies'
    df = nonlinear_test(df, xlist, xlist_name)
    gc.disable()
    # char, macros only
    xlist = columns_char_float + columns_char_industry + columns_macros + columns_inter_char_macros
    xlist_name = 'char_macros'
    df = linear_test(df, xlist, xlist_name)
    xlist = columns_char_float + columns_char_industry + columns_macros
    xlist_name = 'char_macros'
    df = nonlinear_test(df, xlist, xlist_name)
    gc.disable()
    # char, proxies and macros together
    xlist = columns_char_float + columns_char_industry + columns_proxies + columns_inter_char_proxies + columns_macros + columns_inter_char_macros
    xlist_name = 'char_proxies_macros'
    df = linear_test(df, xlist, xlist_name)
    xlist = columns_char_float + columns_char_industry + columns_proxies + columns_macros
    xlist_name = 'char_proxies_macros'
    df = nonlinear_test(df, xlist, xlist_name)
    gc.disable()
    df.to_csv('df_new_char_proxies_macros.csv', index=False)
    
    return df
    
    
df = main()    
df.to_csv('df_new_char_proxies_macros.csv', index=False)
