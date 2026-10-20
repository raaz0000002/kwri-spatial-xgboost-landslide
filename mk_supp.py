import pandas as pd, os
TT=next(x for x in ['_rx2','_rx','_nospec'] if os.path.exists(f'out_metrics_summary{x}.csv'))
with pd.ExcelWriter('Supplementary_Tables.xlsx') as w:
    def put(df,name): df.to_excel(w,sheet_name=name,index=False)
    put(pd.DataFrame({'Table':['S1a','S1b','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17'],
      'Content':['Training table vs district raster: distribution statistics (all predictors)','Training table vs raster values sampled at the 1200 coordinates (incl. CHIRPS rainfall and recomputed distances)',
                 'Per-fold predictor-set decisions, VIF screening, inner-CV AUCs (Model T and Model S)','Per-fold selected hyperparameters (Model T and Model S)',
                 'DeLong tests, all repeats (Model T and Model S)','Permutation importance, all predictors and families, every model (Model T and Model S)',
                 'Out-of-fold performance summary, raw and calibrated (Model T and Model S)','Validation-scheme sensitivity incl. coordinates-only baseline',
                 'Buffered 10-km block CV sensitivity (Model T, RF)','VIF with and without intercept','Sample-level area of applicability (Model T)','Pixel shares below/within/above training range','NoData audit','Quintile class breaks and areas','Class-wise validation with held-out landslides','Non-landslide (pseudo-absence) sensitivity: performance, coordinates baseline, map agreement','Pseudo-absence sensitivity: family permutation importance','VIF on raster-sampled values (with/without curvature)']}),'README')
    put(pd.read_csv('sample_vs_raster.csv'),'S1a_distribution')
    put(pd.read_csv('out_extract_vs_table.csv'),'S1b_point_extraction')
    for t,n in [(TT,'T'),('','S')]:
        put(pd.read_csv(f'out_fold_decisions{t}.csv'),f'S2_decisions_Model{n}')
        put(pd.read_csv(f'out_fold_params{t}.csv'),f'S3_params_Model{n}')
        put(pd.read_csv(f'out_delong{t}.csv'),f'S4_DeLong_Model{n}')
        put(pd.read_csv(f'out_perm_importance{t}.csv'),f'S5_importance_Model{n}')
        put(pd.read_csv(f'out_metrics_summary{t}.csv'),f'S6_performance_Model{n}')
    sc=pd.concat([pd.read_csv('out_scheme_sensitivity.csv').assign(predictor_set=lambda d:d.features.replace({'Predictors':'Model S'})),pd.read_csv(f'out_scheme_sensitivity{TT}.csv').assign(predictor_set='Model T')])
    put(sc,'S7_scheme_sensitivity')
    put(pd.read_csv(f'out_buffer_sensitivity{TT if TT.startswith("_rx") else ""}.csv'),'S8_buffer')
    put(pd.read_csv('out_vif_check.csv').rename(columns={'Unnamed: 0':'predictor'}),'S9_VIF')
    put(pd.read_csv(f'out_sample_aoa{TT if TT.startswith("_rx") else ""}.csv'),'S10_sample_AOA')
    if os.path.exists('out_pixel_coverage.csv'): put(pd.read_csv('out_pixel_coverage.csv'),'S11_pixel_coverage'); put(pd.read_csv('out_nodata_audit.csv'),'S12_nodata_audit'); put(pd.read_csv('out_area_classes.csv'),'S13_quintile_classes'); put(pd.read_csv('out_class_validation.csv'),'S14_class_validation')
    if os.path.exists(f'out_pseudoabsence{TT}.csv'): put(pd.read_csv(f'out_pseudoabsence{TT}.csv'),'S15_pseudoabsence'); put(pd.read_csv(f'out_pseudoabsence_importance{TT}.csv'),'S16_pseudoabs_importance')
    put(pd.read_csv('out_vif_raster.csv').rename(columns={'Unnamed: 0':'predictor'}),'S17_VIF_raster')
print('ok')
