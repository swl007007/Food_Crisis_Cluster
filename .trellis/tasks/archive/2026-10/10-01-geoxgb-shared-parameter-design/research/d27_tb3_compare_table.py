import pandas as pd, json
R = r'C:\Users\swl00\geoxgb_runs\geoxgb-d27-tb3-20261001\stage1_tb3_compare'
p = pd.read_csv(R + r'\pairs.csv'); c = pd.read_csv(R + r'\candidates.csv')
pd.set_option('display.width', 260)
cols = ['horizon','target_month','root_change_tb3_minus_r80','local_change_tb3_minus_r80','e3_gain_tb3','e3_gain_r80','e3_gain_difference_tb3_minus_r80','e2_gain_tb3_own_rows','e2_gain_r80_own_rows','n_terminal_tb3','n_terminal_r80','e4_weight_tb3','e4_weight_r80','n_target']
print(p[cols].round(4).to_string(index=False))
cc = ['method','horizon','target_month','e3_root_crisis_f1','e3_local_crisis_f1','e3_local_minus_root','e3_root_fourclass','e3_local_fourclass','e2_final_minus_root','n_terminal','distinct_terminal_boosters','child_fits_including_rejected','fitting_rows','fitting_dates','fitting_crisis_positives','validation_rows','validation_dates','validation_crisis_positives','validation_only_areas']
print(c[cc].sort_values(['horizon','target_month','method']).round(4).to_string(index=False))
