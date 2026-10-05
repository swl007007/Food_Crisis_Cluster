import pandas as pd
R = r'C:\Users\swl00\geoxgb_runs\geoxgb-d28-rootinc-20261001\stage1_rootinc_compare'
p = pd.read_csv(R + r'\pairs.csv'); c = pd.read_csv(R + r'\candidates.csv')
pd.set_option('display.width', 250)
print(p[['horizon','target_month','root_crisis_f1','local_rootinc','local_r80','e3_gain_rootinc','e3_gain_r80','e3_gain_difference_rootinc_minus_r80','e3_local_fp_change','e3_local_tp_change','e3_local_fn_change','e2_gain_rootinc','e2_gain_r80','n_terminal_rootinc','n_terminal_r80','deployed_rounds_max_rootinc','deployed_rounds_max_r80']].round(4).to_string(index=False))
print(c[['method','horizon','target_month','e3_root_fp','e3_local_fp','e3_local_tp','distinct_terminal_boosters','search_budget_rounds_max','fits_attempted','e3_local_fourclass','e3_root_fourclass']].sort_values(['horizon','target_month','method']).round(4).to_string(index=False))
