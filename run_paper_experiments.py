from sumie.utils.experiment import exps_from_config_diffs

print('=======================================')
print('Running sequence labeling experiments!')
print('=======================================')
exps_from_config_diffs('configs/paper_sequence_labeling_configs.json', generated_configs_dir='generated_configs/', run_exps=True)

print('=======================================')
print('Running baseline experiments!')
print('=======================================')
exps_from_config_diffs('configs/paper_baseline_configs.json', generated_configs_dir='generated_configs/', run_exps=True)
