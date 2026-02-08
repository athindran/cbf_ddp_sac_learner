# CBF-DDP V2 within a training loop
This is an extension for using https://github.com/athindran/CBFDDP_Soft_v2 within a training loop. The code is not clean enough to be made public.

# Usage Instructions

We run three different types of training:

1) Training with no CBFDDP.
2) Training with CBF DDP with no penalty for using the filter.
3) Training CBF DDP with penalty for using the filter in the reward.

The training scripts are as follows:

```
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 123 --hidden_dim 256 --num_train_steps 320000 --filter_type none --training_mode
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 131 --hidden_dim 256 --num_train_steps 320000 --filter_type none --training_mode
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 253 --hidden_dim 256 --num_train_steps 320000 --filter_type none --training_mode
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 324 --hidden_dim 256 --num_train_steps 320000 --filter_type none --training_mode
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 444 --hidden_dim 256 --num_train_steps 320000 --filter_type none --training_mode

python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 123 --hidden_dim 256 --num_train_steps 320000 --filter_type SoftCBF --training_mode
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 131 --hidden_dim 256 --num_train_steps 320000 --filter_type SoftCBF --training_mode
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 253 --hidden_dim 256 --num_train_steps 320000 --filter_type SoftCBF --training_mode
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 324 --hidden_dim 256 --num_train_steps 320000 --filter_type SoftCBF --training_mode
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 444 --hidden_dim 256 --num_train_steps 320000 --filter_type SoftCBF --training_mode


python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 123 --hidden_dim 256 --num_train_steps 320000 --filter_type SoftCBF --training_mode --penalize_safety_filter_active
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 131 --hidden_dim 256 --num_train_steps 320000 --filter_type SoftCBF --training_mode --penalize_safety_filter_active
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 253 --hidden_dim 256 --num_train_steps 320000 --filter_type SoftCBF --training_mode --penalize_safety_filter_active
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 324 --hidden_dim 256 --num_train_steps 320000 --filter_type SoftCBF --training_mode --penalize_safety_filter_active
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 444 --hidden_dim 256 --num_train_steps 320000 --filter_type SoftCBF --training_mode --penalize_safety_filter_active
```

We do four different types of evaluation:

1) Train with no CBF, evaluate with no CBF
2) Train with no CBF, evaluate with CBF
3) Train with CBF with no filtering penalty, evaluate with CBF
4) Train with CBF with filtering penalty, evaluate with CBF

```
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 253 --load_dir $model_dir --load_index 287252 --filter_type none --plot_tag "train_no_filter_eval_no_filter" --miniplot
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 253 --load_dir $model_dir --load_index 287252 --filter_type SoftCBF --plot_tag "train_no_filter_eval_SoftCBF" --miniplot
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 123 --load_dir $model_dir --load_index 319999 --filter_type SoftCBF --plot_tag "train_filter_no_penalty_eval_SoftCBF" --miniplot
python train_sac_withcbfddp.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D_train.yaml -rb 3.0 -ls 'baseline' --seed 324 --load_dir $model_dir --load_index 319999 --filter_type SoftCBF --penalize_safety_filter_active --plot_tag "train_filter_penalty_eval_SoftCBF" --miniplot
```

## Videos

#### Train with no filter, evaluate with no filter - 20 episodes
<p align="center">
<img src="./videos/train_with_no_cbf_and_eval_with_no_cbf.gif" width="480" height="400" />
</p>

#### Train with no filter, evaluate with CBFDDP-SM filter - 20 episodes
<p align="center">
<img src="./videos/train_with_no_cbf_and_eval_with_cbf.gif" width="480" height="400" />
</p>

#### Train with CBFDDP-SM filter and no penalty, evaluate with CBFDDP-SM filter - 20 episodes
<p align="center">
<img src="./videos/train_with_softcbf_no_penalty_eval_rollout.gif" width="480" height="400" />
</p>

#### Train with CBFDDP-SM filter and penalty, evaluate with CBFDDP-SM filter - 20 episodes
<p align="center">
<img src="./videos/train_with_softcbf_penalty_eval_rollout.gif" width="480" height="400" />
</p>


## Acknowledgements

This code is based on the previous codebase of Safe Robotics Lab in Princeton ( https://saferobotics.princeton.edu/ )
For the SAC training loop, we refer to ( https://github.com/MishaLaskin/curl ). The relevant paper has been cited in the dissertation.

