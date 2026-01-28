import numpy as np
import torch

from summary.utils import(
    make_animation_plots,
    make_bicycle_comparison_report,
    plot_run_summary)
from simulators import(
    load_config,
    CarSingleEnv,
    BicycleReachAvoidMargin,
    PrintLogger,
    BicycleCost)
from learned_policies import(
    SacAgent,
    set_seed_everywhere, 
    eval_mode, 
    make_dir,
    Logger,
    ReplayBuffer)
import jax
from shutil import copyfile
import argparse
import imageio
import copy
from typing import Dict
import os
import sys
import time
import json

sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = " "

jax.config.update('jax_platform_name', 'cpu')

from learned_policies import(SacAgent,  
    ReplayBuffer)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_buffer_capacity', default=5000000, type=int)
    # train
    parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=550000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    # eval
    parser.add_argument('--eval_freq', default=100, type=int)
    parser.add_argument('--num_eval_episodes', default=40, type=int)
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.001, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    parser.add_argument('--actor_limit', default=1.0, type=int)
    # # encoder
    # parser.add_argument('--encoder_type', default='MultiRepsequence', type=str)
    # parser.add_argument('--encoder_feature_dim', default=14, type=int)
    # parser.add_argument('--encoder_lr', default=5e-5, type=float)
    # parser.add_argument('--encoder_tau', default=0.05, type=float)
    # parser.add_argument('--num_layers', default=1, type=int)
    # parser.add_argument('--num_filters', default=32, type=int)
    # parser.add_argument('--latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=3e-5, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument('--work_dir', default='/Users/athindranrameshkumar/Documents/Code/cbf_ddp_sac_learner/model_store', type=str)
    parser.add_argument('--load_dir', default='None', type=str)
    parser.add_argument('--load_index', default=0, type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=True, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--log_interval', default=500, type=int)

    parser.add_argument(
        "-cf",
        "--config_file",
        help="Config file path",
        type=str,
        default=os.path.join(
            "./simulators/test_config_yamls",
            "test_config.yaml"))

    parser.add_argument(
        "-rb", "--road_boundary", help="Choose road width", type=float,
        default=3.5
    )

    parser.add_argument(
        "-ls", "--line_search", help="Choose line search", type=str, default='baseline'
    )

    parser.add_argument(
        "-sp", "--stopping_computation", help="Choose stopping path as rollout or analytic", type=str, default='rollout'
    )
    parser.add_argument('--naive_task', dest='naive_task', action='store_true')
    parser.add_argument(
        '--no-naive_task',
        dest='naive_task',
        action='store_false')
    parser.add_argument('--should_animate', dest='should_animate', action='store_true')
    parser.set_defaults(naive_task=False)
    
    args = parser.parse_args()

    return args


def make_sac_agent(obs_shape, action_shape, args, device):
    return SacAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        hidden_dim=args.hidden_dim,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        log_interval=args.log_interval,
    )


def main(config_file, road_boundary, filter_type, is_task_ilqr, line_search, stopping_computation='rollout'):
    args = parse_args()
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    set_seed_everywhere(args.seed)

    ## ------------------------------------- Warmup fields ------------------------------------------ ##
    config = load_config(config_file)
    config_env = config['environment']
    config_agent = config['agent']
    config_solver = config['solver']
    config_solver.LINE_SEARCH = line_search
    config_agent.is_task_ilqr = False
    config_solver.FILTER_TYPE = filter_type
    config_agent.FILTER_TYPE = filter_type

    # use only ILQR for comparison to CBF.
    if filter_type == 'CBF':
        config_solver.ORDER ='ILQR'

    config_cost = config['cost']
    dyn_id = config_agent.DYN
    plot_tag = config_env.tag

    # Provide common fields to cost
    config_cost.N = config_solver.N
    if not hasattr(config_cost, 'V_MIN'):
        config_cost.V_MIN = config_agent.V_MIN
    if not hasattr(config_cost, 'V_MAX'):
        config_cost.V_MAX = config_agent.V_MAX
    if not hasattr(config_cost, 'DELTA_MIN'):
        config_cost.DELTA_MIN = config_agent.DELTA_MIN
    if not hasattr(config_cost, 'DELTA_MAX'):
        config_cost.DELTA_MAX = config_agent.DELTA_MAX

    config_cost.TRACK_WIDTH_RIGHT = road_boundary
    config_cost.TRACK_WIDTH_LEFT = road_boundary
    config_env.TRACK_WIDTH_RIGHT = road_boundary
    config_env.TRACK_WIDTH_LEFT = road_boundary
    config_cost.STOPPING_COMPUTATION_TYPE = stopping_computation

    env = CarSingleEnv(config_env, config_agent, config_cost)
    x_cur = np.array(
        getattr(
            config_solver, "INIT_STATE", [
                0., 0., 0.5, 0., 0.]))
    env.reset(x_cur)

    config_ilqr_cost = copy.deepcopy(config_cost)
    policy_type = None
    cost = None
    config_solver.COST_TYPE = config_cost.COST_TYPE
    if config_cost.COST_TYPE == "Reachavoid":
        if config_solver.FILTER_TYPE == "none":
            policy_type = "iLQRReachAvoid"
            cost = BicycleReachAvoidMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
            env.cost = cost
        else:
            policy_type = "iLQRSafetyFilter"
            task_cost = BicycleCost(
                config_ilqr_cost, copy.deepcopy(
                    env.agent.dyn))
            cost = BicycleReachAvoidMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
            # we use soft margin for an apples-to-apples comparison of the margin
            evaluation_cost = BicycleReachAvoidMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), 'SoftCBF')
            env.cost = cost
    elif config_cost.COST_TYPE == "Reachability":
        if config_solver.FILTER_TYPE == "none":
            policy_type = "iLQRReachability"
            cost = BicycleReachAvoidMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
            env.cost = cost
        else:
            policy_type = "iLQRSafetyFilter"
            task_cost = BicycleCost(
                config_ilqr_cost, copy.deepcopy(
                    env.agent.dyn))
            cost = BicycleReachAvoidMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
            # we use soft margin for an apples-to-apples comparison of the margin
            evaluation_cost = BicycleReachAvoidMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), 'SoftCBF')
            env.cost = cost

    env.agent.init_policy(
        policy_type=policy_type,
        config=config_solver,
        cost=cost,
        evaluation_cost=evaluation_cost,
        task_cost=task_cost)
    max_iter_receding = config_solver.MAX_ITER_RECEDING

    # region: Runs iLQR
    # Warms up jit
    env.agent.get_action(obs=x_cur, state=x_cur, warmup=True)
    env.report()

    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d-%s", ts)    
    env_name = 'racecar_safety-debug' + str(ts)

    args.work_dir = os.path.join(args.work_dir, env_name)
    os.makedirs(args.work_dir, exist_ok=True)
    model_dir = make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = make_dir(os.path.join(args.work_dir, 'buffer'))

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = (env.agent.dyn.dim_u,)
    obs_shape = (env.agent.dyn.dim_x,)

    replay_buffer = ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
    )

    sac_agent = make_sac_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    # Load model
    # sac_agent.load(model_dir='/Users/athindranrameshkumar/Documents/Code/cbf_ddp_sac_learner/model_store/racecar_safety-debug01-26-1769489934/model/',
    #                step=279304)

    L = Logger(args.work_dir, use_tb=args.save_tb)

    animate_dir = os.path.join(args.work_dir, 'animate')

    env.train_sac_agent(sac_agent, replay_buffer, L, args, max_episode_length=max_iter_receding,  config_solver=config_solver, verbose=False)
    #env.evaluate_sac_agent(sac_agent, 40, L, 0, args)

    obs_history, action_history, reward_history, done_history = env.simulate_trajectory_with_sac_agent(
         T_rollout=max_iter_receding, end_criterion='failure', sac_agent=sac_agent, verbose=True, num_trajs=10,
         sample_stochastically=False, should_animate=True, animate_dir=animate_dir, config_solver=config_solver,
    )


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    
    out_folder, plot_tag, config_agent = None, None, None
    jax.clear_caches()
    main(args.config_file, args.road_boundary, filter_type='SoftCBF', is_task_ilqr=(not args.naive_task),         
                                                line_search=args.line_search,
                                                stopping_computation=args.stopping_computation)
