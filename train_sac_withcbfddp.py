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
    parser.add_argument('--replay_buffer_capacity', default=2000000, type=int)
    # train
    parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps', default=10, type=int)
    parser.add_argument('--num_train_steps', default=230000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    # eval
    parser.add_argument('--eval_freq', default=100, type=int)
    parser.add_argument('--num_eval_episodes', default=20, type=int)
    # critic
    parser.add_argument('--critic_lr', default=5e-5, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.05, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=5e-5, type=float)
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
    parser.add_argument('--alpha_lr', default=5e-6, type=float)
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
    parser.add_argument('--training_mode', default=False, action='store_true')
    parser.add_argument('--penalize_safety_filter_active', default=False, action='store_true')

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
    parser.add_argument('--filter_type', help='Choose any/whether safety filter', type=str, default="none")
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


def main(config_file, road_boundary, filter_type, is_task_ilqr, is_task_rl, 
            line_search, stopping_computation='rollout'):
    # Callback after each timestep for plotting and summarizing evaluation
    def rollout_step_callback(
            env: CarSingleEnv,
            state_history,
            obs_history,
            action_history,
            plan_history,
            step_history,
            *args,
            **kwargs):
        solver_info = plan_history[-1]
        states = np.asarray(state_history).T  # last one is the next state.
        action_history = np.asarray(action_history)
        make_animation_plots(
            env,
            obs_history,
            action_history,
            solver_info,
            kwargs['safety_plan'],
            config_solver,
            config_agent,
            np.asarray(kwargs['barrier_filter_indices']),
            np.asarray(kwargs['complete_filter_indices']),
            fig_prog_folder)

        if config_solver.FILTER_TYPE == "none":
            print(
                "[{}]: solver returns status {}, cost {:.1e}, and uses {:.3f}.".format(
                    states.shape[1] - 1,
                    solver_info['status'],
                    solver_info['Vopt'],
                    solver_info['t_process']),
                end=' -> ')
        else:
            print(
                "[{}]: solver returns status {}, Vopt {:.1e}, future Vopt {:.1e}, marginopt {:.1e}, future marginopt {:.1e}, and uses {:.3f}.".format(
                    states.shape[1] - 1,
                    solver_info['status'],
                    solver_info['Vopt'],
                    solver_info['Vopt_next'],
                    solver_info['marginopt'],
                    solver_info['marginopt_next'],
                    solver_info['process_time']))
            # Turn off QCQP solver if it stalls.
            #assert solver_info['process_time']<0.09
    
    # Callback after episode for plotting and summarizing evaluation
    def rollout_episode_callback(
            env,
            state_history,
            obs_history,
            action_history,
            plan_history,
            step_history,
            *args,
            **kwargs):
        plot_run_summary(
            dyn_id,
            env,
            obs_history,
            action_history,
            config_solver,
            config_agent,
            fig_folder,
            **kwargs)
        save_dict = {
            'states': state_history,
            'obses': obs_history,
            'actions': action_history,
            "values": kwargs["value_history"],
            "process_times": kwargs["process_time_history"],
            "barrier_indices": kwargs["barrier_filter_indices"],
            "complete_indices": kwargs["complete_filter_indices"],
            'deviation_history': kwargs['deviation_history'],
            'safety_metrics': kwargs['safety_metric_history'],
            'safe_opt_history': kwargs['safe_opt_history'],
            'task_ctrl_history': kwargs['task_ctrl_history']}
        save_dict_str = os.path.join(fig_folder, "save_data.npy")
        print(f"Saving to: {save_dict_str}")
        np.save(save_dict_str, save_dict)

        solver_info = plan_history[-1]
        if config_solver.FILTER_TYPE != "none":
            print(
                "\n\n --> Barrier filtering performed at {:.3f} steps.".format(
                    solver_info['barrier_filter_steps']))
            print(
                "\n\n --> Complete filtering performed at {:.3f} steps.".format(
                    solver_info['filter_steps']))


    args = parse_args()
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    set_seed_everywhere(args.seed)

    ## ------------------------------------- Warmup fields ------------------------------------------ ##
    config = load_config(config_file)
    config_env = config['environment']
    config_agent = config['agent']
    config_solver = config['solver']
    config_env.penalize_safety_filter_active = args.penalize_safety_filter_active
    config_env.SEED = args.seed
    config_agent.SEED = args.seed
    config_solver.LINE_SEARCH = line_search
    config_agent.is_task_ilqr = is_task_ilqr
    config_agent.is_task_rl = is_task_rl
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
            policy_type = "SACPolicy"
            cost = BicycleReachAvoidMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
            evaluation_cost = BicycleReachAvoidMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), 'SoftCBF')
            env.cost = cost
            task_cost = None
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
            policy_type = "SACPolicy"
            cost = BicycleReachAvoidMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
            evaluation_cost = BicycleReachAvoidMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), 'SoftCBF')
            env.cost = cost
            task_cost = None
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

    # make directory
    if args.training_mode:
        ts = time.gmtime() 
        ts = time.strftime("%m-%d-%s", ts)    
        env_name = 'racecar_safety-debug-tests' + str(ts)
        args.work_dir = os.path.join(args.work_dir, env_name)
        os.makedirs(args.work_dir, exist_ok=True)
        model_dir = make_dir(os.path.join(args.work_dir, 'model'))
        buffer_dir = make_dir(os.path.join(args.work_dir, 'buffer'))
        sim_images_dir = os.path.join(args.work_dir, 'sim_images')
        os.makedirs(sim_images_dir, exist_ok=True)
    else:
        model_dir = make_dir(os.path.join(args.load_dir, 'model'))
        buffer_dir = make_dir(os.path.join(args.load_dir, 'buffer'))
        sim_images_dir = os.path.join(args.load_dir, 'sim_images')
        os.makedirs(sim_images_dir, exist_ok=True)

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

    env.agent.init_policy(
        policy_type=policy_type,
        config=config_solver,
        cost=cost,
        evaluation_cost=evaluation_cost,
        task_cost=task_cost,
        rl_task_policy=sac_agent)

    max_iter_receding = config_solver.MAX_ITER_RECEDING

    # region: Runs iLQR
    # Warms up jit
    env.agent.get_action(obs=x_cur, state=x_cur, warmup=True)
    env.report()
    L = Logger(args.work_dir, use_tb=args.save_tb)

    if args.training_mode:
        env.train_sac_agent(sac_agent, replay_buffer, L, args, max_episode_length=max_iter_receding,  config_solver=config_solver, verbose=False)
        env.evaluate_sac_agent(sac_agent, args.num_eval_episodes, L, args.num_train_steps, args)
    else:
        # Load model
        sac_agent.load(model_dir=model_dir,
                    step=args.load_index)
        sim_images_dir = os.path.join(args.load_dir, 'sim_images')
    # obs_history, action_history, reward_history, done_history = env.simulate_trajectory_with_sac_agent(
    #      T_rollout=max_iter_receding, end_criterion='failure', sac_agent=sac_agent, verbose=True, num_trajs=10,
    #      sample_stochastically=False, should_animate=True, animate_dir=animate_dir, config_solver=config_solver,
    # )

    # Works only with SoftCBF filters now.
    should_animate = True
    for traj_indx in range(10):
        sim_images_dir_per_traj = os.path.join(sim_images_dir, f'traj_{traj_indx}/')
        current_sim_images_dir = os.path.join(sim_images_dir_per_traj,
            "road_boundary=" + str(road_boundary))
        current_sim_images_dir = os.path.join(current_sim_images_dir, args.filter_type)
        os.makedirs(current_sim_images_dir, exist_ok=True)

        copyfile(
            config_file,
            os.path.join(
                current_sim_images_dir,
                'config.yaml'))
        sys.stdout = PrintLogger(
            os.path.join(
                current_sim_images_dir,
                'log.txt'))
        sys.stderr = PrintLogger(
            os.path.join(
                current_sim_images_dir,
                'log.txt'))

        fig_folder = os.path.join(current_sim_images_dir, "figure")
        fig_prog_folder = os.path.join(fig_folder, "progress")
        os.makedirs(fig_prog_folder, exist_ok=True)

        nominal_states, result, traj_info = env.simulate_one_trajectory(
            T_rollout=max_iter_receding, end_criterion='failure',
            rollout_step_callback=rollout_step_callback,
            rollout_episode_callback=rollout_episode_callback,
            advanced_animate=should_animate,
        )

        print(f"--------------------RESULT: {result}----------------------")
        print(traj_info['step_history'][-1]["done_type"])
        constraints: Dict = traj_info['step_history'][-1]['constraints']
        for k, v in constraints.items():
            print(f"{k}: {v[0, 1]:.1e}")
        print("-----------------------------------------------------------")
        
        if should_animate:
            # region: Visualizes
            gif_path = os.path.join(fig_folder, 'rollout.gif')
            frame_skip = 10
            with imageio.get_writer(gif_path, mode='I') as writer:
                for i in range(len(nominal_states) - 1):
                    if frame_skip != 1 and (i + 1) % frame_skip != 0:
                        continue
                    filename = os.path.join(
                        fig_prog_folder, str(i + 1) + ".png")
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    #Image(open(gif_path, 'rb').read(), width=400)
            # endregion

        if args.filter_type == 'SoftCBF':
            make_bicycle_comparison_report(
                sim_images_dir_per_traj,
                plot_folder=f'./sac_safety_filter_summary_rollout_{args.line_search}-{args.stopping_computation}/',
                tag=plot_tag + "_" + str(args.road_boundary) + "_sim_index_" + str(traj_indx) + "_",
                road_boundary=args.road_boundary,
                dt=config_agent.DT,
                cbf_gamma=config_solver.CBF_GAMMA,
                soft_cbf_gamma=config_solver.SOFT_CBF_GAMMA,
                filters=['SoftCBF'])

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    
    out_folder, plot_tag, config_agent = None, None, None
    jax.clear_caches()
    main(args.config_file, args.road_boundary, filter_type=args.filter_type, is_task_ilqr=False, is_task_rl=True,         
                                                line_search=args.line_search,
                                                stopping_computation=args.stopping_computation)
