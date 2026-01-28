from abc import abstractmethod
from typing import Any, Tuple, Optional, Callable, List, Dict, Union
import numpy as np
from gym import spaces
from tqdm import tqdm
from matplotlib import pyplot as plt
import imageio
import os
import time
import json

from .agent import Agent
from .base_env import BaseEnv

from .policy.base_policy import BasePolicy

from learned_policies import(
    SacAgent,
    set_seed_everywhere, 
    eval_mode, 
    make_dir,
    Logger,
    ReplayBuffer)

class BaseSingleEnv(BaseEnv):
    """Implements an environment of a single agent.
    """

    def __init__(self, config_env: Any, config_agent: Any) -> None:
        super().__init__(config_env)
        self.env_type = "single-agent"

        # Action Space.
        action_space = np.array(config_agent.ACTION_RANGE, dtype=np.float32)
        self.action_dim = action_space.shape[0]
        self.action_dim_ctrl = action_space.shape[0]
        self.agent = Agent(config_agent, action_space)
        self.action_space = spaces.Box(
            low=action_space[:, 0], high=action_space[:, 1]
        )
        self.state_dim = self.agent.dyn.dim_x

        self.integrate_kwargs = getattr(config_env, "INTEGRATE_KWARGS", {})
        if "noise" in self.integrate_kwargs:
            if self.integrate_kwargs['noise'] is not None:
                self.integrate_kwargs['noise'] = np.array(
                    self.integrate_kwargs['noise']
                )

    def step(
        self, action: np.ndarray, cast_torch: bool = False
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Implements the step function in the environment.

        Args:
            action (np.ndarray).
            cast_torch (bool): cast state to torch if True.

        Returns:
            np.ndarray: next state.
            float: the reward that ctrl wants to maximize and dstb wants to
                minimize.
            bool: True if the episode ends.
            Dict[str, Any]]: additional information of the step, such as target
                margin and safety margin used in reachability analysis.
        """

        self.cnt += 1
        state_nxt = self.agent.integrate_forward_with_noise(
            state=self.state, control=action, **self.integrate_kwargs
        )[0]
        state_cur = self.state.copy()
        self.state = state_nxt.copy()
        constraints = self.get_constraints(state_cur, action, state_nxt)
        cost = self.get_cost(state_cur, action, state_nxt, constraints)
        #targets = self.get_target_margin(state_cur, action, state_nxt)
        done, info = self.get_done_and_info(state_nxt, constraints, None)

        obs = self.get_obs(state_nxt)

        return obs, cost, done, info

    def step_with_sac_agent(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Implements the step function for the RL environment.

        Args:
            action (np.ndarray).
            cast_torch (bool): cast state to torch if True.

        Returns:
            np.ndarray: next state.
            float: the reward that ctrl wants to maximize
                minimize.
            bool: True if the episode ends.
            Dict[str, Any]]: additional information of the step, such as target
                margin and safety margin used in reachability analysis.
        """

        self.cnt += 1
        action[0] *= self.agent.dyn.ctrl_space[0, 1]
        action[1] *= self.agent.dyn.ctrl_space[1, 1]

        state_nxt = self.agent.integrate_forward_with_noise(
            state=self.state, control=action, **self.integrate_kwargs
        )[0]
        state_cur = self.state.copy()
        self.state = state_nxt.copy()
        constraints = self.get_constraints(state_cur, action, state_nxt)
        done, info = self.get_done_and_info(state_nxt, constraints, None)
        obs = self.get_obs(state_nxt)

        # Penalty for violating constraint
        constraint_values = np.empty((1,))
        for key, constraint_value in constraints.items():
            constraint_values = np.concatenate((constraint_values, constraint_value.ravel()))
        reward_constraint = -250.0 if np.min(constraint_values, axis=0) <= 0.0 else 0.0

        centerline_maintenance_reward = -0.05*np.abs(obs[1])
        yaw_maintenance_reward = 0.0 if np.abs(obs[3])<1.0 else -0.2
        track_completion_reward = 150.0 if (state_nxt[0]>=self.track_len and done) else 0
        control_cost = -0.001 * action[0]**2 - 0.001 * action[1]**2
        progress_cost = max(0.01*obs[0], 0.1)

        reward = float(reward_constraint + centerline_maintenance_reward + progress_cost +
                    yaw_maintenance_reward + track_completion_reward + control_cost)

        return obs, reward, done, info

    @abstractmethod
    def get_cost(
        self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> float:
        """
        Gets the cost given current state, current action, and next state.

        Args:
            state (np.ndarray): current state.
            action (np.ndarray): current action.
            state_nxt (np.ndarray): next state.
            constraints (Dict): each (key, value) pair is the name and value of a
                constraint function.

        Returns:
            float: the cost to minimize.
        """
        raise NotImplementedError

    @abstractmethod
    def get_constraints(
        self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
    ) -> Dict:
        """
        Gets the values of all constaint functions given current state, current
        action, and next state.

        Args:
            state (np.ndarray): current states.
            action (np.ndarray): current actions.
            state_nxt (np.ndarray): next state.

        Returns:
            Dict: each (key, value) pair is the name and value of a constraint
                function.
        """
        raise NotImplementedError

    @abstractmethod
    def get_target_margin(
        self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
    ) -> Dict:
        """
        Gets the values of all target margin functions given current state, current
        action, and next state.

        Args:
            state (np.ndarray): current states.
            action (np.ndarray): current actions.
            state_nxt (np.ndarray): next state.

        Returns:
            Dict: each (key, value) pair is the name and value of a target margin
                function.
        """
        raise NotImplementedError

    @abstractmethod
    def get_done_and_info(
        self, state: np.ndarray, constraints: Dict, targets: Dict,
        final_only: bool = True, end_criterion: Optional[str] = None
    ) -> Tuple[bool, Dict]:
        """
        Gets the done flag and a dictionary to provide additional information of
        the step function given current state, current action, next state,
        constraints, and targets.

        Args:
            state (np.ndarray): current state.
            constraints (Dict): each (key, value) pair is the name and value of a
                constraint function.
            targets (Dict): each (key, value) pair is the name and value of a
                target margin function.

        Returns:
            bool: True if the episode ends.
            Dict: additional information of the step, such as target margin and
                safety margin used in reachability analysis.
        """
        raise NotImplementedError

    def simulate_one_trajectory(
        self, T_rollout: int, end_criterion: str,
        reset_kwargs: Optional[Dict] = None,
        action_kwargs: Optional[Dict] = None,
        rollout_step_callback: Optional[Callable] = None,
        rollout_episode_callback: Optional[Callable] = None,
        advanced_animate: bool = True, **kwargs
    ) -> Tuple[np.ndarray, int, Dict]:
        """
        Rolls out the trajectory given the horizon, termination criterion, reset
        keyword arguments, callback afeter every step, and callout after the
        rollout.

        Args:
            T_rollout (int): rollout horizon.
            end_criterion (str): termination criterion.
            reset_kwargs (Dict): keyword argument dictionary for reset function.
            action_kwargs (Dict): keyword argument dictionary for get_action
                function.
            rollout_step_callback (Callable): function to call after every step.
            rollout_episode_callback (Callable): function to call after rollout.

        Returns:
            np.ndarray: state trajectory.
            int: result (0: unfinished, 1: success, -1: failure).
            Dict: auxiliarry information -
                "action_history": action sequence.
                "plan_history": planning info for every step.
                "reward_history": rewards for every step.
                "step_history": information for every step.
        """
        # Stores the environment attributes and sets to rollout settings.
        timeout_backup = self.timeout
        end_criterion_backup = self.end_criterion
        self.timeout = T_rollout
        self.end_criterion = end_criterion

        state_history = []
        obs_history = []
        action_history = []
        reward_history = []
        plan_history = []
        step_history = []
        value_history = []
        safety_metric_history = []
        process_time_history = []
        solver_iters_history = []
        safe_opt_history = []
        task_ctrl_history = []
        complete_filter_indices = []
        barrier_filter_indices = []
        deviation_history = []
        # Initializes robot.
        controls_initialize = None

        result = 0
        obs = self.reset(**reset_kwargs)
        state_history.append(self.state)
        obs_history.append(obs)

        prev_sol = None
        margin_initializer = None
        prev_ctrl = np.array([0.0, 0.0])
        for t in range(T_rollout):
            #kwargs['state'] = self.state.copy()
            action, solver_info = self.agent.get_action(
                obs=obs, controls=controls_initialize,
                prev_sol=prev_sol, state=self.state, prev_ctrl=prev_ctrl
            )
            prev_ctrl = np.array( action )
            prev_sol = solver_info

            if solver_info['mark_barrier_filter']:
                barrier_filter_indices.append(t)
            if solver_info['mark_complete_filter']:
                complete_filter_indices.append(t)

            # Applies action: `done` and `info` are evaluated at the next
            # state.
            obs, reward, done, step_info = self.step(action)

            # Executes step callback and stores historyory.
            state_history.append(self.state)
            obs_history.append(obs)
            action_history.append(action)
            plan_history.append(solver_info)
            value_history.append(solver_info['Vopt'])
            reward_history.append(reward)
            step_history.append(step_info)
            process_time_history.append(solver_info['process_time'])
            solver_iters_history.append(solver_info['num_iters'])
            deviation_history.append(solver_info['deviation'])
            safe_opt_history.append(solver_info['safe_opt_ctrl'])
            task_ctrl_history.append(solver_info['task_ctrl'])

            if self.agent.compute_evaluation_margin:
                _, margin_info = self.agent.evaluation_margin_solver.get_action(obs=np.array(self.state), state=np.array(self.state), 
                                    controls=margin_initializer)
                safety_metric_history.append(margin_info['Vopt'].ravel()[0])
                margin_initializer = np.array(margin_info['controls'])
            else:
                safety_metrics_history.append(0.0)


            if advanced_animate:
                if self.cost_type=="Reachability":
                    safety_plan = np.asarray(solver_info['states'])
                elif self.cost_type=="Reachavoid":
                    # We plot the safety plan where we enter the target set, then
                    # decelerate and stop to remain safe for infinite time for an
                    # infeasible task plan

                    reachavoid_plan = solver_info['states']
                    reachavoid_plan_ctrl = solver_info['controls']
                    critical = solver_info['critical']

                    target_margins = self.cost.get_mapped_target_margin(
                        reachavoid_plan, reachavoid_plan_ctrl)

                    target_margins = np.array(target_margins)

                    is_inside_target_index = np.argwhere(critical!=0).ravel()

                    if is_inside_target_index.size == 0:
                        is_inside_target_index = 0
                    else:
                        is_inside_target_index = is_inside_target_index[0]

                    target_plan = np.array(
                        reachavoid_plan[:, 0:is_inside_target_index + 1])

                    stopping_plan = self.simulate_stopping_plan(initial_state=np.array(reachavoid_plan[:, is_inside_target_index]),
                                                                stopping_ctrl=np.array([self.agent.dyn.ctrl_space[0, 0], 0]))

                    safety_plan = np.concatenate(
                    (target_plan, np.array(stopping_plan).T), axis=1)
                if rollout_step_callback is not None:
                    rollout_step_callback(
                        self, state_history, obs_history, action_history, plan_history, step_history, safety_plan=safety_plan, 
                                barrier_filter_indices=barrier_filter_indices, complete_filter_indices=complete_filter_indices,
                    )
            else:
                safety_plan = np.asarray(solver_info['states'])

            # Checks termination criterion.
            if done:
                if step_info["done_type"] == "success":
                    result = 1
                elif step_info["done_type"] == "failure":
                    result = -1
                break

            controls_initialize = np.array(solver_info['reinit_controls'])

        if rollout_episode_callback is not None:
            rollout_episode_callback(
                self, state_history, obs_history, action_history, plan_history, step_history, value_history=value_history, process_time_history=process_time_history,
                solver_iters_history=solver_iters_history, deviation_history=deviation_history, safety_metric_history=safety_metric_history,
                safe_opt_history=safe_opt_history, task_ctrl_history=task_ctrl_history,
                barrier_filter_indices=barrier_filter_indices, complete_filter_indices=complete_filter_indices,
                label=self.agent.safety_policy.filter_type
            )
        # Reverts to training setting.
        self.timeout = timeout_backup
        self.end_criterion = end_criterion_backup
        info = dict(
            obs_history=np.array(obs_history), action_history=np.array(action_history),
            plan_history=plan_history, reward_history=np.array(reward_history),
            step_history=step_history
        )

        return np.array(state_history), result, info

    def simulate_trajectory_with_sac_agent(
        self, T_rollout: int, end_criterion: str, 
        num_trajs: int,
        sac_agent: SacAgent, config_solver, verbose: bool = False, 
        sample_stochastically: bool = True, animate_dir: str = './', should_animate: bool = False
    ):
        reset_rejection_sampling_old = self.reset_rej_sampling
        self.reset_rej_sampling = False
         # Stores the environment attributes and sets to rollout settings.
        self.timeout = T_rollout
        self.end_criterion = end_criterion

        for traj_indx in range(num_trajs):
            obs = self.reset()
            done = False
            episode_reward = 0
            obs_history = []
            action_history = []
            reward_history = []
            done_history = []
            train_step = 0
            animate_dir_curr = os.path.join(animate_dir, f'traj_{traj_indx}')
            os.makedirs(animate_dir_curr, exist_ok=True)
            animate_prog_dir = os.path.join(animate_dir_curr, 'images')
            os.makedirs(animate_prog_dir, exist_ok=True)
            
            while not done:
                if should_animate:
                    fig = plt.figure()
                    ax = plt.gca()

                    ax.axis(self.visual_extent)
                    ax.set_aspect('equal')

                    c_obs = 'k'
                    c_ego = 'c'
                    c_trace = 'k'

                    # track, obstacles, footprint
                    self.render_obs(ax=ax, c=c_obs)  
                    self.render_footprint(ax=ax, obs=obs, c=c_ego, lw=0.5)
                    obs_history_numpy = np.array(obs_history)
                    if obs_history_numpy.size > 0:
                        sc = ax.scatter(obs_history_numpy[:, 0], obs_history_numpy[:, 1], s=3, c=c_trace, marker='o')

                    fig.savefig(
                        os.path.join(animate_prog_dir,
                            str(train_step) + ".png"), dpi=200, bbox_inches="tight"
                    )
                    plt.close('all')

                # center crop image
                with eval_mode(sac_agent):
                    if sample_stochastically:
                        action = sac_agent.sample_action(obs)
                    else:
                        action = sac_agent.select_action(obs)

                obs, reward, done, step_info = self.step_with_sac_agent(action)
                episode_reward += reward
                obs_history.append(obs)
                action_history.append(action)
                reward_history.append(reward)
                done_history.append(done)
                train_step += 1

            if verbose:
                print(f"--------------------RESULT----------------------")
                print(f'End criterion: {step_info["done_type"]}')
                constraints: Dict = step_info['constraints']
                for k, v in constraints.items():
                    print(f"{k}: {v[0, 1]:.1e}")
                print(f"Episode_reward: {episode_reward}")
                print("-----------------------------------------------------------")
            
            if should_animate:
                # region: Visualizes
                gif_path = os.path.join(animate_dir_curr, 'rollout.gif')
                frame_skip = 5
                with imageio.get_writer(gif_path, mode='I') as writer:
                    for i in range(train_step - 1):
                        if frame_skip != 1 and (i + 1) % frame_skip != 0:
                            continue
                        filename = os.path.join(
                            animate_prog_dir, str(i + 1) + ".png")
                        image = imageio.imread(filename)
                        writer.append_data(image)
                        #Image(open(gif_path, 'rb').read(), width=400)
                # endregion

        self.reset_rej_sampling = reset_rejection_sampling_old

        return obs_history, action_history, reward_history, done_history

    def evaluate_sac_agent(self, sac_agent, num_episodes, L, step, args):
        all_ep_rewards = []
        reset_rejection_sampling_old = self.reset_rej_sampling
        self.reset_rej_sampling = False

        def run_eval_loop(sample_stochastically=True):
            start_time = time.time()
            prefix = 'stochastic_' if sample_stochastically else ''
            for i in range(num_episodes):
                obs = self.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    # center crop image
                    with eval_mode(sac_agent):
                        if sample_stochastically:
                            action = sac_agent.sample_action(obs)
                        else:
                            action = sac_agent.select_action(obs)

                    obs, reward, done, INFO = self.step_with_sac_agent(action)
                    episode_reward += reward

                L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
                all_ep_rewards.append(episode_reward)
            
            L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
            mean_ep_reward = np.mean(all_ep_rewards)
            best_ep_reward = np.max(all_ep_rewards)
            L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
            L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

        run_eval_loop(sample_stochastically=False)
        L.dump(step)

        self.reset_rej_sampling = reset_rejection_sampling_old

        return

    def train_sac_agent(self, sac_agent, replay_buffer, L, args, max_episode_length, config_solver, verbose=True):
        episode, episode_reward, done = 0, 0, True
        reset_rejection_sampling_old = self.reset_rej_sampling
        self.reset_rej_sampling = False
        animate_dir = make_dir(os.path.join(args.work_dir, 'animations'))
        model_dir = make_dir(os.path.join(args.work_dir, 'model'))
        buffer_dir = make_dir(os.path.join(args.work_dir, 'buffer'))

        with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        start_time = time.time()
    
        for train_step in range(args.num_train_steps):
            if done and episode % args.eval_freq == 0:
                L.log('eval/episode', episode, train_step)
                self.evaluate_sac_agent(sac_agent, args.num_eval_episodes, L, train_step, args)
                if args.save_model:
                    sac_agent.save(model_dir, train_step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            if done:
                if train_step > 0:
                    if True:
                        L.log('train/duration', time.time() - start_time, train_step)
                        L.dump(train_step)
                    if verbose:
                        print(f"--------------------RESULT: ----------------------")
                        print(f'End criterion: {step_info["done_type"]}')
                        constraints: Dict = step_info['constraints']
                        for k, v in constraints.items():
                            print(f"{k}: {v[0, 1]:.1e}")
                        print("-----------------------------------------------------------")
                    
                    if episode % args.eval_freq == 0:
                        _, _, _, _ = self.simulate_trajectory_with_sac_agent(
                            T_rollout=max_episode_length, end_criterion='failure', sac_agent=sac_agent, verbose=verbose, num_trajs=2,
                                sample_stochastically=False, should_animate=True, animate_dir=animate_dir + '_' + str(train_step), config_solver=config_solver,
                            )

                    start_time = time.time()
                #if train_step % args.log_interval == 0:
                if True:
                    L.log('train/episode_reward', episode_reward, train_step)
               
                obs = self.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                
                #if train_step % args.log_interval == 0:
                if True:
                    L.log('train/episode', episode, train_step)

            # sample action for data collection
            if train_step < args.init_steps:
                action = self.action_space.sample()
            else:
                with eval_mode(sac_agent):
                    action = sac_agent.sample_action(obs)

            # run training update
            if train_step >= args.init_steps:
                num_updates = 1 
                for _ in range(num_updates):
                    sac_agent.update(replay_buffer, L, train_step)

            next_obs, reward, done, step_info = self.step_with_sac_agent(action)
    
            # allow infinity bootstrap
            done_bool = 0 if episode_step + 1 == max_episode_length else float(
                done
            )
            episode_reward += reward
            replay_buffer.add(obs, action, reward, next_obs, done_bool)

            obs = np.array(next_obs)
            episode_step += 1

        self.reset_rej_sampling = reset_rejection_sampling_old

        return

    def simulate_stopping_plan(
            self, initial_state: np.ndarray, stopping_ctrl: np.ndarray):
        """
          Simulates the stopping plan from an initial state by applying maximum deceleration.
          Deceleration policy renders the target set controlled invariant.
        """
        states = [initial_state]
        current_state = np.array(initial_state)
        while current_state[2] > self.agent.dyn.v_min:
            current_state, _ = self.agent.integrate_forward(
                current_state, stopping_ctrl)
            states.append(current_state)

        return states

    def simulate_task_plan(self, initial_state: np.ndarray,
                           task_policy: BasePolicy, nsteps: int, is_ilqr: bool):
        """
          UNUSED: Simulates the task plan from an initial state
        """
        states = [initial_state]
        current_state = np.array(initial_state)
        idx = 0
        while idx < nsteps:
            idx = idx + 1
            if is_ilqr:
                task_ctrl = task_policy.get_action(
                    current_state, None, state=current_state)
            else:
                task_ctrl = task_policy(current_state)
            current_state, _ = self.agent.integrate_forward(
                current_state, task_ctrl)
            states.append(current_state)

        return states

    # Unused currently.
    def simulate_trajectories(
        self, num_trajectories: int, T_rollout: int, end_criterion: str,
        reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
        action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
        rollout_step_callback: Optional[Callable] = None,
        rollout_episode_callback: Optional[Callable] = None, return_info=False,
        **kwargs
    ):
        """
        Rolls out multiple trajectories given the horizon, termination criterion,
        reset keyword arguments, callback afeter every step, and callback after the
        rollout. Need to call env.reset() after this function to revert back to the
        training mode.
        """

        if isinstance(reset_kwargs_list, list):
            assert num_trajectories == len(reset_kwargs_list), (
                "The length of reset_kwargs_list does not match with",
                "the number of rollout trajectories"
            )
        if isinstance(action_kwargs_list, list):
            assert num_trajectories == len(action_kwargs_list), (
                "The length of action_kwargs_list does not match with",
                "the number of rollout trajectories"
            )

        results = np.empty(shape=(num_trajectories,), dtype=int)
        length = np.empty(shape=(num_trajectories,), dtype=int)
        trajectories = []
        info_list = []
        use_tqdm = kwargs.get('use_tqdm', False)
        if use_tqdm:
            iterable = tqdm(
                range(num_trajectories),
                desc='sim trajs',
                leave=False)
        else:
            iterable = range(num_trajectories)

        for trial in iterable:
            if isinstance(reset_kwargs_list, list):
                reset_kwargs = reset_kwargs_list[trial]
            else:
                reset_kwargs = reset_kwargs_list
            if isinstance(action_kwargs_list, list):
                action_kwargs = action_kwargs_list[trial]
            else:
                action_kwargs = action_kwargs_list

            state_history, result, info = self.simulate_one_trajectory(
                T_rollout=T_rollout, end_criterion=end_criterion,
                reset_kwargs=reset_kwargs, action_kwargs=action_kwargs,
                rollout_step_callback=rollout_step_callback,
                rollout_episode_callback=rollout_episode_callback, **kwargs
            )
            trajectories.append(state_history)
            results[trial] = result
            length[trial] = len(state_history)
            info_list.append(info)
        if return_info:
            return trajectories, results, length, info_list
        else:
            return trajectories, results, length
