from typing import Tuple, Any
import numpy as np
from functools import partial
from jax import Array as DeviceArray
import jax
from jax import numpy as jnp
from jax import custom_jvp
from jax import random

from .base_dynamics import BaseDynamics


class Bicycle4D(BaseDynamics):

    def __init__(self, config: Any, action_space: np.ndarray) -> None:
        """
        Implements the bicycle dynamics (for Princeton race car). The state is the
        center of the rear axis.
        Args:
            config (Any): an object specifies configuration.
            action_space (np.ndarray): action space.
        """
        super().__init__(config, action_space)
        self.dim_x = 4  # [x, y, v, psi, delta].

        # load parameters
        self.wheelbase: float = config.WHEELBASE  # vehicle chassis length
        self.v_min = 0
        self.v_max = config.V_MAX
        self.rear_wheel_offset = 0.4 * self.wheelbase
        self.noise_var = jnp.array([0.01, 0.01, 0.01, 0.01])
        self.stopping_ctrl = jnp.array([self.ctrl_space[0, 0], 0.])

    @partial(jax.jit, static_argnames='self')
    def apply_rear_offset_correction(self, state: DeviceArray):
        """
        Correct for moving from the rear wheel to centroid.

        Args:
            state (DeviceArray): [x, y, v, psi].
        Returns:
            state_offset 
        """
        state_offset = state.at[0].set(state[0] + self.rear_wheel_offset*jnp.cos(state[3]))
        state_offset = state_offset.at[1].set(state_offset[1] + self.rear_wheel_offset*jnp.sin(state[3]))

        return state_offset

    @partial(jax.jit, static_argnames='self')
    def revert_rear_offset_correction(self, state: DeviceArray):
        """
        Correct for moving from the centroid to rear wheel.

        Args:
            state (DeviceArray): [x, y, v, psi].
        Returns:
            state_offset 
        """
        state_offset = state.at[0].set(state[0] - self.rear_wheel_offset*jnp.cos(state[3]))
        state_offset = state_offset.at[1].set(state_offset[1] - self.rear_wheel_offset*jnp.sin(state[3]))

        return state_offset

    @partial(jax.jit, static_argnames=['self'])
    def get_batched_rear_offset_correction(self, nominal_states):
        jac = jax.jit(jax.vmap(self.apply_rear_offset_correction, in_axes=(1), out_axes=(1)))
        return jac(nominal_states)
        
    @partial(jax.jit, static_argnames=['self'])
    def get_batched_reverse_rear_offset_correction(self, nominal_states):
        jac = jax.jit(jax.vmap(self.revert_rear_offset_correction, in_axes=(1), out_axes=(1)))
        return jac(nominal_states)

    @partial(jax.jit, static_argnames='self')
    def check_stopped(
        self, state: DeviceArray
    ):
        return (state[2]>self.v_min)

    @partial(jax.jit, static_argnames='self')
    def get_stopping_ctrl(
        self, state: DeviceArray
    ):
        stopping_ctrl = jnp.array(self.stopping_ctrl)
        return stopping_ctrl

    @partial(jax.jit, static_argnames='self')
    def compute_stopping_path(self, state):
        # Choose stopping control as braking control with zero steering.
        stopping_ctrl = jnp.zeros((2,))
        stopping_ctrl = stopping_ctrl.at[0].set(self.ctrl_space[0, 0])

        # CAUTION: Assume a upper limit on maximum number of steps to stop to JIT this function.
        max_num_steps_to_stop = 280
        stopping_states = jnp.zeros((self.dim_x, max_num_steps_to_stop))
        dt_steps_to_stop = jnp.arange(0, max_num_steps_to_stop)*self.dt

        vel_to_stop = jnp.maximum(state[2] + stopping_ctrl[0]*dt_steps_to_stop, 0.0)
        vel_not_stopped = (vel_to_stop>0)
        time_to_stop = jnp.maximum(-state[2]/stopping_ctrl[0], 0.0)
        disp_to_stop = state[2]*dt_steps_to_stop + 0.5*stopping_ctrl[0]*dt_steps_to_stop**2
        stopping_distance = state[2]*time_to_stop + 0.5*stopping_ctrl[0]*time_to_stop**2
        disp_to_stop = vel_not_stopped*disp_to_stop + (1 - vel_not_stopped)*stopping_distance

        theta_to_stop = state[3]*jnp.ones((max_num_steps_to_stop,))

        stopping_states = stopping_states.at[0, :].set(state[0] + disp_to_stop*jnp.cos(theta_to_stop))
        stopping_states = stopping_states.at[1, :].set(state[1] + disp_to_stop*jnp.sin(theta_to_stop))
        stopping_states = stopping_states.at[2, :].set(vel_to_stop)
        stopping_states = stopping_states.at[3, :].set(theta_to_stop)

        stopping_ctrls = jnp.repeat(stopping_ctrl[:, jnp.newaxis], max_num_steps_to_stop, axis=1)

        return stopping_states, stopping_ctrls

    @partial(jax.jit, static_argnames='self')
    def integrate_forward_jax(
        self, state: DeviceArray, control: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:
        """Clips the control and computes one-step time evolution of the system.
        Args:
            state (DeviceArray): [x, y, v, psi].
            control (DeviceArray): [accel, omega].
        Returns:
            DeviceArray: next state.
            DeviceArray: clipped control.
        """
        # Clips the controller values between min and max accel and steer
        # values.
        ctrl_clip = jnp.clip(
            control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])

        state_nxt = self._integrate_forward(state, ctrl_clip, add_disturbance=False, key=jax.random.PRNGKey(43))

        return state_nxt, ctrl_clip

    @partial(jax.jit, static_argnames='self')
    def integrate_forward_jax_with_noise(
        self, state: DeviceArray, control: DeviceArray, seed: int
    ) -> Tuple[DeviceArray, DeviceArray]:
        """Clips the control and computes one-step time evolution of the system.
        Args:
            state (DeviceArray): [x, y, v, psi].
            control (DeviceArray): [accel, omega].
        Returns:
            DeviceArray: next state.
            DeviceArray: clipped control.
        """
        # Clips the controller values between min and max accel and steer
        # values.
        ctrl_clip = jnp.clip(
            control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])

        state_nxt = self._integrate_forward(state, ctrl_clip, add_disturbance=True, key=jax.random.PRNGKey(seed))

        return state_nxt, ctrl_clip


    @partial(jax.jit, static_argnames='self')
    def disc_deriv(
        self, state: DeviceArray, control: DeviceArray, add_disturbance, key
    ) -> DeviceArray:
        @jax.jit
        def true_fn(args):
            deriv_out = args[0]
            noise = jax.random.uniform(key, shape=(self.dim_x, ))
            noise = noise * self.noise_var
            return deriv_out + noise, noise

        @jax.jit
        def false_fn(args):
            return args

        deriv = jnp.zeros((self.dim_x,))
        deriv = deriv.at[0].set(state[2] * jnp.cos(state[3]))
        deriv = deriv.at[1].set(state[2] * jnp.sin(state[3]))
        deriv = deriv.at[2].set(control[0])
        deriv = deriv.at[3].set(state[2] * jnp.tan(control[1]) / self.wheelbase)
        deriv_out, noise = jax.lax.cond(add_disturbance, true_fn, false_fn, (deriv, jnp.zeros(self.dim_x)))
        return deriv_out

    @partial(jax.jit, static_argnames='self')
    def _integrate_forward(
        self, state: DeviceArray, control: DeviceArray, add_disturbance: bool = False, key: DeviceArray = jax.random.PRNGKey(43),
    ) -> DeviceArray:
        """ Computes one-step time evolution of the system: x_+ = f(x, u).
        The discrete-time dynamics is as below:
            x_k+1 = x_k + v_k cos(psi_k) dt
            y_k+1 = y_k + v_k sin(psi_k) dt
            v_k+1 = v_k + u0_k dt
            psi_k+1 = psi_k + v_k tan(u1_k) / L dt
        Args:
            state (DeviceArray): [x, y, v, psi, delta].
            control (DeviceArray): [accel, omega].
        Returns:
            DeviceArray: next state.
        """
        return self._integrate_forward_dt(state, control, self.dt, add_disturbance, key)

    @partial(jax.jit, static_argnames='self')
    def _integrate_forward_dt(
        self, state: DeviceArray, ctrl_clip: DeviceArray, dt: float, add_disturbance: bool = False, key=jax.random.PRNGKey(43),
    ) -> DeviceArray:
        k1 = self.disc_deriv(state, ctrl_clip, add_disturbance, key)
        k2 = self.disc_deriv(state + k1 * dt / 2, ctrl_clip, add_disturbance, key)
        k3 = self.disc_deriv(state + k2 * dt / 2, ctrl_clip, add_disturbance, key)
        k4 = self.disc_deriv(state + k3 * dt, ctrl_clip, add_disturbance, key)

        state_nxt = state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        # state_nxt = state_nxt.at[2].set(
        #     jnp.clip(state_nxt[2], self.v_min, self.v_max)
        # )

        return state_nxt

    @partial(jax.jit, static_argnames='self')
    def get_jacobian_fx(
        self, obs: DeviceArray, control: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:
        Ac = jnp.array([[0, 0, jnp.cos(obs[3]), -obs[2] * jnp.sin(obs[3])],
                        [0, 0, jnp.sin(obs[3]), obs[2] * jnp.cos(obs[3])],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        Ad = jnp.eye(self.dim_x) + Ac * self.dt + \
            0.5 * Ac @ Ac * self.dt * self.dt

        return Ad

    @partial(jax.jit, static_argnames='self')
    def get_jacobian_fu(
        self, obs: DeviceArray, control: DeviceArray
    ) -> DeviceArray:
        Bc = jnp.array([[0, 0],
                       [0, 0],
                       [1, 0],
                       [0, obs[2] / (1e-6 + self.wheelbase * jnp.cos(control[1])**2)]])

        Bd = self.dt * Bc

        return Bd

    @partial(jax.jit, static_argnames='self')
    def get_jacobian(
        self, nominal_states: DeviceArray, nominal_controls: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:
        jac = jax.jit(
            jax.vmap(
                self.get_jacobian_fx_fu, in_axes=(
                    1, 1), out_axes=(
                    2, 2)))
        return jac(nominal_states, nominal_controls)

    @partial(jax.jit, static_argnames='self')
    def get_jacobian_fx_fu(self, obs: DeviceArray,
                           control: DeviceArray) -> Tuple:
        Ac = jnp.array([[0, 0, jnp.cos(obs[3]), -1 * obs[2] * jnp.sin(obs[3])],
                        [0, 0, jnp.sin(obs[3]), obs[2] * jnp.cos(obs[3])],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        Bc = jnp.array([[0, 0],
                       [0, 0],
                       [1, 0],
                       [0, obs[2] / (1e-6 + self.wheelbase * jnp.cos(control[1])**2)]])

        Ad = jnp.eye(self.dim_x) + Ac * self.dt + \
            0.5 * Ac @ Ac * self.dt * self.dt
        Bd = self.dt * Bc

        return Ad, Bd
