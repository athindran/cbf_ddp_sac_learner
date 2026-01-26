from typing import Tuple, Any
import numpy as np
from functools import partial
from jax import Array as DeviceArray
import jax
from jax import numpy as jnp
from jax import custom_jvp
from jax import random

from .base_dynamics import BaseDynamics


class PointMass4D(BaseDynamics):

    def __init__(self, config: Any, action_space: np.ndarray) -> None:
        """
        Implements the 2D double integrator dynamics.
        Args:
            config (Any): an object specifies configuration.
            action_space (np.ndarray): action space.
        """
        super().__init__(config, action_space)
        self.dim_x = 4  # [x, y, vx, vy].

        # load parameters
        self.v_min = 0.0
        self.v_max = config.V_MAX
        self.noise_var = jnp.array([0.001, 0.001, 0.001, 0.001])
        self.stopping_ctrl = jnp.array([self.ctrl_space[0, 0], 0.0])

    @partial(jax.jit, static_argnames='self')
    def apply_rear_offset_correction(self, state: DeviceArray):
        return state

    @partial(jax.jit, static_argnames=['self'])
    def get_batched_rear_offset_correction(self, nominal_states):
        jac = jax.jit(jax.vmap(self.apply_rear_offset_correction, in_axes=(1), out_axes=(1)))
        return jac(nominal_states)

    @partial(jax.jit, static_argnames='self')
    def check_stopped(
        self, state: DeviceArray, stopping_ctrl: DeviceArray
    ):
        vx_has_crossed_zero = (state[2]*stopping_ctrl[0])>0
        vy_has_crossed_zero = (state[3]*stopping_ctrl[1])>0

        return ~(vx_has_crossed_zero & vy_has_crossed_zero)

    @partial(jax.jit, static_argnames='self')
    def compute_stopping_path(self, state):
        # Choose stopping control based on initial sign of vx and vy.
        # If vx>0, ax<0 and vice versa. 
        stopping_ctrl = jnp.zeros((2,))
        stopping_ctrl = stopping_ctrl.at[0].set(-jnp.sign(state[2])*self.ctrl_space[0, 1])
        stopping_ctrl = stopping_ctrl.at[1].set(-jnp.sign(state[3])*self.ctrl_space[1, 1])

        # CAUTION: Assume a upper limit on maximum number of steps to stop to JIT this function.
        max_num_steps_to_stop = 200
        stopping_states = jnp.zeros((self.dim_x, max_num_steps_to_stop,))
        dt_steps_to_stop = jnp.arange(0, max_num_steps_to_stop)*self.dt

        vx_to_stop = jnp.maximum((state[2] + stopping_ctrl[0]*dt_steps_to_stop)*jnp.sign(state[2]), 0.0)
        vx_to_stop = vx_to_stop*jnp.sign(state[2])
        vx_not_stopped = (vx_to_stop != 0.0)
        dx_to_stop = state[0] + state[2]*dt_steps_to_stop + 0.5*stopping_ctrl[0]*dt_steps_to_stop**2
        vx_to_stop = vx_to_stop*vx_not_stopped
        tx_stop = jnp.abs(state[2]/stopping_ctrl[0])
        dx_stopping_distance = state[0] + state[2]*tx_stop + 0.5*stopping_ctrl[0]*tx_stop**2
        dx_to_stop = vx_not_stopped*dx_to_stop + (1 - vx_not_stopped)*dx_stopping_distance

        vy_to_stop = jnp.maximum((state[3] + stopping_ctrl[1]*dt_steps_to_stop)*jnp.sign(state[3]), 0.0)
        vy_to_stop = vy_to_stop*jnp.sign(state[3])
        vy_not_stopped = (vy_to_stop != 0.0)
        dy_to_stop = state[1] + state[3]*dt_steps_to_stop + 0.5*stopping_ctrl[1]*dt_steps_to_stop**2
        vy_to_stop = vy_to_stop*vx_not_stopped
        ty_stop = jnp.abs(state[3]/stopping_ctrl[1])
        dy_stopping_distance = state[1] + state[3]*ty_stop + 0.5*stopping_ctrl[1]*ty_stop**2
        dy_to_stop = vy_not_stopped*dy_to_stop + (1 - vy_not_stopped)*dy_stopping_distance

        stopping_states = stopping_states.at[0, :].set(dx_to_stop)
        stopping_states = stopping_states.at[2, :].set(vx_to_stop)
        stopping_states = stopping_states.at[1, :].set(dy_to_stop)
        stopping_states = stopping_states.at[3, :].set(vy_to_stop)

        stopping_ctrls = jnp.repeat(stopping_ctrl[:, jnp.newaxis], max_num_steps_to_stop, axis=1)

        return stopping_states, stopping_ctrls

    @partial(jax.jit, static_argnames='self')
    def get_stopping_ctrl(
        self, state: DeviceArray, stopping_ctrl: DeviceArray
    ):  
        # Check whether previous stopping policy has caused the velocity to change sign
        vx_has_crossed_zero = (state[2]*stopping_ctrl[0])>0
        vy_has_crossed_zero = (state[3]*stopping_ctrl[1])>0
        # Derive new stopping policy
        stopping_ctrl = jnp.zeros((2,))
        stopping_ctrl = stopping_ctrl.at[0].set(-jnp.sign(state[2])*self.ctrl_space[0, 1])
        stopping_ctrl = stopping_ctrl.at[1].set(-jnp.sign(state[3])*self.ctrl_space[1, 1])
        # Zero out the corresponding acceleration if velocity has already crossed zero.
        stopping_ctrl = stopping_ctrl.at[0].set(stopping_ctrl[0]*(1 - vx_has_crossed_zero))
        stopping_ctrl = stopping_ctrl.at[1].set(stopping_ctrl[1]*(1 - vy_has_crossed_zero))

        return stopping_ctrl

    @partial(jax.jit, static_argnames='self')
    def integrate_forward_jax(
        self, state: DeviceArray, control: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:
        """Clips the control and computes one-step time evolution of the system.
        Args:
            state (DeviceArray): [x, y, vx, vy].
            control (DeviceArray): [accel_x, accel_y].
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
            state (DeviceArray): [x, y, vx, vy].
            control (DeviceArray): [ax, ay].
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
        self, state: DeviceArray, control: DeviceArray, add_disturbance: bool, key: DeviceArray
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
        deriv = deriv.at[0].set(state[2])
        deriv = deriv.at[1].set(state[3])
        deriv = deriv.at[2].set(control[0])
        deriv = deriv.at[3].set(control[1])
        deriv_out, noise = jax.lax.cond(add_disturbance, true_fn, false_fn, (deriv, jnp.zeros(self.dim_x)))
        return deriv_out

    @partial(jax.jit, static_argnames='self')
    def _integrate_forward(
        self, state: DeviceArray, control: DeviceArray, add_disturbance: bool = False, key: DeviceArray = jax.random.PRNGKey(43),
    ) -> DeviceArray:
        return self._integrate_forward_dt(state, control, self.dt, add_disturbance, key)

    @partial(jax.jit, static_argnames='self')
    def _integrate_forward_dt(
        self, state: DeviceArray, ctrl_clip: DeviceArray, dt: float, add_disturbance: bool, key: DeviceArray,
    ) -> DeviceArray:
        k1 = self.disc_deriv(state, ctrl_clip, add_disturbance, key)
        k2 = self.disc_deriv(state + k1 * dt / 2, ctrl_clip, add_disturbance, key)
        k3 = self.disc_deriv(state + k2 * dt / 2, ctrl_clip, add_disturbance, key)
        k4 = self.disc_deriv(state + k3 * dt, ctrl_clip, add_disturbance, key)

        state_nxt = state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

        return state_nxt

    @partial(jax.jit, static_argnames='self')
    def get_jacobian_fx(
        self, obs: DeviceArray, control: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:
        Ac = jnp.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
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
                       [0, 1]])

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
        Ac = jnp.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        Bc = jnp.array([[0, 0],
                       [0, 0],
                       [1, 0],
                       [0, 1]])

        Ad = jnp.eye(self.dim_x) + Ac * self.dt + \
            0.5 * Ac @ Ac * self.dt * self.dt
        Bd = self.dt * Bc

        return Ad, Bd
