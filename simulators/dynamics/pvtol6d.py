from typing import Tuple, Any
import numpy as np
from functools import partial
from jax import Array as DeviceArray
import jax
from jax import numpy as jnp
from jax import custom_jvp
from jax import random

from .base_dynamics import BaseDynamics

class Pvtol6D(BaseDynamics):

    def __init__(self, config: Any, action_space: np.ndarray) -> None:
        """
        Implements the PVTOL dynamics.

        See https://murray.cds.caltech.edu/index.php/Python-control/Example:_Vertical_takeoff_and_landing_aircraft for reference.
        Args:
            config (Any): an object specifies configuration.
            action_space (np.ndarray): action space.
        """
        super().__init__(config, action_space)
        self.dim_x = 6  # [x, y, theta, xdot, ydot, thetadot].
        self.mass = 4.0
        self.sys_inertia = 0.0475
        self.thrust_offset = 0.25
        self.g = 9.8
        self.damping = 0.05
        self.noise_var = jnp.array([0.01, 0.01, 0.01, 0.01, 0.001, 0.001])

    @partial(jax.jit, static_argnames='self')
    def integrate_forward_jax_with_noise(
        self, state: DeviceArray, control: DeviceArray, seed: int
    ) -> Tuple[DeviceArray, DeviceArray]:
        """Clips the control and computes one-step time evolution of the system.
        Args:
            state (DeviceArray): [x, y, theta, xdot, ydot, thetadot].
            control (DeviceArray): [Fx, Fy].
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
    def integrate_forward_jax(
        self, state: DeviceArray, control: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:
        """Clips the control and computes one-step time evolution of the system.
        Args:
            state (DeviceArray): [x, y, v, psi, delta].
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
        deriv = deriv.at[0].set(state[3])
        deriv = deriv.at[1].set(state[4])
        deriv = deriv.at[2].set(state[5])
        deriv = deriv.at[3].set((-1*state[3]*self.damping/self.mass) + (control[0] * jnp.cos(state[2])/self.mass) - (control[1] * jnp.sin(state[2])/self.mass))
        deriv = deriv.at[4].set((-1*self.g - 1*state[4]*self.damping/self.mass) + (control[1] * jnp.cos(state[2])/self.mass) + (control[0] * jnp.sin(state[2])/self.mass))
        deriv = deriv.at[5].set(control[0] * self.thrust_offset/self.sys_inertia)
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
            psi_k+1 = psi_k + v_k tan(delta_k) / L dt
            delta_k+1 = delta_k + u1_k dt
        Args:
            state (DeviceArray): [x, y, v, psi, delta].
            control (DeviceArray): [accel, omega].
        Returns:
            DeviceArray: next state.
        """
        return self._integrate_forward_dt(state, control, self.dt, add_disturbance, key)

    @partial(jax.jit, static_argnames='self')
    def _integrate_forward_dt(
        self, state: DeviceArray, ctrl_clip: DeviceArray, dt: float, add_disturbance: bool, key: DeviceArray
    ) -> DeviceArray:
        k1 = self.disc_deriv(state, ctrl_clip, add_disturbance, key)
        k2 = self.disc_deriv(state + k1 * dt / 2, ctrl_clip, add_disturbance, key)
        k3 = self.disc_deriv(state + k2 * dt / 2, ctrl_clip, add_disturbance, key)
        k4 = self.disc_deriv(state + k3 * dt, ctrl_clip, add_disturbance, key)

        state_nxt = state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

        return state_nxt

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
        Ac = jnp.array([[0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0,
                         0,
                         (- control[0] * jnp.sin(obs[2]) - control[1] * jnp.cos(obs[2]))/self.mass,
                         -self.damping/self.mass,
                         0,
                         0],
                        [0, 
                         0, 
                         (control[0] * jnp.cos(obs[2]) - control[1] * jnp.sin(obs[2]))/self.mass,
                         0,
                         -self.damping/self.mass,
                         0],
                        [0, 0, 0, 0, 0, 0]])

        Bc = jnp.array([[0, 0],
                       [0, 0],
                       [0, 0],
                       [jnp.cos(obs[2])/self.mass, -jnp.sin(obs[2])/self.mass],
                       [jnp.sin(obs[2])/self.mass, jnp.cos(obs[2])/self.mass],
                       [self.thrust_offset/self.sys_inertia, 0]])

        Ad = jnp.eye(self.dim_x) + Ac * self.dt + \
            0.5 * Ac @ Ac * self.dt * self.dt
        Bd = self.dt * Bc

        return Ad, Bd