"""General-purpose discrete PID controller.

Discretisation (sample time Ts):

* Integral term - Tustin / trapezoidal rule
      I[k] = I[k-1] + Ki * Ts/2 * (e[k] + e[k-1])
* Derivative term - backward Euler with a first-order low-pass filter
  (time constant Tf), i.e. the discretisation of Kd*s / (Tf*s + 1):
      D[k] = Tf/(Tf+Ts) * D[k-1] + Kd/(Tf+Ts) * (e[k] - e[k-1])
* Output: u[k] = sat(Kp*e[k] + I[k] + D[k])

Anti-windup uses conditional integration: the integrator is frozen while the
output is saturated and the error would push it further into saturation.
"""


class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, ts=0.032, tf=0.05,
                 output_limits=(float("-inf"), float("inf"))):
        if ts <= 0:
            raise ValueError("sample time must be positive")
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ts, self.tf = ts, tf
        self.u_min, self.u_max = output_limits
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_error = None

    def update(self, error):
        """Return the control output for the current error sample."""
        prev = error if self.prev_error is None else self.prev_error

        # Filtered derivative (backward Euler)
        a = self.tf / (self.tf + self.ts)
        self.derivative = a * self.derivative + self.kd / (self.tf + self.ts) * (error - prev)

        # Trapezoidal integral, computed tentatively for anti-windup
        integral = self.integral + self.ki * self.ts / 2.0 * (error + prev)

        u = self.kp * error + integral + self.derivative
        u_sat = min(max(u, self.u_min), self.u_max)

        saturated_high = u > self.u_max and error > 0
        saturated_low = u < self.u_min and error < 0
        if not (saturated_high or saturated_low):
            self.integral = integral

        self.prev_error = error
        return u_sat
