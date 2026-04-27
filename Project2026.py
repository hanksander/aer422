import numpy as np
import matplotlib.pyplot as plt
import control as ct


# ============================================================
# GLOBAL MODEL DATA
# ============================================================

m = 250000.0       # kg
U0 = 65.1          # m/s, trim speed approximation

A4 = np.array([
    [-0.021,  0.122,   0.0,  -9.81],
    [-0.2,   -0.512,  65.1,   0.0 ],
    [ 0.0,   -0.006, -0.402,  0.0 ],
    [ 0.0,    0.0,    1.0,    0.0 ]
])

B4 = np.array([
    [ 0.292],
    [ -1.96 ],
    [-0.4  ],
    [ 0.0  ]
])


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def gm_db_text(gm):
    if np.isfinite(gm) and gm > 0:
        return f"{20*np.log10(gm):.2f} dB"
    return "inf/undefined"


def print_margins(name, L):
    gm, pm, wcg, wcp = ct.margin(L)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"GM  = {gm_db_text(gm)}")
    print(f"PM  = {pm:.2f} deg")
    print(f"Wcg = {wcg}")
    print(f"Wcp = {wcp}")
    return gm, pm, wcg, wcp


def plot_poles(poles, title):
    plt.figure()
    plt.plot(np.real(poles), np.imag(poles), "x")
    plt.axvline(0, color="k", linestyle="--")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title(title)
    plt.grid(True)
    plt.show()


# ============================================================
# TASK 1: AIRCRAFT + ACTUATOR + PITCH SAS + PITCH PI
# ============================================================

def make_A5():
    """
    5-state aircraft + elevator actuator model.
    States: [u, w, q, theta, deltaE]
    Input: delta_a
    """
    A5 = np.zeros((5, 5))
    B5 = np.zeros((5, 1))

    A5[0:4, 0:4] = A4
    A5[0:4, 4:5] = B4

    A5[4, 4] = -10.0
    B5[4, 0] = 10.0

    return A5, B5


def make_pitch_sas(Kq):
    """
    Adds pitch-rate SAS:
        delta_a = -Kq*q + delta_c
    """
    A5, B5 = make_A5()
    A = A5.copy()

    A[4, 2] += -10.0 * Kq

    Ctheta = np.array([[0, 0, 0, 1, 0]])
    Cq = np.array([[0, 0, 1, 0, 0]])

    sys_theta = ct.ss(A, B5, Ctheta, [[0]])
    sys_q = ct.ss(A, B5, Cq, [[0]])

    return A, B5, sys_theta, sys_q


def plot_Kq_poles(Kq_values):
    plt.figure()

    for Kq in Kq_values:
        A, _, _, _ = make_pitch_sas(Kq)
        poles = np.linalg.eigvals(A)
        plt.plot(np.real(poles), np.imag(poles), "x")

    plt.axvline(0, color="k", linestyle="--")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title("Task 1: Pitch SAS Pole Movement vs Kq")
    plt.grid(True)
    plt.show()

def sweep_Kq_numeric(Kq_values):
    print("Kq        poles")
    print("-------------------------------")

    good = []

    for Kq in Kq_values:
        A_sas, _, _, _ = make_pitch_sas(Kq)
        poles = np.linalg.eigvals(A_sas)

        max_real = np.max(np.real(poles))

        # Sort poles by real part
        poles_sorted = sorted(poles, key=lambda p: np.real(p), reverse=True)

        print(f"\nKq = {Kq:.3f}, max real = {max_real:.4f}")
        for p in poles_sorted:
            print(f"   {p:.4f}")

        if max_real < 0:
            good.append((Kq, poles))

    return good

def make_pitch_PI_loop(Kq, Ktheta, a_theta):
    """
    Open-loop pitch PI:
        PI = Ktheta*(s+a)/s
        L = PI * theta/delta_c
    """
    _, _, sys_theta, _ = make_pitch_sas(Kq)
    s = ct.TransferFunction.s
    PI = Ktheta * (s + a_theta) / s
    L = PI * sys_theta
    return L


def plot_pitch_PI_design(Kq, Ktheta, a_theta):
    L = make_pitch_PI_loop(Kq, Ktheta, a_theta)

    omega = np.logspace(-3, 2, 1000)

    plt.figure()
    ct.bode_plot(L, omega=omega, dB=True, display_margins = True)
    plt.suptitle("Task 1: Pitch PI Open-Loop Bode")
    plt.show()
    """
    plt.figure()
    ct.bode_plot(L, display_margins = True)
    plt.title("Task 1: Pitch PI Margins")
    plt.show()
    """
    print_margins("Task 1 Pitch PI Margins", L)

    T = ct.feedback(L, 1)

    t = np.linspace(0, 80, 1500)
    t, y = ct.step_response(T, t)

    plt.figure()
    plt.plot(t, y)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\theta/\theta_c$")
    plt.title("Closed-Loop Pitch Attitude Step Response")
    plt.grid(True)
    plt.show()

    ct.root_locus_plot(L,grid = False)
    plt.title('PI Root Locus Plot')
    plt.show()

    print("\nTask 1 closed-loop pitch poles:")
    print(ct.poles(T))

def find_pitch_PI_candidates(Kq):
    s = ct.TransferFunction.s
    _, _, sys_theta, _ = make_pitch_sas(Kq)

    Ktheta_values = -np.linspace(0.1, 20.0, 400)
    a_values = np.logspace(-4, 0, 80)  # 0.001 to 1.0

    candidates = []

    print("Ktheta        a_theta      GM[dB]       PM[deg]      Wcp")
    print("----------------------------------------------------------")

    for a_theta in a_values:
        for Ktheta in Ktheta_values:
            PI = Ktheta * (s + a_theta) / s
            L = PI * sys_theta

            gm, pm, wcg, wcp = ct.margin(L)

            gm_db = 20*np.log10(gm) if np.isfinite(gm) and gm > 0 else np.inf

            if np.isfinite(wcp):
                # loose filter first
                if pm > 60 and 0.9 <= wcp <= 1.2:
                    candidates.append((Ktheta, a_theta, gm_db, pm, wcp))

    # sort by closeness to desired values
    candidates.sort(key=lambda x: abs(x[4] - 1.0) + 0.03*abs(x[3] - 65))

    for Ktheta, a_theta, gm_db, pm, wcp in candidates[:25]:
        print(f"{Ktheta:10.4f}  {a_theta:10.5f}  {gm_db:10.2f}  {pm:10.2f}  {wcp:8.3f}")

    return candidates
# ============================================================
# AUTO-THROTTLE DESIGN
# ============================================================

def make_speed_loop_plant(Kq, Ktheta, a_theta):
    """
    Pitch autopilot closed, thrust loop open.
    Input: deltaT [N]
    Output: u [m/s]
    States: [u, w, q, theta, deltaE, ztheta]
    """
    A = np.zeros((6, 6))
    B = np.zeros((6, 1))

    A[0:4, 0:4] = A4
    A[0:4, 4:5] = B4

    # thrust acceleration into u_dot
    B[0, 0] = 1.0 / m

    # elevator + SAS + pitch PI, theta_c = 0 for speed loop
    A[4, 2] += -10.0 * Kq
    A[4, 3] += -10.0 * Ktheta
    A[4, 4] += -10.0
    A[4, 5] += 10.0 * Ktheta * a_theta

    # ztheta_dot = theta_c - theta
    A[5, 3] = -1.0

    C = np.array([[1, 0, 0, 0, 0, 0]])
    D = np.array([[0]])

    return ct.ss(A, B, C, D)


def sweep_Ku(Kq, Ktheta, a_theta, Ku_values_kN):
    """
    Ku units: kN/(m/s)
    Physical control law:
        deltaT = -Ku*u
    """
    P = make_speed_loop_plant(Kq, Ktheta, a_theta)

    print("\nAuto-throttle Ku sweep")
    print("Ku[kN/(m/s)]       GM          PM[deg]       Wcp[rad/s]")
    print("--------------------------------------------------------")

    good = []

    for Ku_kN in Ku_values_kN:
        Ku_N = Ku_kN * 1000.0
        L = Ku_N * P

        gm, pm, wcg, wcp = ct.margin(L)

        print(f"{Ku_kN:12.4f}   {gm_db_text(gm):>12}   {pm:10.2f}   {wcp:12.4f}")

        if pm > 90 and np.isfinite(wcp) and 0.5 <= wcp <= 0.6:
            good.append((Ku_kN, pm, wcp, gm))

    print("\nGood Ku candidates:")
    for Ku_kN, pm, wcp, gm in good:
        print(f"Ku={Ku_kN:.4f} kN/(m/s), PM={pm:.2f}, Wcp={wcp:.4f}, GM={gm_db_text(gm)}")

    return good


def plot_auto_throttle_design(Kq, Ktheta, a_theta, Ku_kN):
    P = make_speed_loop_plant(Kq, Ktheta, a_theta)
    L = Ku_kN * 1000.0 * P

    omega = np.logspace(-3, 2, 1000)

    plt.figure()
    ct.bode_plot(L, omega=omega, dB=True, display_margins = True)
    plt.suptitle(f"Auto-throttle Open-Loop Bode, Ku={Ku_kN:g} kN/(m/s)")
    plt.show()

    plt.figure()
    ct.root_locus_plot(L,grid = False)
    # Display the plot
    plt.title('Auto Throttle Root Locus Plot')
    plt.show()

    print_margins("Auto-throttle Margins", L)


# ============================================================
# TASK 2: FULL 9-STATE MODEL
# ============================================================

def coupler_output_coeffs(KE, sign=-1.0, z_pi=0.3, z_lead=0.06, p_lead=0.6):
    """
    Coupler:
        theta_c/E = sign*KE*(s+z_pi)/s * (s+z_lead)/(s+p_lead)

    Default from notes:
        sign*KE*(s+0.3)(s+0.06)/(s(s+0.6))

    State realization:
        z1_dot = z2
        z2_dot = -p_lead*z2 + E
        theta_c = sign*KE*(E + (z_pi+z_lead-p_lead)*z2
                            + z_pi*z_lead*z1)

    For default values:
        theta_c = sign*KE*(E -0.24*z2 +0.018*z1)
    """
    c_z1 = sign * KE * (z_pi * z_lead)
    c_z2 = sign * KE * (z_pi + z_lead - p_lead)
    c_E = sign * KE
    return c_z1, c_z2, c_E


def make_full_sys(
    Kq,
    Ktheta,
    a_theta,
    Ku_kN,
    KE,
    R,
    sign=-1.0,
    z_pi=0.3,
    z_lead=0.06,
    p_lead=0.6
):
    """
    Full 9-state model.

    State order:
        x = [u, w, q, theta, deltaE, ztheta, zE1, zE2, d]^T

    Input:
        Ec

    Outputs:
        [u, d, E, alpha, theta, gamma, deltaE]^T
    """
    A = np.zeros((9, 9))
    B = np.zeros((9, 1))

    # aircraft
    A[0:4, 0:4] = A4
    A[0:4, 4:5] = B4

    # auto-throttle
    Ku_N = Ku_kN * 1000.0
    A[0, 0] += -Ku_N / m

    # E = Ec - d/R for coupler input
    # For state feedback terms, Ec = 0 gives E = -d/R.
    # This sign is often the source of trouble. If unstable, try E = +d/R.
    E_from_d = -1.0 / R
    E_from_Ec = 1.0

    c_z1, c_z2, c_E = coupler_output_coeffs(
        KE, sign=sign, z_pi=z_pi, z_lead=z_lead, p_lead=p_lead
    )

    # theta_c = c_z1*zE1 + c_z2*zE2 + c_E*E
    theta_c_from_z1 = c_z1
    theta_c_from_z2 = c_z2
    theta_c_from_d = c_E * E_from_d
    theta_c_from_Ec = c_E * E_from_Ec

    # actuator:
    # deltaE_dot = -10 deltaE + 10[-Kq*q + Ktheta(theta_c-theta) + Ktheta*a*ztheta]
    A[4, 2] += -10.0 * Kq
    A[4, 3] += -10.0 * Ktheta
    A[4, 4] += -10.0
    A[4, 5] += 10.0 * Ktheta * a_theta
    A[4, 6] += 10.0 * Ktheta * theta_c_from_z1
    A[4, 7] += 10.0 * Ktheta * theta_c_from_z2
    A[4, 8] += 10.0 * Ktheta * theta_c_from_d
    B[4, 0] += 10.0 * Ktheta * theta_c_from_Ec

    # pitch PI integrator:
    # ztheta_dot = theta_c - theta
    A[5, 3] += -1.0
    A[5, 6] += theta_c_from_z1
    A[5, 7] += theta_c_from_z2
    A[5, 8] += theta_c_from_d
    B[5, 0] += theta_c_from_Ec

    # coupler states:
    # zE1_dot = zE2
    # zE2_dot = -p_lead*zE2 + E
    A[6, 7] = 1.0
    A[7, 7] = -p_lead
    A[7, 8] = E_from_d
    B[7, 0] = E_from_Ec

    # glide-slope displacement:
    # d_dot = U0*gamma = U0*theta - w
    A[8, 1] = -1.0
    A[8, 3] = U0

    # outputs [u, d, E, alpha, theta, gamma, deltaE]
    C = np.zeros((7, 9))
    D = np.zeros((7, 1))

    C[0, 0] = 1.0              # u
    C[1, 8] = 1.0              # d
    C[2, 8] = -1.0 / R         # E = Ec - d/R, D adds Ec
    D[2, 0] = 1.0
    C[3, 1] = 1.0 / U0         # alpha = w/U0
    C[4, 3] = 1.0              # theta
    C[5, 3] = 1.0              # gamma = theta - alpha
    C[5, 1] = -1.0 / U0
    C[6, 4] = 1.0              # deltaE

    sys = ct.ss(A, B, C, D)
    return sys, A, B, C, D


# ============================================================
# TASK 3: COUPLER DESIGN
# ============================================================

def coupler_tf(KE, sign=-1.0, z_pi=0.3, z_lead=0.06, p_lead=0.6):
    s = ct.TransferFunction.s
    PI = (s + z_pi) / s
    Lead = (s + z_lead) / (s + p_lead)
    return sign * KE * PI * Lead


def make_coupler_design_plant(Kq, Ktheta, a_theta, Ku_kN, R):
    """
    Plant for coupler tuning.
    Input: theta_c
    Output: E = -d/R
    """
    A = np.zeros((7, 7))
    B = np.zeros((7, 1))

    # states [u, w, q, theta, deltaE, ztheta, d]
    A[0:4, 0:4] = A4
    A[0:4, 4:5] = B4

    # auto-throttle
    Ku_N = Ku_kN * 1000.0
    A[0, 0] += -Ku_N / m

    # actuator + SAS + pitch PI
    A[4, 2] += -10.0 * Kq
    A[4, 3] += -10.0 * Ktheta
    A[4, 4] += -10.0
    A[4, 5] += 10.0 * Ktheta * a_theta

    # theta_c input
    B[4, 0] = 10.0 * Ktheta

    # ztheta_dot = theta_c - theta
    A[5, 3] = -1.0
    B[5, 0] = 1.0

    # d_dot = U0*theta - w
    A[6, 1] = -1.0
    A[6, 3] = U0

    C = np.zeros((1, 7))
    C[0, 6] = -1.0 / R

    D = np.zeros((1, 1))

    return ct.ss(A, B, C, D)


def sweep_coupler(
    Kq,
    Ktheta,
    a_theta,
    Ku_kN,
    R,
    KE_values,
    sign_values=(-1.0, 1.0),
    z_pi_values=(0.3,),
    z_lead_values=(0.06,),
    p_lead_values=(0.6,)
):
    P = make_coupler_design_plant(Kq, Ktheta, a_theta, Ku_kN, R)

    candidates = []

    print("\nCoupler sweep")
    print("sign     KE        zpi     zlead    plead       GM          PM       Wcp")
    print("------------------------------------------------------------------------")

    for sign in sign_values:
        for z_pi in z_pi_values:
            for z_lead in z_lead_values:
                for p_lead in p_lead_values:
                    for KE in KE_values:
                        Gc = coupler_tf(KE, sign, z_pi, z_lead, p_lead)
                        L = Gc * P

                        gm, pm, wcg, wcp = ct.margin(L)

                        if np.isfinite(wcp) and pm >= 60 and 0.1 <= wcp <= 0.2:
                            candidates.append((sign, KE, z_pi, z_lead, p_lead, gm, pm, wcp))

                        if np.isfinite(wcp) and 0.08 <= wcp <= 0.25:
                            print(
                                f"{sign:4.0f}  {KE:9.5g}  {z_pi:7.3f}  {z_lead:7.3f}  "
                                f"{p_lead:7.3f}  {gm_db_text(gm):>12}  {pm:8.2f}  {wcp:8.4f}"
                            )

    print("\nGood coupler candidates:")
    for cand in candidates[:20]:
        sign, KE, z_pi, z_lead, p_lead, gm, pm, wcp = cand
        print(
            f"sign={sign:+.0f}, KE={KE:.6g}, zpi={z_pi}, zlead={z_lead}, "
            f"plead={p_lead}, GM={gm_db_text(gm)}, PM={pm:.2f}, Wcp={wcp:.4f}"
        )

    return candidates


def plot_coupler_design(Kq, Ktheta, a_theta, Ku_kN, KE, R, sign=-1.0,
                        z_pi=0.3, z_lead=0.06, p_lead=0.6):
    P = make_coupler_design_plant(Kq, Ktheta, a_theta, Ku_kN, R)
    Gc = coupler_tf(KE, sign, z_pi, z_lead, p_lead)
    L = Gc * P

    omega = np.logspace(-3, 1, 1000)

    plt.figure()
    ct.bode_plot(L, omega=omega, dB=True, display_margins = True)
    plt.suptitle("Task 3: Coupler Open-Loop Bode")
    plt.show()

    plt.figure()
    ct.root_locus_plot(L,grid = False)
    # Display the plot
    plt.title('Coupler (Lead + PI) Root Locus Plot')
    plt.show()
    print_margins("Task 3 Coupler Margins", L)


# ============================================================
# TASKS 3 AND 4: TIME RESPONSES
# ============================================================

def simulate_full_response(Kq, Ktheta, a_theta, Ku_kN, KE, R, sign=-1.0,
                           z_pi=0.3, z_lead=0.06, p_lead=0.6,
                           d0=40.0, tfinal=100.0, title_suffix=""):
    sys, A, B, C, D = make_full_sys(
        Kq, Ktheta, a_theta, Ku_kN, KE, R,
        sign=sign, z_pi=z_pi, z_lead=z_lead, p_lead=p_lead
    )

    t = np.linspace(0, tfinal, 2500)
    U = np.zeros_like(t)

    x0 = np.zeros(9)

    # Initial aircraft states:
    x0[0] = 0.0              # u(0)
    x0[1] = 0.0              # w(0), alpha(0)=w/U0=0
    x0[2] = 0.0              # q(0)
    x0[3] = np.deg2rad(0.0)  # theta(0)
    x0[4] = 0.0              # deltaE(0)
    x0[5] = 0.0              # ztheta(0)
    x0[6] = 0.0              # zE1(0)
    x0[7] = 0.0              # zE2(0)
    x0[8] = d0               # glide-slope position error

    t, y = ct.forced_response(sys, T=t, U=U, X0=x0)

    labels = [
        "u [m/s]",
        "d [m]",
        "E [rad]",
        "alpha [rad]",
        "theta [rad]",
        "gamma [rad]",
        "deltaE [rad]"
    ]

    def plot_full_response_stacked(t, y, labels, title="Full System Response"):
        n = len(labels)

        fig, axes = plt.subplots(n, 1, figsize=(10, 10), sharex=True)

        for i in range(n):
            axes[i].plot(t, y[i, :], linewidth=1)
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True)

        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(title)

        plt.tight_layout()
        plt.show()

    plot_full_response_stacked(
        t, y, labels,
        title=f"{title_suffix}, d0={d0:+.0f} m"
    )

    poles = np.linalg.eigvals(A)

    max_delta_rad = np.max(np.abs(y[6, :]))
    max_delta_deg = np.degrees(max_delta_rad)

    print(f"\nFull-system poles {title_suffix}, d0={d0:+.0f} m:")
    print(poles)

    print(f"\nMax elevator deflection {title_suffix}, d0={d0:+.0f} m:")
    print(f"{max_delta_rad:.5f} rad = {max_delta_deg:.2f} deg")

    if max_delta_deg <= 20:
        print("Elevator constraint PASSED.")
    else:
        print("Elevator constraint FAILED.")

    return t, y, poles


def compare_ranges(Kq, Ktheta, a_theta, Ku_kN,
                   KE_4000, KE_200,
                   sign_4000=-1.0, sign_200=-1.0,
                   z_pi=0.3, z_lead=0.06, p_lead=0.6,
                   d0=40.0):
    t = np.linspace(0, 100, 2500)
    U = np.zeros_like(t)

    plt.figure()

    for R, KE, sign in [(4000.0, KE_4000, sign_4000), (200.0, KE_200, sign_200)]:
        sys, A, B, C, D = make_full_sys(
            Kq, Ktheta, a_theta, Ku_kN, KE, R,
            sign=sign, z_pi=z_pi, z_lead=z_lead, p_lead=p_lead
        )

        x0 = np.zeros(9)
        x0[8] = d0

        t_out, y = ct.forced_response(sys, T=t, U=U, X0=x0)
        plt.plot(t_out, y[1, :], label=f"R={R:g} m, KE={KE:g}")

    plt.xlabel("Time [s]")
    plt.ylabel("d [m]")
    plt.title("Range Scheduling Comparison")
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# MAIN EXECUTION: TASKS 1-4
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # SELECTED GAINS FROM YOUR WORK
    # --------------------------------------------------------
    Kq = -2.9
    Ktheta = -4.5887
    a_theta = 0.0169

    # Replace this after Ku sweep.
    Ku_kN = 141.7059
    #Ku_kN = 129.1429
    #Ku_kN = 119.0924

    # Replace these after coupler sweep.
    KE_4000 = 16.5
    KE_200 = 0.84

    coupler_sign = -1.0
    z_pi = 13
    z_lead = 0.01
    p_lead = 25

    """
    Add the Lead of the Coupler to target a maximum phase lead around 0.15 r/s. Add the PI to
    achieve crossover at 0.1-0.2 r/s, PM>= 60, and GM=20 dB. See that the bandwidth
    (crossover frequency) is decreasing as the design progresses.
    ↑ z_pi → stronger low-frequency correction
    
       → faster elimination of d(t)
       → larger elevator spike

    ↓ z_pi → smoother, less aggressive response
       → slower settling
    
    Increase p_lead / z_lead ratio →
    more phase margin
    more stability
    allows larger KE
    """

    sys9, A9, B9, C9, D9 = make_full_sys(
        Kq, Ktheta, a_theta, Ku_kN,
        KE_4000, 4000.0,
        sign=-coupler_sign,
        z_pi=z_pi,
        z_lead=z_lead,
        p_lead=p_lead
    )

    poles = np.linalg.eigvals(A9)
    print(poles)
    # --------------------------------------------------------
    # TASK 1
    # --------------------------------------------------------
    print("\n======================")
    print("TASK 1: PITCH CONTROL")
    print("======================")

    #plot_Kq_poles([ -2.9])
    #plot_pitch_PI_design(Kq, Ktheta, a_theta)
    #candidates = find_pitch_PI_candidates(Kq)



    # --------------------------------------------------------
    # AUTO-THROTTLE
    # --------------------------------------------------------

    print("\n======================")
    print("AUTO-THROTTLE TUNING")
    print("======================")
    
    #Ku_values = np.linspace(1, 400, 200)
    #good_Ku = sweep_Ku(Kq, Ktheta, a_theta, Ku_values)
    

    #plot_auto_throttle_design(Kq, Ktheta, a_theta, Ku_kN)
    """
    # --------------------------------------------------------
    # TASK 2: FULL MODEL MATRICES
    # --------------------------------------------------------
    print("\n======================")
    print("TASK 2: FULL 9-STATE MODEL")
    print("======================")

    sys9, A9, B9, C9, D9 = make_full_sys(
        Kq, Ktheta, a_theta, Ku_kN, KE_4000, 4000.0,
        sign=coupler_sign, z_pi=z_pi, z_lead=z_lead, p_lead=p_lead
    )

    print("\nState order:")
    print("[u, w, q, theta, deltaE, ztheta, zE1, zE2, d]^T")

    print("\nInput:")
    print("Ec")

    print("\nOutputs:")
    print("[u, d, E, alpha, theta, gamma, deltaE]^T")

    print("\nA9 =")
    print(A9)

    print("\nB9 =")
    print(B9)

    print("\nC9 =")
    print(C9)

    print("\nD9 =")
    print(D9)
    """
    # --------------------------------------------------------
    # TASK 3: COUPLER DESIGN AT R = 4000 m
    # --------------------------------------------------------
    print("\n======================")
    print("TASK 3: COUPLER DESIGN AT R = 4000 m")
    print("======================")

    #KE_values = np.logspace(-2, 2, 180)
    """
    candidates_4000 = sweep_coupler(
        Kq, Ktheta, a_theta, Ku_kN,
        R=4000.0,
        KE_values=KE_values,
        sign_values=([-1.0]),
        z_pi_values=([13,14,15,16,17,18,19.20]),
        z_lead_values=([0.01, 0.02]),
        p_lead_values=([25, 30,35,40,45,50])
    )
    
    
    plot_coupler_design(
        Kq, Ktheta, a_theta, Ku_kN,
        KE=KE_4000,
        R=4000.0,
        sign=coupler_sign,
        z_pi=z_pi,
        z_lead=z_lead,
        p_lead=p_lead
    )

    """
    simulate_full_response(
        Kq, Ktheta, a_theta, Ku_kN,
        KE=KE_4000,
        R=4000.0,
        sign=-coupler_sign,
        z_pi=z_pi,
        z_lead=z_lead,
        p_lead=p_lead,
        d0=40.0,
        tfinal=70.0,
        title_suffix="R=4000 m"
    )
    



    
    # --------------------------------------------------------
    # TASK 4: RANGE = 200 m
    # --------------------------------------------------------
    print("\n======================")
    print("TASK 4: RANGE = 200 m")
    print("======================")
    """
    candidates_200 = sweep_coupler(
        Kq, Ktheta, a_theta, Ku_kN,
        R=200.0,
        KE_values=KE_values,
        sign_values=(-1.0, 1.0),
        z_pi_values=(0.1, 0.2, 0.3),
        z_lead_values=(0.03, 0.04, 0.06, 0.08),
        p_lead_values=(0.3, 0.4, 0.6, 0.8)
    )
    """
    """
    plot_coupler_design(
        Kq, Ktheta, a_theta, Ku_kN,
        KE=KE_200,
        R=200.0,
        sign=coupler_sign,
        z_pi=z_pi,
        z_lead=z_lead,
        p_lead=p_lead
    )
    """
    simulate_full_response(
        Kq, Ktheta, a_theta, Ku_kN,
        KE=KE_200,
        R=200.0,
        sign=-coupler_sign,
        z_pi=z_pi,
        z_lead=z_lead,
        p_lead=p_lead,
        d0=40.0,
        tfinal=70.0,
        title_suffix="R=200 m"
    )
    """
    compare_ranges(
        Kq, Ktheta, a_theta, Ku_kN,
        KE_4000=KE_4000,
        KE_200=KE_200,
        sign_4000=coupler_sign,
        sign_200=coupler_sign,
        z_pi=z_pi,
        z_lead=z_lead,
        p_lead=p_lead,
        d0=40.0
    )
    """
