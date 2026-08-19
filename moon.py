import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS
# ============================================================

G = 1.0
M_EARTH = 600.0
EARTH_RADIUS = 0.7

N_PARTICLES = 300
M_PARTICLE = 0.2

DT = 0.01
STEPS = 1500

R_MIN = 1.2
R_MAX = 4.5

ORBITAL_VELOCITY_FACTOR = 0.90
VELOCITY_DISPERSION = 0.04

SOFTENING = 0.10

# Small effective collision radius
BASE_RADIUS = 0.015

ESCAPE_RADIUS = 7.0

rng = np.random.default_rng(42)


# ============================================================
# INITIAL CONDITIONS
# ============================================================

r = rng.uniform(R_MIN, R_MAX, N_PARTICLES)
theta = rng.uniform(0, 2*np.pi, N_PARTICLES)

positions = np.column_stack((
    r * np.cos(theta),
    r * np.sin(theta)
))

masses = np.full(N_PARTICLES, M_PARTICLE)

# Circular orbital velocity
v_circular = np.sqrt(G * M_EARTH / r)

vx = (
    -ORBITAL_VELOCITY_FACTOR
    * v_circular
    * np.sin(theta)
)

vy = (
    ORBITAL_VELOCITY_FACTOR
    * v_circular
    * np.cos(theta)
)

vx += rng.normal(0, VELOCITY_DISPERSION, N_PARTICLES)
vy += rng.normal(0, VELOCITY_DISPERSION, N_PARTICLES)

velocities = np.column_stack((vx, vy))


# ============================================================
# PARTICLE RADIUS
# ============================================================

def particle_radius(m):

    # Constant-density assumption:
    # R ∝ M^(1/3)

    return BASE_RADIUS * (m / M_PARTICLE)**(1/3)


# ============================================================
# ACCELERATION
# ============================================================

def calculate_acceleration(pos, mass):

    n = len(pos)

    acceleration = np.zeros_like(pos)

    # --------------------------------------------------------
    # Earth -> debris
    # --------------------------------------------------------

    r_vec = pos

    r2 = np.sum(
        r_vec**2,
        axis=1,
        keepdims=True
    )

    acceleration += (
        -G * M_EARTH * r_vec
        / (r2 + SOFTENING**2)**1.5
    )

    # --------------------------------------------------------
    # Debris -> debris
    # --------------------------------------------------------

    displacement = (
        pos[:, None, :]
        - pos[None, :, :]
    )

    distance2 = np.sum(
        displacement**2,
        axis=2
    )

    inv_distance3 = (
        distance2 + SOFTENING**2
    )**(-1.5)

    np.fill_diagonal(inv_distance3, 0)

    acceleration += (
        -G
        * np.sum(
            mass[None, :, None]
            * displacement
            * inv_distance3[:, :, None],
            axis=1
        )
    )

    return acceleration


# ============================================================
# COLLISION + ACCRETION
# ============================================================

def merge_collisions(pos, vel, mass):

    n = len(pos)

    if n == 0:
        return pos, vel, mass

    used = np.zeros(n, dtype=bool)

    new_pos = []
    new_vel = []
    new_mass = []

    for i in range(n):

        if used[i]:
            continue

        used[i] = True

        current_pos = pos[i].copy()
        current_vel = vel[i].copy()
        current_mass = mass[i]

        # Check collisions with the state at this timestep
        for j in range(i + 1, n):

            if used[j]:
                continue

            distance = np.linalg.norm(
                current_pos - pos[j]
            )

            collision_distance = (
                particle_radius(current_mass)
                + particle_radius(mass[j])
            )

            if distance < collision_distance:

                total_mass = (
                    current_mass + mass[j]
                )

                # Centre of mass
                current_pos = (
                    current_mass * current_pos
                    + mass[j] * pos[j]
                ) / total_mass

                # Momentum conservation
                current_vel = (
                    current_mass * current_vel
                    + mass[j] * vel[j]
                ) / total_mass

                current_mass = total_mass

                used[j] = True

        new_pos.append(current_pos)
        new_vel.append(current_vel)
        new_mass.append(current_mass)

    return (
        np.array(new_pos),
        np.array(new_vel),
        np.array(new_mass)
    )


# ============================================================
# REMOVE EARTH IMPACTS / ESCAPED PARTICLES
# ============================================================

crashed = 0
escaped = 0


def remove_lost_particles(pos, vel, mass):

    global crashed, escaped

    if len(pos) == 0:
        return pos, vel, mass

    distance = np.linalg.norm(pos, axis=1)

    hit_earth = distance < EARTH_RADIUS
    left_system = distance > ESCAPE_RADIUS

    crashed += np.sum(hit_earth)
    escaped += np.sum(left_system)

    keep = ~(hit_earth | left_system)

    return (
        pos[keep],
        vel[keep],
        mass[keep]
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

time_history = []
number_history = []
largest_mass_history = []
total_mass_history = []
angular_momentum_history = []


# ============================================================
# INITIAL ACCELERATION
# ============================================================

acceleration = calculate_acceleration(
    positions,
    masses
)


# ============================================================
# MAIN N-BODY LOOP
# ============================================================

for step in range(STEPS):

    time = step * DT

    # Velocity-Verlet / leapfrog style integration

    velocities += 0.5 * acceleration * DT

    positions += velocities * DT

    # Remove material hitting Earth / escaping
    positions, velocities, masses = (
        remove_lost_particles(
            positions,
            velocities,
            masses
        )
    )

    # Accretion
    positions, velocities, masses = (
        merge_collisions(
            positions,
            velocities,
            masses
        )
    )

    # Recalculate gravitational acceleration
    acceleration = calculate_acceleration(
        positions,
        masses
    )

    velocities += 0.5 * acceleration * DT

    # --------------------------------------------------------
    # Record diagnostics
    # --------------------------------------------------------

    time_history.append(time)

    number_history.append(len(masses))

    if len(masses) > 0:

        largest_mass_history.append(
            np.max(masses)
        )

        total_mass_history.append(
            np.sum(masses)
        )

        Lz = np.sum(
            masses
            * (
                positions[:, 0] * velocities[:, 1]
                -
                positions[:, 1] * velocities[:, 0]
            )
        )

        angular_momentum_history.append(Lz)

    else:

        largest_mass_history.append(0)
        total_mass_history.append(0)
        angular_momentum_history.append(0)


# ============================================================
# FOUR GRAPHS
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 9)
)

time = np.array(time_history)


# ------------------------------------------------------------
# GRAPH 1 — NUMBER OF BODIES
# ------------------------------------------------------------

axes[0, 0].plot(
    time,
    number_history
)

axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Number of bodies")

axes[0, 0].set_title(
    "Debris population"
)

axes[0, 0].grid(alpha=0.3)


# ------------------------------------------------------------
# GRAPH 2 — LARGEST AGGREGATE
# ------------------------------------------------------------

axes[0, 1].plot(
    time,
    largest_mass_history
)

axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Largest mass")

axes[0, 1].set_title(
    "Growth of largest aggregate"
)

axes[0, 1].grid(alpha=0.3)


# ------------------------------------------------------------
# GRAPH 3 — TOTAL MASS
# ------------------------------------------------------------

axes[1, 0].plot(
    time,
    total_mass_history
)

axes[1, 0].set_xlabel("Time")
axes[1, 0].set_ylabel("Remaining debris mass")

axes[1, 0].set_title(
    "Total debris mass"
)

axes[1, 0].grid(alpha=0.3)


# ------------------------------------------------------------
# GRAPH 4 — ANGULAR MOMENTUM
# ------------------------------------------------------------

axes[1, 1].plot(
    time,
    angular_momentum_history
)

axes[1, 1].set_xlabel("Time")
axes[1, 1].set_ylabel("$L_z$")

axes[1, 1].set_title(
    "Angular momentum"
)

axes[1, 1].grid(alpha=0.3)


plt.tight_layout()
plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

from IPython.display import HTML

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(snapshots),
    interval=30,
    blit=True
)

plt.close(fig)

HTML(ani.to_jshtml())