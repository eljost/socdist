#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.spatial.distance import pdist, squareform


FIGSIZE = (10, 8)


def engrad(coords, scale=1e-3):
    dists = pdist(coords)
    dists2 = dists**2

    triu_x, triu_y = np.triu_indices(coords.shape[0], k=1)
    # 1/r**2
    energy = scale * (1/dists2).sum()
    gradient_ = scale * -2*coords[triu_x] / (dists2*dists2)[:,None]

    gradient = np.zeros_like(coords)
    for x, g in zip(triu_x, gradient_):
        gradient[x] += g
    return energy, gradient


def optimize(coords, scale=1e-3, max_cycles=10, max_step=0.05):
    points = coords.shape[0]
    for i in range(max_cycles):
        e, g = engrad(coords, scale)
        print(f"{i:02d} {e:.6f}")

        if e < (points/2):
            print("Converged")
            break
        
        step = -g
        step_norms = np.linalg.norm(step, axis=1)
        max_step = step_norms.max()
        step = step / (max_step / 0.1)
        coords += step
        coords = np.clip(coords, 0, 1)
    return coords


def run(points=300, moving=0.125, infected=0.05, death_rate=0.02,
        recover_steps=150, steps=1000, seed=None, opt=False):
    if seed is not None:
        np.random.seed(seed)

    radius = 0.01
    velocity = 0.2
    dt = 0.02

    # Derived quantities
    points_moving = int(moving * points)
    moving_inds = np.random.choice(points, size=points_moving, replace=False)
    infected_num = int(infected * points)
    infected_inds = np.random.choice(points, size=infected_num, replace=False).tolist()
    recover_at = {recover_steps: infected_inds}
    already_recovered = set()
    points_dead = set()

    diameter = 2*radius
    min_x = 0 + radius
    max_x = 1 - radius
    min_y = 0 + radius
    max_y = 1 - radius

    print(f"{moving:.2%} moving, {points_moving} points")
    print(f"{infected:.2%} infected, {infected_num} points")

    coords = np.random.rand(points, 2)
    velocities = np.zeros_like(coords)
    directions = np.random.rand(points_moving, 2)
    directions /= np.linalg.norm(directions, axis=1)[:,None]
    velocities[moving_inds] = velocity * directions

    if opt:
        coords = optimize(coords)

    fig, ax0 = plt.subplots(figsize=FIGSIZE)
    fig.suptitle(f"{points} Points, recover after {recover_steps}")

    # All
    scatter = ax0.scatter(*coords.T, s=20)
    # Infected
    scatter_i = ax0.scatter(*coords[infected_inds].T, s=20, color="red")
    # Recovered
    scatter_r = ax0.scatter(*coords[list(already_recovered)].T, s=20, color="lightgreen")
    # Dead
    scatter_d = ax0.scatter(*coords[list(points_dead)].T, s=20, color="black")

    stack_x = np.arange(steps)
    stack_y = np.zeros((4, steps))

    def frame_gen():
        i = 0
        while infected_num > 0:
            yield i
            i += 1

    cur_frame = 0
    def propagate(frame):
        nonlocal coords
        nonlocal infected_inds
        nonlocal already_recovered
        nonlocal infected_num
        nonlocal cur_frame
        nonlocal points_dead
        cur_frame = frame

        # Recovery
        try:
            recovering = recover_at[frame]
            # import pdb; pdb.set_trace()
            infected_inds = list(set(infected_inds) - set(recovering))
            print("\tRecovered: ", recovering)
            already_recovered |= set(recovering)
        except KeyError:
            pass

        # Deaths
        death_roulette = np.random.rand(len(infected_inds))
        died_inds = np.flatnonzero(death_roulette <= death_rate)
        died_set = set(died_inds)
        if died_set:
            print(f"{len(died_set)} died!")
        velocities[died_inds] = (0., 0.)
        infected_inds = list(set(infected_inds) - died_set)
        points_dead |= died_set
        # import pdb; pdb.set_trace()

        if (frame % 50) == 0:
            print(frame)

        # Check for left/right reflection
        reflect_x = np.logical_or(coords[:,0] <= min_x, coords[:,0] >= max_x)
        velocities[reflect_x,0] *= -1
        # Check for up/down reflection
        reflect_y = np.logical_or(coords[:,1] <= min_y, coords[:,1] >= max_y)
        velocities[reflect_y,1] *= -1

        # Propagate
        step = dt*velocities 
        coords += step

        # Check for infection
        #   Calculate points in close contact, only consider infected points
        infected_coords = coords[infected_inds]
        dists = np.linalg.norm(coords - infected_coords[:,None,:], axis=2)
        _, new_infections = np.where(dists <= diameter)
        #   Update infected inds
        infected_set = set(infected_inds)
        new_infections = (set(new_infections) - infected_set - already_recovered
                          - set(points_dead))
        if new_infections:
            key = frame + recover_steps
            try:
                recover_at[key].extend(list(new_infections))
            except KeyError:
                recover_at[key] = list(new_infections)
        infected_inds = list(set(infected_inds) | set(new_infections))

        # All
        scatter.set_offsets(coords)
        # Infected
        scatter_i.set_offsets(coords[infected_inds])
        # Recovered
        scatter_r.set_offsets(coords[list(already_recovered)])
        scatter_d.set_offsets(coords[list(points_dead)])

        infected_num = len(infected_inds)
        recovered_num = len(already_recovered)
        dead_num = len(points_dead)
        pure = points - recovered_num - infected_num
        healthy = points - infected_num
        ratio = infected_num / points
        stack_y[:,frame] = (infected_num, recovered_num, dead_num, pure)

        if (frame % 50) == 0:
            title = f"Frame {frame:03d}, {ratio:.2%} infected"
            ax0.set_title(title)

    # propagate(0)
    # propagate(1)
    # return
    anim = animation.FuncAnimation(fig,
                                   propagate,
                                   frames=frame_gen(),
                                   interval=5,
                                   repeat=False,
    )
    plt.show()

    fig, ax = plt.subplots()
    stacks = ax.stackplot(stack_x, stack_y, colors=("red", "lightgreen", "k", "blue"))
    fig.suptitle(f"Step {cur_frame}")
    plt.show()


if __name__ == "__main__":
    seed = 20180325
    run(seed=seed, opt=True)
