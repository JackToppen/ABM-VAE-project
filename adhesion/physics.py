import numpy as np
import math
from numba import jit, prange


@jit(nopython=True, parallel=True, cache=True)
def jkr_forces(number_edges, edges, delete, locations, radii, state, contact_forces, poisson,
               youngs, adhesion_h, adhesion_l):
    """ Calculates forces based on the JKR model.
    """
    # go through the edges array
    for edge_index in prange(number_edges):
        # get the cell indices of the edge
        cell_1 = edges[edge_index][0]
        cell_2 = edges[edge_index][1]

        # get work of adhesion
        if (state[cell_1] + state[cell_2]) % 2 == 0:
            adhesion = adhesion_h
        else:
            adhesion = adhesion_l

        # get the vector between the centers of the cells, the magnitude of this vector and the overlap of the cells
        vector = locations[cell_1] - locations[cell_2]
        mag = np.linalg.norm(vector)
        overlap = (radii[cell_1] + radii[cell_2] - mag) / 1e6    # convert to meters

        # calculate effective radius (convert to meters) and E
        big_r = ((1/radii[cell_1] + 1/radii[cell_2]))**(-1) / 1e6
        big_e = (2 * (1 - poisson**2) / youngs)**(-1)

        # value used to non-dimensionalize overlap magnitude
        overlap_ = (((math.pi * adhesion) / big_e) ** (2/3)) * (big_r ** (1/3))

        # get the nondimensionalized overlap
        d = overlap / overlap_

        # check to see if the cells will have a force interaction based on the nondimensionalized distance
        if d > -0.360562:
            # plug the value of d into polynomial approximation for nondimensionalized force
            f = (-0.0886 * d ** 3) + (0.7559 * d ** 2) + (0.8378 * d) - 1.325

            # convert from the nondimensionalized force to find the JKR force
            jkr_force = f * math.pi * adhesion * big_r

            # if the magnitude is 0 use the zero vector, otherwise find the normalized vector for each axis
            normal = np.array([0.0, 0.0, 0.0])
            if mag != 0:
                normal += vector / mag

            # adds the force as a vector in opposite directions to each cell's force holder
            contact_forces[cell_1] += jkr_force * normal
            contact_forces[cell_2] -= jkr_force * normal

        # remove the edge if the it fails to meet the criteria for distance, simulating that the bond is broken
        else:
            delete[edge_index] = 1

    return contact_forces, delete


@jit(nopython=True, parallel=True, cache=True)
def dmt_forces(number_edges, edges, locations, radii, state, contact_forces, poisson,
               youngs, adhesion_h, adhesion_l):
    """ Calculates forces based on the DMT model.
    """
    # go through the edges array
    for edge_index in prange(number_edges):
        # get the cell indices of the edge
        cell_1 = edges[edge_index][0]
        cell_2 = edges[edge_index][1]

        # get the vector between the centers of the cells, the magnitude of this vector and the overlap of the cells
        vector = locations[cell_1] - locations[cell_2]
        mag = np.linalg.norm(vector)
        overlap = (radii[cell_1] + radii[cell_2] - mag) / 1e6    # convert to meters
        if overlap > 0:

            # get work of adhesion
            if (state[cell_1] + state[cell_2]) % 2 == 0:
                adhesion = adhesion_h
            else:
                adhesion = adhesion_l

            # calculate effective radius (convert to meters) and E
            big_r = ((1/radii[cell_1] + 1/radii[cell_2]))**(-1) / 1e6
            big_e = (2 * (1 - poisson**2) / youngs)**(-1)

            # calculate the contact radius
            a = (big_r * overlap) ** (1/2)

            # calculate dmt force
            dmt_force = (4*big_e*(a**3) / (3*big_r)) - 2 * adhesion * math.pi * big_r

            # if the magnitude is 0 use the zero vector, otherwise find the normalized vector for each axis
            normal = np.array([0.0, 0.0, 0.0])
            if mag != 0:
                normal += vector / mag

            # adds the force as a vector in opposite directions to each cell's force holder
            contact_forces[cell_1] += dmt_force * normal
            contact_forces[cell_2] -= dmt_force * normal

    return contact_forces


@jit(nopython=True, parallel=True, cache=True)
def hertz_forces(number_edges, edges, locations, radii, contact_forces, poisson,
                 youngs):
    """ Calculates forces based on the Hertzian model.
    """
    # go through the edges array
    for edge_index in prange(number_edges):
        # get the cell indices of the edge
        cell_1 = edges[edge_index][0]
        cell_2 = edges[edge_index][1]

        # get the vector between the centers of the cells, the magnitude of this vector and the overlap of the cells
        vector = locations[cell_1] - locations[cell_2]
        mag = np.linalg.norm(vector)
        overlap = (radii[cell_1] + radii[cell_2] - mag) / 1e6    # convert to meters
        if overlap > 0:

            # calculate effective radius (convert to meters) and E
            big_r = ((1/radii[cell_1] + 1/radii[cell_2]))**(-1) / 1e6
            big_e = (2 * (1 - poisson**2) / youngs)**(-1)

            # calculate the contact radius
            a = (big_r * overlap) ** (1/2)

            # calculate hertz force
            hertz_force = 4/3 * big_e * (big_r)**(1/2) * (overlap)**(3/2)

            # if the magnitude is 0 use the zero vector, otherwise find the normalized vector for each axis
            normal = np.array([0.0, 0.0, 0.0])
            if mag != 0:
                normal += vector / mag

            # adds the force as a vector in opposite directions to each cell's force holder
            contact_forces[cell_1] += hertz_force * normal
            contact_forces[cell_2] -= hertz_force * normal

    return contact_forces


@jit(nopython=True, parallel=True, cache=True)
def apply_forces(number_agents, locations, radii, contact_forces, motility_forces,
                 stokes, size, move_dt):
    """ Applies forces based on motility and contacts.
    """
    for index in prange(number_agents):
        # stokes law for velocity fluid viscosity, convert radii to meters
        friction = 0.4
        velocity = (motility_forces[index] + contact_forces[index]) / friction

        # set the new location, convert velocity to m/s
        new_location = locations[index] + move_dt * (velocity * 1e6)

        # loop over all directions of space and check if new location is in the space
        for i in range(0, 3):
            if new_location[i] > size[i]:
                locations[index][i] = size[i]
            elif new_location[i] < 0:
                locations[index][i] = 0
            else:
                locations[index][i] = new_location[i]

    return locations
