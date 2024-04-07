import numpy as np
import random as r
import math

# import pythonabm library and the physics models
from pythonabm import Simulation, record_time, commandline_param
from physics import *


class CellSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self):
        # initialize the Simulation object
        Simulation.__init__(self)

        # read parameters from YAML file and add them as instance variables
        self.yaml_parameters("adhesion/general.yaml")

        # contact mechanics parameters
        s = commandline_param("-s", float)
        self.adhesion_h = 0.2*s*1e-4 +  2e-4    # kg s^-1
        self.adhesion_l = 2e-4    # kg s^-1
        self.stokes = 0.4    # N s^-1 m^-1
        self.youngs = 1000    # Pa
        self.poisson = 0.5

        # other parameters
        self.move_force = 1e-8    # N
        self.grav_force = 2e-9    # N
        self.step_dt = 36   # s
        self.move_dt = 6   # s

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # add agents to the simulation, indicate agent subtypes
        self.add_agents(self.num_blue_agents, agent_type="blue")
        self.add_agents(self.num_yellow_agents, agent_type="yellow")

        # indicate agent arrays for storing agent values
        self.indicate_arrays("locations", "radii", "colors", "state", "contact_forces", "motility_forces")

        # function for uniform circle distribution
        def uniform_circle():
            center_x, center_y = self.size[0] / 2, self.size[1] / 2
            radius = 200 * math.sqrt(r.random())
            angle = math.tau * r.random()
            return np.array([center_x + radius * math.cos(angle), center_y + radius * math.sin(angle), 0])

        # set initial agent values
        self.locations = self.agent_array(vector=3, initial=lambda: uniform_circle())
        self.radii = self.agent_array(initial=lambda: r.gauss(5, 0.25))
        self.colors = self.agent_array(vector=3, initial={"blue": (0, 0, 255), "yellow": (255, 255, 0)}, dtype=int)
        self.state = self.agent_array(initial={"blue": 0, "yellow": 1}, dtype=bool)
        self.contact_forces = self.agent_array(vector=3)    # hertz, DMT, JKR
        self.motility_forces = self.agent_array(vector=3)    # random movement

        # indicate agent graphs and create a graph for holding agent neighbors
        self.indicate_graphs("contact_graph")
        self.contact_graph = self.agent_graph()

        # call once to seed the cells
        self.cell_physics()

        # record initial values
        self.step_values()
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # call the following methods that update agent values
        # self.die()
        # self.reproduce()
        self.random_move()
        self.gravity()
        self.cell_physics()

        # add/remove agents from the simulation
        self.update_populations()
        # self.step_values()
        # self.step_image()

    def end(self):
        """ Overrides the end() method from the Simulation class.
        """
        self.step_values()
        self.step_image()
        # self.create_video()

    @record_time
    def reproduce(self):
        """ Determine which agents will hatch a new agent during this step.
        """
        for index in range(self.number_agents):
            if r.random() < 0.1:
                self.mark_to_hatch(index)

    @record_time
    def die(self):
        """ Determine which agents will die during this step.
        """
        for index in range(self.number_agents):
            if r.random() < 0.1:
                self.mark_to_remove(index)

    @record_time
    def random_move(self):
        """ Move specified magnitude in random direction.
        """
        # get random vector for each cell
        for index in range(self.number_agents):
            self.motility_forces[index] += self.random_vector() * self.move_force

    @record_time
    def gravity(self):
        # get random vector for each cell
        for index in range(self.number_agents):
            center = np.array([self.size[0]/2, self.size[1]/2, 0])
            vector = center - self.locations[index]
            mag = np.linalg.norm(vector)

            self.motility_forces[index] += self.grav_force * (vector / 200)

    @record_time
    def cell_physics(self):
        """
        """
        # calculate the total number of steps
        steps = int(self.step_dt / self.move_dt)

        # iterate through move steps
        for step in range(steps):
            # update graph for pairs of contacting cells (clear is set to False for JKR)
            self.get_neighbors(self.contact_graph, 2 * np.amax(self.radii), clear=False)

            # calculate the contact forces
            self.calculate_jkr()
            # self.calculate_dmt()
            # self.calculate_hertz()

            # turn size into numpy array and convert radii to meters
            size = np.asarray(self.size)

            # calculate new locations
            locations = apply_forces(self.number_agents, self.locations, self.radii, self.contact_forces,
                                     self.motility_forces, self.stokes, size, self.move_dt)

            # update the locations and reset contact forces back to zero
            self.locations = locations
            self.contact_forces[:, :] = 0

        self.motility_forces[:, :] = 0

    def calculate_jkr(self):
        """
        """
        # get the edges as an array, count them, and create holder used to delete edges
        edges = np.array(self.contact_graph.get_edgelist())
        number_edges = len(edges)
        delete = np.zeros(number_edges, dtype=bool)

        # only run if there are edges
        if number_edges > 0:
            forces, delete = jkr_forces(number_edges, edges, delete, self.locations, self.radii, self.state,
                                        self.contact_forces, self.poisson, self.youngs, self.adhesion_h, self.adhesion_l)

            # update the graph to remove any adhesions that broke and forces
            self.contact_graph.delete_edges(np.arange(number_edges)[delete])
            self.contact_forces = forces

    def calculate_dmt(self):
        """
        """
        # get the edges as an array, count them, and create holder used to delete edges
        edges = np.array(self.contact_graph.get_edgelist())
        number_edges = len(edges)

        # only run if there are edges
        if number_edges > 0:
            forces = dmt_forces(number_edges, edges, self.locations, self.radii, self.state,
                                self.contact_forces, self.poisson, self.youngs, self.adhesion_h, self.adhesion_l)

            # update forces
            self.contact_forces = forces

    def calculate_hertz(self):
        """
        """
        # get the edges as an array, count them, and create holder used to delete edges
        edges = np.array(self.contact_graph.get_edgelist())
        number_edges = len(edges)

        # only run if there are edges
        if number_edges > 0:
            forces = hertz_forces(number_edges, edges, self.locations, self.radii,
                                  self.contact_forces, self.poisson, self.youngs)

            # update forces
            self.contact_forces = forces


    @classmethod
    def simulation_mode_0(cls, name, output_dir):
        """ Creates a new brand new simulation and runs it through
            all defined steps.

            :param name: The name of the simulation.
            :param output_dir: Path to simulation output directory.
            :type name: str
            :type output_dir: str
        """
        # make simulation instance, update name, and add paths
        sim = cls()
        sim.name = name
        sim.set_paths(output_dir)

        # set up the simulation agents and run the simulation
        sim.full_setup()
        sim.run_simulation()


if __name__ == "__main__":
    CellSimulation.start("outputs")
