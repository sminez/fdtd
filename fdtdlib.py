'''
FDTD Implementation in numpy and matplotlib

'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.constants import speed_of_light, epsilon_0, mu_0

# .: Plot configuration :.
sns.set(style='white')
sns.set_context('notebook', font_scale=1.5)
gruvbox = [
        '#cc241d', '#fabd2f', '#458588', '#b16286',
        '#689d6a', '#fe8019', '#7c6f64', '#504945'
        ]
sns.set_palette(gruvbox)

# .: Numeric Constants :.
c0 = speed_of_light
ε0 = epsilon_0
μ0 = mu_0


# .: Source pulse functions :.
def gaussian(t, t0, σ):
    '''
    A standard Gaussian Curve:
      t  :: Variable
      t0 :: Time Shift
      σ  :: Standard Deviation
    '''
    return np.exp(-(t - t0) ** 2 / (2 * σ ** 2)) / 2


class FDTD_base:
    n_dimensions = None
    mediums = None
    boundaries = None

    def __init__(self, source_Fmax, λ_rmax, axis_lengths, medium, boundary):
        '''
        source_Fmax :: Int
            Maximum frequency that will be resolved.
        λ_rmax :: Int
            Number of grid cells used to cover the mininimum
            wavelength at the maximum resolved frequency.
        axis_lengths :: Tuple[Int, ...]
            Number of grid cells to use for of each axis in the simulation.
        medium :: (Str | Tuple[Float, Float])
            Either a tuple of (εr, μr) for the main simulation
            medium or a string specifying a default medium.
        boundary :: Str
            Numeric boundary conditions to use at the end of the
            simulation grid.
        '''
        self.axis_lengths = axis_lengths
        self.boundary = boundary
        self.medium = medium
        self.λ_rmax = λ_rmax
        self.source_Fmax = source_Fmax
        self.λ_min = c0 / source_Fmax
        self.dt = self.dx = self.dy = self.dz = 0
        self.sources = {'E': [], 'M': []}
        self.device_regions = []

    def initialise_grid(self, devices, sources):
        '''
        Set the primary simulation medium along with added device
        regions and compute source effects.
        '''
        def components(k):
            comp = [np.full(a, k) for a in self.axis_lengths if a > 0]
            if len(comp) == 1:
                return comp[0]
            else:
                return np.array(comp)

        εr, μr = self.mediums.get(self.medium, self.medium)

        try:
            self.εr = components(εr)
            self.μr = components(μr)
        except:
            raise TypeError('Invalid medium specification')

        self.insert_devices(devices)

        n_max = np.sqrt(max(self.εr) * max(self.μr))
        dk = min(10, (self.λ_min / n_max / self.λ_rmax))
        self.dx = self.dy = self.dz = dk

        # Ensure that the wave travels one grid cell in 2dz time intervals
        # NOTE:: Devices should not be allowed at boundaries!
        n_boundary = np.sqrt(self.εr[0] * self.μr[0])
        l_max = max(self.axis_lengths)

        source_travel_time = n_max * l_max * dk / c0
        self.dt = n_boundary * dk / (2 * c0)

        self.compute_sources_and_duration(sources, source_travel_time)
        self.set_update_coefficients()

    def insert_devices(self, devices):
        '''
        Implementations are required to define how to add sources to their
        n-dimensional εr and μr arrays
        '''
        raise NotImplementedError

    def compute_sources_and_duration(self, sources, source_travel_time):
        '''
        Sources are specifed with a type (E/M) and a coordinate position
        that is compatible with the formulation of the Implementation
        update equations.
        '''
        source_duration = 0.5 / self.source_Fmax
        source_time_offset = 5 * source_duration

        # NOTE:: This is an arbitrary choice of the source pulse duration
        #        and two passes over the grid.
        simulation_time = 2 * source_time_offset + 2 * source_travel_time

        self.num_steps = int(np.ceil(simulation_time / self.dt))

        time_intervals = np.arange(0, self.num_steps) * self.dt

        for source in sources:
            source_type, source_position = source
            source_pulse = np.apply_along_axis(
                    gaussian, 0, time_intervals,
                    source_time_offset, source_duration)
            self.sources[source_type].append((source_position, source_pulse))

    def set_update_coefficients(self):
        '''
        Pre-compute any constant update coefficents required for update_fields
        >> This is called as part of initialise_grid()
        '''
        raise NotImplementedError

    def initialise_plot(self, devices, figsize, dpi):
        '''
        Set up the animation figure, axes and lines
        '''
        raise NotImplementedError

    def update_fields(self, step, x, lines):
        '''
        For a given time step, this updates every point in the simulation
        grid to its new value and returns a new set of lines to plot.
        NOTE:: Due to the use of a Yee Lattice, the E and H fields are
               staggered in both space and time. This simplifies the
               update equations but does mean that interpolation is
               required if you want to obtain values for discrete points.
        '''
        raise NotImplementedError

    def run(self, sources, devices, figsize, dpi, filename):
        '''
        Add in Electric and Magnetic sources then run the simulation.
        If a filename is specified then an mp4 will be saved under that name
        in the working directory.

        sources :: A list of source tuples defining the source as Electric or
                   Magnetic and a location.
                        (('E'|'M'), z-coordinate)
                    e.g. ('E', 50), ('M', 0)
        devices :: A list of device tuples specifying εr, μr and a region.
                        (2, 2, (20, 50))
        '''
        # Compute remaining grid parameters based on sources
        self.initialise_grid(devices, sources)
        self.set_update_coefficients()

        fig, ax, lines, x = self.initialise_plot(devices, figsize, dpi)
        # NOTE:: Matplotlib's animation framework will loop and
        # add additional sources if we don't prevent it!
        self.use_sources = True

        def init():
            '''Set the legend and return the initial plot'''
            plt.legend(handles=lines)
            return lines

        def animate(step):
            if step == self.num_steps - 1:
                # Turn off the source after the first loop
                self.use_sources = False
            # Update both fields twice and then plot
            updated_lines = self.update_fields(step, x, lines)
            return updated_lines

        anim = animation.FuncAnimation(
                fig, animate, init_func=init,
                frames=self.num_steps, interval=0.1)

        if filename:
            if filename.endswith('.mp4'):
                anim.save(
                    '{}'.format(filename), writer='ffmpeg',
                    fps=100, bitrate=2000)
            elif filename.endswith('.gif'):
                anim.save(
                    '{}'.format(filename), writer='imagemagick',
                    fps=100, dpi=dpi)
            else:
                raise ValueError('filename must end in either .mp4 or .gif')

        sns.set_context('notebook', font_scale=2)
        plt.show()


##############################################################################
# .: Implementations :. #
#########################
class FDTD_1D_Maxwell(FDTD_base):
    ''' A 1D simulation of classical EM.
    Available default mediums: air, free_space
    Available boundary conditions: Drichlet, periodic, perfect
    '''
    n_dimensions = 1
    mediums = {
            'air': (1, 1),
            'free_space': (ε0, μ0)
        }
    boundaries = {
            'Drichlet': {'values': (0, 0)},
            'periodic': {'indices': (0, -1)},
            'perfect': None
        }

    def __init__(self, source_Fmax=5e9, λ_rmax=20,
                 grid_size=200, medium='air', boundary='Drichlet'):
        '''Default values to initialise a 1D simulation'''
        super().__init__(
            source_Fmax=source_Fmax,
            λ_rmax=λ_rmax,
            axis_lengths=(0, 0, grid_size),
            medium=medium,
            boundary=boundary
        )
        # TODO:: remove!
        # Conveniance to not index into axis_lengths
        self.grid_size = grid_size
        # Initialise E and H fields
        self.Ey = np.zeros(self.grid_size)
        self.Hx = np.zeros(self.grid_size)

    def insert_devices(self, devices):
        '''
        Implementations are required to define how to add sources to their
        n-dimensional εr and μr arrays
        '''
        for device in devices:
            εr, μr, region = device
            self.εr[slice(*region)] = εr
            self.μr[slice(*region)] = μr

    def set_update_coefficients(self):
        '''
        NOTE: As this is a 1D case, the wave propagation k is being
              taken to be along the z axis and the resulting E and H
              fields are along the y and x axes respectively.
        '''
        self.Ey_δε = np.full(self.grid_size, (c0 * self.dt))
        self.Ey_δε /= self.εr
        self.Hx_δμ = np.full(self.grid_size, (c0 * self.dt))
        self.Hx_δμ /= self.μr

    def initialise_plot(self, devices, figsize, dpi):
        '''
        Set up the animation figure, axes and lines
        '''
        x, y = figsize
        fig = plt.figure(figsize=(x/dpi, y/dpi), dpi=dpi)
        ax = plt.axes(xlim=(0, self.grid_size-2), ylim=(-2, 2))

        ax.set_title('1D EM Response to Gaussian Pulse')
        ax.legend(prop=dict(size=10))

        # Shade device regions
        for device in devices:
            ε, μ, (start, stop) = device
            if ε == μ:
                color = 'grey'
            elif ε > μ:
                color = 'yellow'
            else:
                color = 'red'
            if ε * μ > self.εr[0] * self.μr[0]:
                alpha = 0.5
            else:
                alpha = 0.3
            ax.axvspan(start, stop, alpha=alpha, color=color)

        x = np.arange(0, self.grid_size)

        lines = [
            ax.plot(x, self.Hx, label='Magnetic Field (x-component)', lw=2)[0],
            ax.plot(x, self.Ey, label='Electric Field (y-component)', lw=2)[0]
            ]
        # Initialise to current values
        lines[0].set_data(x, self.Hx)
        lines[1].set_data(x, self.Ey)

        return fig, ax, lines, x

    def update_fields(self, step, x, lines):
        '''
        For a given time step, this updates every point in the simulation
        grid to its new value.
        NOTE:: Due to the use of a Yee Lattice, the E and H fields are
               staggered in both space and time. This simplifies the
               update equations but does mean that interpolation is
               required if you want to obtain values for discrete points.
        '''
        if self.boundary == 'Drichlet':
            Hx_boundary, Ey_boundary = 0, 0
        elif self.boundary == 'periodic':
            Hx_boundary, Ey_boundary = self.Hx[0], self.Ey[-1]
        elif self.boundary == 'perfect':
            raise ValueError('I know I said this was here...just not yet.')
        else:
            raise ValueError('Invalid boundary condition')

        for point in range(self.grid_size-2):
            self.Hx[point] += (
                self.Hx_δμ[point]*(self.Ey[point+1]-self.Ey[point])/self.dz
            )

        # Deal with final H field boundary condition
        final = self.grid_size - 1
        self.Hx[final] += (
                self.Hx_δμ[final]*(Hx_boundary-self.Ey[final])/self.dz
            )

        # Add in the effect of M sources if we are in the first loop
        if self.use_sources:
            for source in self.sources['M']:
                source_position, M_source = source
                self.Hx[source_position] += M_source[step]

        # Deal with initial E field boundary condition
        self.Ey[0] += (self.Ey_δε[0]*(self.Hx[0]-Ey_boundary)/self.dz)

        for point in range(1, self.grid_size):
            self.Ey[point] += (
                self.Ey_δε[point]*(self.Hx[point]-self.Hx[point-1])/self.dz
            )

        # Add in the effect of E sources if we are in the first loop
        if self.use_sources:
            for source in self.sources['E']:
                source_position, E_source = source
                self.Ey[source_position] += E_source[step]

        # Set the new plot values and return them
        lines[0].set_data(x, self.Hx)
        lines[1].set_data(x, self.Ey)
        return lines

if __name__ == '__main__':
    sources = [('E', 0)]
    # High-low-high ||::::::||
    devices = [
        (2, 2, (95, 100)),
        (0.5, 0.5, (100, 125)),
        (2, 2, (125, 130))
    ]
    devices = []
    filename = 'air_and_unmatched_materials.mp4'
    filename = None

    simulation = FDTD_1D_Maxwell(
            source_Fmax=5e9, λ_rmax=20, grid_size=200,
            medium='air', boundary='Drichlet'
        )
    simulation.run(
        sources, devices, figsize=(500, 300), dpi=50, filename=filename
    )
