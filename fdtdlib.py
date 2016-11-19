'''
FDTD Implementation in numpy and matplotlib

NOTE:: This is currently tied to a 1D simulation and work needs to be
       done to allow a choice of 1,2,3D.
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


class FDTD:
    '''
    Base class for FDTD simulations.
    At the moment this defaults to a 1D simulation of classical EM.
    '''
    def __init__(self, source_Fmax=5e9, λ_rmax=20,
                 grid_size=200, medium='air', boundary='Drichlet'):
        '''
        source_Fmax :: Maximum frequency that you want to resolve.
        λ_rmax      :: Number of grid cells used to cover the mininimum
                       wavelength at maximum frequency.
        grid_size   :: Length of each axis in the simulation.
        medium      :: Either a tuple of (εr, μr) for the primary simulation
                       medium or one of the strings "air" or "free_space".
        boundary    :: Numeric boundary conditions to use at the end of the
                       simulation grid: "Drichlet", "periodic" or "perfect".
        '''
        self.grid_size = grid_size
        self.boundary = boundary
        self.medium = medium
        self.λ_rmax = λ_rmax
        self.source_Fmax = source_Fmax
        self.λ_min = c0 / source_Fmax

        self.sources = {'E': [], 'M': []}
        self.device_regions = []

    def run(self, sources=[], devices=[],
            figsize=(600, 400), dpi=50, filename=None):
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
                   TODO:: Allow εr and μr to be lambdas
        '''
        # Initialise E and H fields
        self.Ey = np.zeros(self.grid_size)
        self.Hx = np.zeros(self.grid_size)
        # Compute remaining grid parameters based on sources
        dz, dt, εr, μr = self.initialise_grid(devices, sources)
        self.set_update_coefficients(εr, μr, dt)

        fig, ax, lines, x = self.initialise_plot(
                devices, εr[0]*μr[0], figsize, dpi)
        # NOTE:: Matplotlib's animation framework will loop and
        # add additional sources if we don't prevent it!
        self.use_sources = True

        def init():
            lines[0].set_data(x, self.Hx)
            lines[1].set_data(x, self.Ey)
            plt.legend(handles=lines)
            return lines

        def animate(step):
            if step == self.num_steps - 1:
                # Turn off the source after the first loop
                self.use_sources = False
            # Update both fields and then plot
            self.update_fields(step, dz)
            lines[0].set_data(x, self.Hx)
            lines[1].set_data(x, self.Ey)
            return lines

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

    def initialise_grid(self, devices, sources):
        '''
        Initialise the simulation space with relative values for ε and μ
        '''
        if self.medium == 'free_space':
            ε_relative = np.full(self.grid_size, ε0)
            μ_relative = np.full(self.grid_size, μ0)
        elif self.medium == 'air':
            ε_relative = np.ones(self.grid_size)
            μ_relative = np.ones(self.grid_size)
        else:
            try:
                εr, μr = self.medium
                ε_relative = np.full(self.grid_size, εr)
                μ_relative = np.full(self.grid_size, μr)
                # Fun for creating some non-uniform spaces
                # ε_relative = np.linspace(εr, μr, self.grid_size)
                # μ_relative = np.linspace(εr, μr, self.grid_size)
            except:
                raise TypeError('Invalid medium')

        for device in devices:
            εr, μr, region = device
            ε_relative[slice(*region)] = εr
            μ_relative[slice(*region)] = μr

        # Maximum refractive index
        n_max = np.sqrt(max(ε_relative) * max(μ_relative))
        dz = min(10, (self.λ_min / n_max / self.λ_rmax))

        # Ensure that the wave travels one grid cell in 2dz time intervals
        n_boundary = np.sqrt(ε_relative[0] * μ_relative[0])

        dt = n_boundary * dz / (2 * c0)

        # Compute the minimum simulation duration in terms of steps
        source_duration = 0.5 / self.source_Fmax
        source_time_offset = 5 * source_duration
        # Worst case time taken for the pulse to travel once over the grid
        source_travel_time = n_max * self.grid_size * dz / c0
        # NOTE:: Using pulse begining/end and 3 propagations
        simulation_time = 2 * source_time_offset + 2 * source_travel_time

        self.num_steps = int(np.ceil(simulation_time / dt))

        # Compute source pulses
        time_intervals = np.arange(0, self.num_steps) * dt

        for source in sources:
            # TODO:: allow pulses other than gaussian
            pulse_func = gaussian
            source_type, source_position = source
            source_pulse = np.apply_along_axis(
                    pulse_func, 0, time_intervals,
                    source_time_offset, source_duration)
            self.sources[source_type].append((source_position, source_pulse))

        return dz, dt, ε_relative, μ_relative

    def set_update_coefficients(self, ε_relative, μ_relative, dt):
        '''
        Compute the FDTD update coefficents based on the timestep dt
        and the current values of (ε|μ)_relative.

        NOTE: As this is a 1D case, the wave propagation k is being
              taken to be along the z axis and the resulting E and H
              fields are along the y and x axes respectively.
        '''
        self.Ey_δε = np.full(self.grid_size, (c0 * dt))
        self.Ey_δε /= ε_relative
        self.Hx_δμ = np.full(self.grid_size, (c0 * dt))
        self.Hx_δμ /= μ_relative

    def initialise_plot(self, devices, εrμr, figsize, dpi):
        '''
        Set up the animation figure, axes and lines
        '''
        x, y = figsize
        fig = plt.figure(figsize=(x/dpi, y/dpi), dpi=dpi)
        ax = plt.axes(xlim=(0, self.grid_size-2), ylim=(-2, 2))

        ax.set_title('1D EM Response to Gaussian Pulse')
        ax.set_xlabel('z-axis')
        ax.set_ylabel('EM field magnitude')
        ax.legend(prop=dict(size=10))

        # Shade device regions
        for device in devices:
            ε, μ, (start, stop) = device
            # Identify regions visibly
            if ε == μ:
                color = 'grey'
            elif ε > μ:
                color = 'yellow'
            else:
                color = 'red'
            if ε*μ > εrμr:
                alpha = 0.7
            else:
                alpha = 0.3
            ax.axvspan(start, stop, alpha=alpha, color=color)

        x = np.arange(0, self.grid_size)

        lines = [
            ax.plot(x, self.Hx, label='Magnetic Field (x-component)', lw=2)[0],
            ax.plot(x, self.Ey, label='Electric Field (y-component)', lw=2)[0]
            ]

        return fig, ax, lines, x

    def update_fields(self, step, dz):
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

        # NOTE:: These can _not_ be collapsed into a single loop as FDTD
        #        requires a time-consistant state around the point being
        #        updated.
        for point in range(self.grid_size-2):
            self.Hx[point] += (
                self.Hx_δμ[point] * (self.Ey[point+1] - self.Ey[point]) / dz
            )

        # Deal with final H field boundary condition
        final = self.grid_size - 1
        self.Hx[final] += (
                self.Hx_δμ[final] * (Hx_boundary - self.Ey[final]) / dz
            )

        # Add in the effect of M sources if we are in the first loop
        if self.use_sources:
            for source in self.sources['M']:
                source_position, M_source = source
                self.Hx[source_position] += M_source[step]

        # Deal with initial E field boundary condition
        self.Ey[0] += (self.Ey_δε[0] * (self.Hx[0] - Ey_boundary) / dz)

        for point in range(1, self.grid_size):
            self.Ey[point] += (
                self.Ey_δε[point] * (self.Hx[point] - self.Hx[point-1]) / dz
            )

        # Add in the effect of E sources if we are in the first loop
        if self.use_sources:
            for source in self.sources['E']:
                source_position, E_source = source
                self.Ey[source_position] += E_source[step]

if __name__ == '__main__':
    sources = [('E', 0)]
    devices = [(2, 1, (90, 120)), (0.6, 1, (120, 150))]
    devices = []
    filename = 'air_and_unmatched_materials.mp4'
    filename = None

    simulation = FDTD(
            source_Fmax=5e9, λ_rmax=20, grid_size=200,
            medium='air', boundary='Drichlet'
        )
    simulation.run(
        sources, devices, figsize=(500, 300), dpi=50, filename=filename
    )
