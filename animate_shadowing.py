'''
Script to animate shadows. 

TODO: Generalise so this can be used for all visualisations by passign data to be plotted on facets and colour scale to be used. 
Add ability to click on a facet to get information about it. 

'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import matplotlib.animation as animation
from stl import mesh

# Global variables to control the animation state
is_paused = False
current_frame = 0

def onPress(event):
    global is_paused
    if event.key == ' ':
        is_paused ^= True  # Toggle pause state with each 'p' key press

def animate_shadowing(path_to_shape_model_file, insolation_array, rotation_axis, sunlight_direction, timesteps_per_day):
    ''' 
    This function animates the temperature evolution of the body for the final day of the model run. It rotates the body and updates the temperature of each facet.

    It uses the same rotation_matrix function as the visualise_shape_model function, and the same update function as the visualise_shape_model function but it updates the temperature of each facet at each frame using the temperature array from the data cube.
    '''
    global current_frame

    # Load the shape model from the STL file
    shape_mesh = mesh.Mesh.from_file(path_to_shape_model_file)

    # Create a figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Connect the key press event to toggle animation pause
    fig.canvas.mpl_connect('key_press_event', onPress)

    # Auto scale to the mesh size
    scale = shape_mesh.points.flatten('C')
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_aspect('equal')

    # Fix the view
    ax.view_init(elev=0, azim=0)

    # Get the current limits after autoscaling
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # Find the maximum range
    max_range = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
 
    # Calculate the middle points of each axis
    mid_x = np.mean(xlim)
    mid_y = np.mean(ylim)
    mid_z = np.mean(zlim)

    # Set new limits based on the maximum range to ensure equal scaling
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    ax.set_axis_off()

    # Initialize colour map and normalization
    norm = plt.Normalize(insolation_array.min(), insolation_array.max())
    colormap = plt.cm.rainbow #plt.cm.binary_r  # Use plt.cm.coolwarm to ensure compatibility

    # Create a ScalarMappable object with the normalization and colormap
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])

    line_length = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) * 0.5

    # Add the colour scale bar to the figure
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Insolation (W/m^2)', rotation=270, labelpad=20)

    plt.figtext(0.05, 0.01, 'Rotate with mouse, pause/resume with spacebar', fontsize=10, ha='left')

    def update(frame, shape_mesh, ax):
        global current_frame
        if not is_paused:
            current_frame = (current_frame + 1) % timesteps_per_day
        else:
            return

        # Rotate the mesh
        theta = (2 * np.pi / timesteps_per_day) * current_frame
        rot_mat = rotation_matrix(rotation_axis, theta)
        rotated_vertices = np.dot(shape_mesh.vectors.reshape((-1, 3)), rot_mat.T).reshape((-1, 3, 3))

        # Get temperatures for the current frame and apply colour map
        temp_for_frame = insolation_array[:, current_frame % timesteps_per_day]
        face_colours = colormap(norm(temp_for_frame))

        for art in reversed(ax.collections):
            art.remove()

        ax.add_collection3d(art3d.Poly3DCollection(rotated_vertices, facecolors=face_colours, linewidths=0, edgecolors='k', alpha=1.0))

        # Plot the reversed sunlight direction arrow pointing towards the center
        shift_factor = line_length * 2
        arrow_start = shift_factor * sunlight_direction
        ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                -sunlight_direction[0], -sunlight_direction[1], -sunlight_direction[2],
                length=line_length, color='orange', linewidth=2)

    # Animate
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, timesteps_per_day), fargs=(shape_mesh, ax), blit=False)

    # Display rotation period and solar distance as text
    plt.figtext(0.05, 0.95, f'Shadowing on body for one rotation', fontsize=14, ha='left')

    plt.show()

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
