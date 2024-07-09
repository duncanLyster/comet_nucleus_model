'''
Script to animate model quantities on the shape. The model is rotated about a specified axis and the quantity is plotted on the shape. You can move the camera around the shape and pause the animation with the arrow keys and spacebar. Right click on the shape to get the value of the quantity at that point.

TODO:
1) Leave animation up (or save it in an interactable mode) but allow the main code to continue running.
2) Fix BUG - plot window doesn't close when you hit x.
3) Fix BUG - camera movement is jumpy when using arrow keys.
4) Sort out colour scale bar values - should be more sensibly spaced.
5) Fix BUG - segmentation fault the second time this script is run.
6) Put a vertical line on the plot to show the current time.
7) Press 'C' to clear the selected cells.

NOTE: This is currently very slow when called from the main script for large shape models.
'''

import logging
import traceback
import sys
from contextlib import contextmanager

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('animation_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

import pyvista as pv
import vtk
from stl import mesh
import time
import math
import matplotlib.pyplot as plt
import gc
import weakref
import numpy as np  

@contextmanager
def animation_context():
    try:
        yield
    finally:
        # Perform thorough cleanup here
        logging.debug("Starting cleanup in animation_context")
        pv.close_all()
        plt.close('all')
        gc.collect()
        time.sleep(1)  # Allow time for complete cleanup
        logging.debug("Cleanup in animation_context complete")

class AnimationState:
    def __init__(self):
        self.is_paused = False
        self.current_frame = 0
        self.camera_phi = math.pi / 2
        self.camera_theta = math.pi / 2
        self.camera_radius = None
        self.selected_cells = []
        self.fig, self.ax = None, None

def on_press(state):
    state.is_paused = not state.is_paused

def update_camera_position(plotter, state):
    # Convert spherical coordinates to Cartesian coordinates
    x = state.camera_radius * math.sin(state.camera_phi) * math.cos(state.camera_theta)
    y = state.camera_radius * math.sin(state.camera_phi) * math.sin(state.camera_theta)
    z = state.camera_radius * math.cos(state.camera_phi)

    plotter.camera_position = [(x, y, z), (0, 0, 0), (0, 0, 1)]
    # plotter.reset_camera()
    plotter.camera.view_angle = 30
    plotter.render()

def move_up(plotter, state):
    state.camera_phi -= math.pi / 36  # Decrease polar angle
    update_camera_position(plotter, state)

def move_down(plotter, state):
    state.camera_phi += math.pi / 36  # Increase polar angle
    update_camera_position(plotter, state)

def move_left(plotter, state):
    state.camera_theta -= math.pi / 36  # Decrease azimuthal angle
    update_camera_position(plotter, state)

def move_right(plotter, state):
    state.camera_theta += math.pi / 36  # Increase azimuthal angle
    update_camera_position(plotter, state)

def round_up_to_nearest(x, base):
    return base * math.ceil(x / base)

def round_down_to_nearest(x, base):
    return base * math.floor(x / base)

def clear_selected_cells(state):
    state.selected_cells = []
    if state.fig is not None:
        plt.close(state.fig)
        state.fig = None
        state.ax = None
    print("Selected cells cleared.")

def rotation_matrix(axis, theta):
    axis = [a / math.sqrt(sum(a**2 for a in axis)) for a in axis]
    a = math.cos(theta / 2.0)
    b, c, d = [-axis[i] * math.sin(theta / 2.0) for i in range(3)]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return [[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]]

def animate_model(path_to_shape_model_file, plotted_variable_array, rotation_axis, sunlight_direction, 
                  timesteps_per_day, colour_map, plot_title, axis_label, animation_frames, 
                  save_animation, save_animation_name, background_colour):

    logging.debug("Starting animate_model function")
    state = AnimationState()
    plotter = None

    with animation_context():
        try: 
            # Start timer
            start_time = time.time()

            # Load shape model
            logging.debug("Loading shape model")
            shape_mesh = mesh.Mesh.from_file(path_to_shape_model_file)
            logging.debug("Shape model loaded successfully")
                    
            # Check if the plotted variable array is the correct shape
            if shape_mesh.vectors.shape[0] != plotted_variable_array.shape[0]:
                print("The plotted variable array must have the same number of rows as the number of cells in the shape model.")
                return
            
            # Ensure all points in the plotted variable array are not identical
            if np.all(plotted_variable_array == plotted_variable_array[0]):
                print("All points in the plotted variable array are identical. Please check the data.")
                return
            
            vertices = shape_mesh.points.reshape(-1, 3)
            faces = [[3, 3*i, 3*i+1, 3*i+2] for i in range(shape_mesh.vectors.shape[0])]

            # Create a PyVista mesh
            logging.debug("Creating PyVista mesh")
            pv_mesh = pv.PolyData(vertices, faces)
            pv_mesh.cell_data[axis_label] = plotted_variable_array[:, 0]
            logging.debug("PyVista mesh created successfully")

            # Determine text color based on background color
            text_color = 'white' if background_colour == 'black' else 'black' 
            bar_color = (1, 1, 1) if background_colour == 'black' else (0, 0, 0)

            # Animation dimension calculations
            bounding_box = pv_mesh.bounds
            max_dimension = max(bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2], bounding_box[5] - bounding_box[4])
            state.camera_radius = max_dimension * 5

            # Create a Plotter object
        
            logging.debug("Starting to set up plotter")
            plotter = pv.Plotter()
            plotter.add_key_event('space', lambda: on_press(state))
            plotter.add_key_event('Up', lambda: move_up(plotter, state))
            plotter.add_key_event('Down', lambda: move_down(plotter, state))
            plotter.add_key_event('Left', lambda: move_left(plotter, state))
            plotter.add_key_event('Right', lambda: move_right(plotter, state))
            plotter.add_key_event('c', lambda: clear_selected_cells(state))
            plotter.iren.initialize()

            # update_camera_position(plotter, state)

            # Axis
            cylinder_start = [0, 0, 0]
            cylinder_length = max_dimension  # Adjust the scale factor as needed
            cylinder = pv.Cylinder(center=cylinder_start, direction=rotation_axis, height=cylinder_length, radius=max_dimension/200)

            plotter.add_mesh(cylinder, color='green')

            # Sunlight vector
            sunlight_start = [sunlight_direction[i] * max_dimension for i in range(3)]
            sunlight_length = max_dimension * 0.3  # Adjust the scale factor as needed
            sunlight_arrow = pv.Arrow(start=sunlight_start, direction=[-d for d in sunlight_direction], scale=sunlight_length)

            plotter.add_mesh(sunlight_arrow, color='yellow')

            # Add the text to the window
            plotter.add_text("Press spacebar to pause, right click to select a facet.", position='lower_edge', font_size=10, color=text_color)

            logging.debug("Plotter setup complete")

            logging.debug("Setting up update function")

            def update(caller, event, state):
                logging.debug(f"Updating frame: {state.current_frame}")
                if not state.is_paused:
                    state.current_frame = (state.current_frame + sampling_interval) % timesteps_per_day

                    if state.current_frame >= num_frames:
                        state.current_frame = 0

                    theta = (2 * math.pi / timesteps_per_day) * state.current_frame
                    rot_mat = rotation_matrix(rotation_axis, theta)
                    try:
                        rotated_vertices = np.dot(vertices, np.array(rot_mat).T)
                        pv_mesh.points = rotated_vertices
                        pv_mesh.cell_data[axis_label] = plotted_variable_array[:, state.current_frame % timesteps_per_day].copy()
                    except Exception as e:
                        print(f"Error updating mesh vertices: {e}")
                        return

                    plotter.render()

                logging.debug("Frame update complete")

            logging.debug("Update function setup complete")

            plotter.iren.add_observer('TimerEvent', lambda caller, event: update(caller, event, state))
            plotter.iren.create_timer(100)

            def plot_picked_cell_over_time(state, cell_id):
                if state.selected_cells is None:
                    state.selected_cells = [cell_id]
                else:
                    state.selected_cells.append(cell_id)

                if not state.selected_cells:
                    return  # Don't plot if there are no selected cells

                if state.fig is None or state.ax is None:
                    plt.ion()
                    state.fig, state.ax = plt.subplots()

                state.ax.clear() 

                for cell in state.selected_cells:
                    values_over_time = plotted_variable_array[cell, :]
                    time_steps = [i / timesteps_per_day for i in range(len(values_over_time))]
                    state.ax.plot(time_steps, values_over_time, label=f'Cell {cell}')

                state.ax.set_xlabel('Local time (days)')
                state.ax.set_ylabel(axis_label)
                state.ax.set_title(f'Diurnal variation of {axis_label} of selected Cells')
                state.ax.legend()
                state.fig.canvas.draw()
                state.fig.canvas.flush_events()


            logging.debug("Setting up picking functionality")
            def on_pick(state, picked_mesh):
                if picked_mesh is not None:
                    cell_id = picked_mesh['vtkOriginalCellIds'][0]
                    plot_picked_cell_over_time(state, cell_id)

            logging.debug("Picking functionality setup complete")

            # Add the mesh to the plotter
            logging.debug("Adding mesh to plotter")
            mesh_actor = plotter.add_mesh(pv_mesh, scalars=axis_label, cmap=colour_map, show_edges=False)
            logging.debug("Mesh added to plotter")
            plotter.enable_element_picking(callback=lambda picked_mesh: on_pick(state, picked_mesh), mode='cell', show_message=False)

            text_color_rgb = (1, 1, 1) if background_colour == 'black' else (0, 0, 0)

            # Scale bar properties
            logging.debug("Setting up scalar bar")
            plotter.scalar_bar.GetLabelTextProperty().SetColor(bar_color)
            plotter.scalar_bar.SetPosition(0.2, 0.05)  # Set the position of the bottom left corner of the scalar bar
            plotter.scalar_bar.GetLabelTextProperty().SetJustificationToCentered()  # Center the labels
            plotter.scalar_bar.SetTitle(axis_label)
            plotter.scalar_bar.GetTitleTextProperty().SetColor(text_color_rgb)
            logging.debug("Scalar bar setup complete")

            # Get the range of the plotted variable
            min_val = min(min(row) for row in plotted_variable_array)
            max_val = max(max(row) for row in plotted_variable_array)
            range_val = max_val - min_val

            # Determine a suitable interval
            interval = 10 ** (math.floor(math.log10(range_val)) - 1)
            rounded_min_val = round_down_to_nearest(min_val, interval)
            rounded_max_val = round_up_to_nearest(max_val, interval)

            # Adjust interval if necessary to avoid too many/few labels
            while (rounded_max_val - rounded_min_val) / interval > 10:
                interval *= 2
            while (rounded_max_val - rounded_min_val) / interval < 4:
                interval /= 2

            # Create custom labels
            labels = []
            label = rounded_min_val
            while label <= rounded_max_val:
                labels.append(label)
                label += interval

            # Convert the custom labels to a vtkDoubleArray
            vtk_labels = vtk.vtkDoubleArray()
            vtk_labels.SetNumberOfValues(len(labels))
            for i, label in enumerate(labels):
                vtk_labels.SetValue(i, label)

            # Set the custom labels
            plotter.scalar_bar.SetCustomLabels(vtk_labels)

            # Enable the use of custom labels
            plotter.scalar_bar.SetUseCustomLabels(True)

            plotter.add_text(plot_title, position='upper_edge', font_size=12, color=text_color)
            plotter.add_text("Press spacebar to pause, right click to select a facet, 'c' to clear selected facets.", position='lower_edge', font_size=10, color=text_color)
            plotter.background_color = background_colour

            # Calculate the sampling interval
            num_frames = len(plotted_variable_array[0])
            sampling_interval = max(1, num_frames // animation_frames)


            logging.debug("Starting animation loop")
            if save_animation:
                plotter.open_gif(save_animation_name)
                for _ in range(animation_frames):
                    update(plotter, None, state)
                    plotter.write_frame()
                plotter.close()
            else:
                end_time = time.time()
                print(f'Animation took {end_time - start_time:.2f} seconds to run.')
                logging.debug("Showing interactive animation")
                plotter.show() # interactive_update=True
                plotter.close()
                logging.debug("Interactive animation closed")

            # Cleanup
            logging.debug("Starting cleanup process")
            if plotter:
                logging.debug("Closing plotter")
                plotter.close()
                logging.debug("Plotter closed")
                logging.debug("Running deep clean on plotter")
                plotter.deep_clean()
                logging.debug("Plotter deep clean complete")

            logging.debug("Deleting objects")
            del pv_mesh
            del shape_mesh
            del plotter

            if state.fig:
                logging.debug("Closing matplotlib figure")
                plt.close(state.fig)
                logging.debug("Matplotlib figure closed")

            logging.debug("Closing all PyVista and Matplotlib windows")
            pv.close_all()
            plt.close('all')

            if 'vtk_labels' in locals():
                logging.debug("Removing observers from vtk_labels")
                vtk_labels.RemoveAllObservers()
                del vtk_labels

            logging.debug("Resetting state")
            state.fig = None
            state.ax = None
            state.selected_cells = []

            logging.debug("Running garbage collection")
            gc.collect()
            logging.debug("Garbage collection complete")

        except Exception as e:
            logging.error(f"An error occurred in animate_model: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            if plotter:
                logging.debug("Attempting to close plotter due to error")
                try:
                    plotter.close()
                except:
                    logging.error("Failed to close plotter")
            raise  # Re-raise the exception after logging

        finally:
            logging.debug("animate_model function finished")
