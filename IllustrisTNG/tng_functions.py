import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo  # Planck 2018 parameters

# Read simulation header file for misc info
def get_header_info( snap_num, get_info='dark_matter_particle_mass' ):
    
    # Read header to find standardized mass of dark matter particles
    with h5py.File(il.snapshot.snapPath(SIM_DIR, snap_num),'r') as f:
        header = dict(f['Header'].attrs)
        
        if get_info == 'dark_matter_particle_mass':            
            return header['MassTable'][1]
        elif get_info == 'redshift':
            return header['Redshift']
        else:
            print("UNKNOWN INFO REQUEST")


# Useful function for constructing/deconstructing subhalo ids.
def generate_subhalo_id_raw(snap_num, subfind_id):
    # Convert input to integers in case they are passed as strings
    snap_num = int(snap_num)
    subfind_id = int(subfind_id)
    # Calculate the SubhaloIDRaw
    subhalo_id_raw = snap_num * 10**12 + subfind_id
    return subhalo_id_raw

def deconstruct_subhalo_id_raw(subhalo_id_raw):
    # Convert input to integer in case it is passed as a string
    subhalo_id_raw = int(subhalo_id_raw)
    # Extract SnapNum and SubfindID from SubhaloIDRaw
    snap_num = subhalo_id_raw // 10**12
    subfind_id = subhalo_id_raw % 10**12
    return (snap_num, subfind_id)


def explore_hdf5(file_path):
    """
    Recursively explore the contents of an HDF5 file.
    
    Parameters:
        file_path (str): The path to the HDF5 file to explore.
    
    Returns:
        dict: A dictionary containing information about the file's structure.
    """
    def recurse_through_group(group, prefix=''):
        """Helper function to recurse through groups and datasets."""
        info = {}
        for key, item in group.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                # Gathering information for datasets
                info[path] = {
                    'type': 'Dataset',
                    'data_type': str(item.dtype),
                    'shape': item.shape,
                    'size': item.size,
                    'compression': item.compression
                }
            elif isinstance(item, h5py.Group):
                # Recurse into groups
                info[path] = {
                    'type': 'Group',
                    'contents': recurse_through_group(item, path)
                }
        return info
    
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as file:
        return recurse_through_group(file)

# Converting redshift to time function

# This snapshot to redshift data is from the following TNG sim file:
# "sims.TNG/TNG50-1/postprocessing/tracer_tracks/tr_all_groups_99_parent_indextype.hdf5"
snapshot_data = {
    'snapshots': np.array([99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83,
        82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66,
        65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
        48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
        14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0]).astype(int), 
        'redshifts': np.array([2.2204460e-16, 9.5216669e-03, 2.3974428e-02, 3.3724371e-02,
        4.8523631e-02, 5.8507323e-02, 7.3661387e-02, 8.3884433e-02,
        9.9401802e-02, 1.0986994e-01, 1.2575933e-01, 1.4187621e-01,
        1.5274876e-01, 1.6925204e-01, 1.8038526e-01, 1.9728418e-01,
        2.1442504e-01, 2.2598839e-01, 2.4354018e-01, 2.6134327e-01,
        2.7335334e-01, 2.9771769e-01, 3.1007412e-01, 3.2882974e-01,
        3.4785384e-01, 3.6068764e-01, 3.8016787e-01, 3.9992696e-01,
        4.1996893e-01, 4.4029784e-01, 4.6091780e-01, 4.8183295e-01,
        5.0304753e-01, 5.2456582e-01, 5.4639220e-01, 5.7598084e-01,
        5.9854329e-01, 6.2142873e-01, 6.4464182e-01, 6.7611039e-01,
        7.0010638e-01, 7.3263615e-01, 7.5744140e-01, 7.9106826e-01,
        8.1671000e-01, 8.5147089e-01, 8.8689691e-01, 9.2300081e-01,
        9.5053136e-01, 9.9729425e-01, 1.0355104e+00, 1.0744579e+00,
        1.1141505e+00, 1.1546028e+00, 1.2062581e+00, 1.2484726e+00,
        1.3023784e+00, 1.3575766e+00, 1.4140983e+00, 1.4955121e+00,
        1.5312390e+00, 1.6042346e+00, 1.6666696e+00, 1.7435706e+00,
        1.8226893e+00, 1.9040896e+00, 2.0020282e+00, 2.1032696e+00,
        2.2079256e+00, 2.3161108e+00, 2.4442258e+00, 2.5772903e+00,
        2.7331426e+00, 2.8957851e+00, 3.0081310e+00, 3.2830331e+00,
        3.4908614e+00, 3.7087743e+00, 4.0079451e+00, 4.1768351e+00,
        4.4280338e+00, 4.6645179e+00, 4.9959335e+00, 5.2275810e+00,
        5.5297656e+00, 5.8466139e+00, 6.0107574e+00, 6.4915977e+00,
        7.0054169e+00, 7.2362761e+00, 7.5951071e+00, 8.0121727e+00,
        8.4494762e+00, 9.0023403e+00, 9.3887711e+00, 9.9965906e+00,
        1.0975643e+01, 1.1980213e+01, 1.4989173e+01, 2.0046492e+01],
        dtype=np.float64)
        }

snapshot_data['Gyr'] = cosmo.age(snapshot_data['redshifts']).value

# Create a dictionary that maps snapshot number to time in Gyrs
snap_time = {}
for i in range(len(snapshot_data['snapshots'])):
    snap_time[snapshot_data['snapshots'][i]] = snapshot_data['Gyr'][i]

def snap_to_time(snap_num):
    return snap_time[snap_num]


def xy_scatter_plot(xy_pts=None, pt_labels=None, \
                    plot_lines=None, plot_labels=None, \
                    sig_pts=None, sig_labels=None, \
                    vector_list=None, vector_labels=None, \
                    axis = 'xy', \
                    title="Scatter Plot with Labels", std_dev=3, \
                    square = True, figsize=None, legend=True, 
                    save_loc = None):
    """
    Plots a scatter plot with labeled points having unique colors and large, unlabeled dots.
    Ignores points that are more than 3 standard deviations from the mean in either dimension.

    Args:
    xy_pts (np.ndarray): An N x 2 array of xy coordinates.
    pt_labels (np.ndarray): An N-sized array of labels for each point in xy_pts.
    plot_lines (list(np.ndarray)): An N2 list of M x 2 arrays of xy coordinates for lines to plot.
    plot_labels (list): An N2 sized list to label the plot_lines.
    sig_pts (np.ndarray): An M x 2 array of xy coordinates for significant positions.
    sig_labels (list): An M sized list to label the sig_pts.
    title (str): Title of the plot.
    """

    # Ensure input integrity for scatter
    if xy_pts is not None:
        assert xy_pts.shape[1] >= 2, "xy_pts should be N x 2 in shape"

        if pt_labels is not None:
            assert xy_pts.shape[0] == len(pt_labels), "xy_pts and pt_labels must have the same length"

    # Ensure input integrity for lines
    if plot_lines is not None:
        if type(plot_lines) is not list:
            plot_lines = [plot_lines]

        if plot_labels is not None:
            assert len(plot_labels) == len(plot_lines), "plot_labels must match the number of plot_lines"

    # Ensure input integrity for significant points
    if sig_pts is not None:
        assert sig_pts.shape[1] >= 2, "sig_pts should be N x 2 in shape"

        if sig_labels is not None:
            assert len(sig_labels) == sig_pts.shape[0], "sig_labels must match the number of sig_pts"

    # Ensure input integrity for vectors (2D vectors)
    if vector_list is not None:
        assert vector_list.shape[1] == 2, "vector_list should be N x 2 x 2 in shape"
        assert vector_list.shape[2] >= 2, "vector_list should be N x 2 x 2 in shape"

        if vector_labels is not None:
            assert len(vector_labels) == vector_list.shape[0], "vector_labels must match the number of vectors"

    # Create a plot
    fig, ax = plt.subplots()

    # Remove outliers from data
    if xy_pts is not None:
        # Calculate the mean and standard deviation to ignore plotting outliers.
        mean_xy = np.mean(xy_pts, axis=0)
        std_xy = np.std(xy_pts, axis=0)

        # Create masks for data within specified standard deviations
        within_bounds = np.all(np.abs(xy_pts - mean_xy) <= std_dev * std_xy, axis=1)

        # Filter xy_pts and pt_labels based on the mask
        xy_pts = xy_pts[within_bounds]

        # Generate a color map from labels
        if pt_labels is not None:
            pt_labels = pt_labels[within_bounds]
            unique_labels = np.unique(pt_labels)
            colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))  # Using jet colormap
            label_color_dict = dict(zip(unique_labels, colors))

        # Generate a single label for all pts if labels are not provided
        else:
            pt_labels = np.zeros(xy_pts.shape[0])
            unique_labels = np.unique(pt_labels)
            colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))  # Using jet colormap
            label_color_dict = dict(zip(unique_labels, colors))

        # Plotting labeled points
        for label in unique_labels:
            indices = pt_labels == label
            ax.scatter(xy_pts[indices, 0], xy_pts[indices, 1], label=str(label),
                       color=label_color_dict[label], s=1)  # s is the size of the normal dots

    # Plot lines if given
    if plot_lines is not None:
        if plot_labels is None:
            plot_labels = [f"Line {i}" for i in range(len(plot_lines))]

        for i, line in enumerate(plot_lines):
            ax.plot(line[:, 0], line[:, 1], label=plot_labels[i], linewidth=1)

    # Plotting significant points
    if sig_pts is not None:
        if sig_labels is None:
            sig_labels = [f"Point {i}" for i in range(sig_pts.shape[0])]

        for i, sig_pt in enumerate(sig_pts):
            ax.scatter(sig_pt[0], sig_pt[1], color='black', s=25, edgecolor='black', linewidth=1)
            ax.text(sig_pt[0], sig_pt[1], f"{sig_labels[i]}", fontsize=8)

    # Add vectors if given
    if vector_list is not None:
        if vector_labels is None:
            vector_labels = [f"Vector {i}" for i in range(vector_list.shape[0])]

        for i, vectors in enumerate(vector_list):
            ax.plot([vectors[0, 0], vectors[1, 0]], [vectors[0, 1], vectors[1, 1]],
                    marker='o', linewidth=1, label=vector_labels[i])

    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    if legend:
        # Optional: Add a legend
        ax.legend(title="Labels")

    # Make square 
    if square:
        ax.set_aspect('equal', 'box')

    # Set figure size
    if figsize is not None:
        fig.set_size_inches(figsize)


    # Show plot
    if save_loc is None:
        plt.show()

    if save_loc is not None:
        fig.savefig(save_loc)



def interactive_3d_scatter_plot(xyz_pts=None, pt_labels=None, \
                                plot_lines = None, plot_labels = None, \
                                sig_pts = None, sig_labels = None, \
                                vector_list = None, vector_labels = None, \
                                title="Scatter Plot with Labels", std_dev=4,
                                save_loc = None):
    """
    Plots an interactive 3D scatter plot with labeled points having unique colors and large, unlabeled dots.

    Args:
    xyz_pts (np.ndarray): An N1 x 3 array of xyz coordinates.
    labels (np.ndarray): An N1-sized array of labels for each point in xyz_pts.
    plot_lines (list(np.ndarray)): An N2 list of M x 3 arrays of xyz coordinates for lines to plot. 
    plot_label (list): An N2 sized list to labelt the plot_lines
    sig_pts (np.ndarray): An N4 x 3 array of xyz coordinates for significant positions.
    sig_labels (list): An N4 sized list to label the sig_pts
    vector_list (np.ndarray): An N5 x 2 x 3 array of xyz coordinates for vectors to plot.
    vector_labels (list): An N5 sized list to label the vectors
    title (str): Title of the plot.
    """

    import plotly.graph_objects as go

    # Ensure input integrity for scatter
    if xyz_pts is not None:
        assert xyz_pts.shape[1] >= 3, "xyz_pts should be N x 3 in shape"

        if pt_labels is not None:
            assert xyz_pts.shape[0] == len(pt_labels), "xyz_pts and pt_labels must have the same length"

    # Ensure input integrity for lines
    if plot_lines is not None:

        if type(plot_lines) is not list:
            plot_lines = [plot_lines]

        if plot_labels is not None:
            assert len(plot_labels) == len(plot_lines), "plot_labels must match the number of plot_lines"

    # Ensure input integrity for significant points
    if sig_pts is not None:
        assert sig_pts.shape[1] >= 3, "large_dots should be N x 3 in shape"

        if sig_labels is not None:
            assert len(sig_labels) == sig_pts.shape[0], "sig_labels must match the number of sig_pts"

    # Ensure input integrity for vectors
    if vector_list is not None:
        assert vector_list.shape[1] == 2, "vector_list should be N x 2 x 3 in shape"
        assert vector_list.shape[2] == 3, "vector_list should be N x 2 x 3 in shape"

        if vector_labels is not None:
            assert len(vector_labels) == vector_list.shape[0], "vector_labels must match the number of vectors"


    # Create a Plotly figure
    fig = go.Figure()

    # Define a color palette
    default_colors = ['blue', 'red', 'green']

    # Remove outlyers from data
    if xyz_pts is not None:
        # Calculate the mean and standard deviation.  To ignore plotting outlyers.
        mean_xyz = np.mean(xyz_pts, axis=0)
        std_xyz = np.std(xyz_pts, axis=0)

        # Create masks for data within 4 standard deviations
        within_bounds = np.all(np.abs(xyz_pts - mean_xyz) <= std_dev * std_xyz, axis=1)

        # Filter xy_data and labels based on the mask
        xyz_pts = xyz_pts[within_bounds]

        if pt_labels is not None:
            pt_labels = pt_labels[within_bounds]
            unique_labels = np.unique(pt_labels)

            # Use predefined colors if there are up to three unique labels
            if len(unique_labels) <= 3:
                colors = default_colors[:len(unique_labels)]
            else:
                # Cycle through the default colors if there are more than three unique labels
                colors = [default_colors[i % 3] for i in range(len(unique_labels))]
                
            label_color_dict = dict(zip(unique_labels, colors))
        else:
            pt_labels = np.zeros(xyz_pts.shape[0])
            unique_labels = np.unique(pt_labels)
            label_color_dict = {0: 'blue'}

        # Plotting labeled points
        for label in unique_labels:
            indices = pt_labels == label
            fig.add_trace(go.Scatter3d(
                x=xyz_pts[indices, 0], y=xyz_pts[indices, 1], z=xyz_pts[indices, 2],
                mode='markers',
                marker=dict(size=2, color=label_color_dict[label]),
                name=str(label)
            ))



    # Plot lines if given
    if plot_lines is not None:

        if plot_labels is None:
            plot_labels = [f"Line {i}" for i in range(len(plot_lines))]

        for i, line in enumerate(plot_lines):
            fig.add_trace(go.Scatter3d(
                x=line[:, 0], y=line[:, 1], z=line[:, 2],
                mode='lines+markers',
                line=dict(width=2),
                marker=dict(size=2),
                name=plot_labels[i]
            ))
 

    # Plotting significant points
    if sig_pts is not None:

        if sig_labels is None:
            sig_labels = [f"Point {i}" for i in range(sig_pts.shape[0])]

        for i, sig_pt in enumerate(sig_pts):
            fig.add_trace(go.Scatter3d(
                x=[sig_pt[0]], y=[sig_pt[1]], z=[sig_pt[2]],
                mode='markers',
                marker=dict(size=7, color='black'),
                name=sig_labels[i]))

            fig.add_trace(go.Scatter3d(
                x=[sig_pt[0]], y=[sig_pt[1]], z=[sig_pt[2]],
                mode='text',
                text=f"{sig_labels[i]}",
                textposition="top center",
                name=sig_labels[i]))
            
        
    # Add vectors if given
    if vector_list is not None:

        if vector_labels is None:
            vector_labels = [f"Vector {i}" for i in range(vector_list.shape[0])]

        for i, vectors in enumerate(vector_list):
            fig.add_trace(go.Scatter3d(
                x=[vectors[0,0], vectors[1,0]],
                y=[vectors[0,1], vectors[1,1]],
                z=[vectors[0,2], vectors[1,2]],
                mode='lines+markers',
                marker=dict(size=5, color='blue'),
                line=dict(color='blue', width=2),
                name=vector_labels[i]
            ))

    # Update plot layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Z Coordinate',
            # aspectmode='cube'  # This makes x, y, z scales equal
        ),
        legend_title="Labels",

        width=1600,
        height=800,
    )

    # Show plot
    fig.show()