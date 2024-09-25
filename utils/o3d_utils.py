######################################
## Utility functions for Open3D.     #
## Author: Peizhi Yan                #
##   Date: 12/20/2023                #
## Update: 02/26/2024                #
######################################

import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt


def find_boundary_vertices(triangles):
    # Dictionary to count the occurrences of each edge
    edge_dict = {}

    # Iterate over each triangle
    for triangle in triangles:
        # For each edge in the triangle
        for i in range(3):
            # Create an edge tuple, sorted to avoid directional issues
            edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))

            # Count the occurrence of the edge
            if edge in edge_dict:
                edge_dict[edge] += 1
            else:
                edge_dict[edge] = 1

    # Find all edges that only appear once - these are boundary edges
    boundary_edges = [edge for edge, count in edge_dict.items() if count == 1]

    # Extract the unique vertices from these edges
    boundary_vertices = set(v for edge in boundary_edges for v in edge)

    return list(boundary_vertices)


def arap_optimization(V_old, V_new, Faces, handle_ids):
    """ As-rigid-as-possible
    inputs
        - V_old: vertices on old mesh [N, 3]
        - V_new: vertices on new mesh [N, 3]
        - Faces: tri-mesh faces [F, 3]
        - handle_ids: the ids of handle vertices, list
    returns
        - V_opt: optimized mesh vertives
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V_old)  # dtype vector3d (float)
    mesh.triangles = o3d.utility.Vector3iVector(Faces) # dtype vector3i (int)

    vertices = mesh.vertices

    ## set the handle vertices to be part vertices
    handle_pos = []
    for vid in handle_ids:
        handle_pos.append(V_new[vid])
    constraint_ids = o3d.utility.IntVector(handle_ids)
    constraint_pos = o3d.utility.Vector3dVector(handle_pos)

    #_original_stdout = sys.stdout
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh = mesh.deform_as_rigid_as_possible(constraint_ids, constraint_pos, max_iter=2) 
    V_opt = np.asarray(mesh.vertices, dtype=np.float32)
    
    return V_opt


def _read_obj_file(file_path, uv=False):
    """ Read the .obj file contents, and returns the mesh data
    NOTE: currently we only does not read normals, colors and uvs
    """
    vertices = [] # vertices
    faces = []    # triangle faces
    uvcoords = [] # uv coordinates values are in 0~1
    uvfaces = []  # uv indices

    with open(file_path, 'r') as file:
        for line in file:
            components = line.strip().split()
            if not components:
                continue
            if components[0] == 'v':  # Vertex
                vertex = [float(coord) for coord in components[1:4]]
                vertices.append(vertex)
            elif components[0] == 'vt':  # UV texture map coordinates
                uvcoord = [float(coord) for coord in components[1:3]]
                uvcoords.append(uvcoord)
            elif components[0] == 'f':  # Face
                face = [int(index.split('/')[0]) - 1 for index in components[1:4]]
                faces.append(face)
                if uv:
                    uvface = [int(index.split('/')[1]) - 1 for index in components[1:4]]
                    uvfaces.append(uvface)
    if uv:
        return np.array(vertices), np.array(faces), np.array(uvcoords), np.array(uvfaces)
    else:
        return np.array(vertices), np.array(faces)


def read_triangle_mesh(file_path):
    """ Read a triangle mesh using Open3D
    - for .ply files, simply uses Open3D's build-in function
    - for .obj files, read its contents and form the mesh
      this is because Open3D will mess the loaded vertex order
    """
    if file_path.endswith('.obj'):
        V, F, _ = _read_obj_file(file_path) # Vertices, Faces, UV coordinates (not used here)
        return form_mesh(V=V, T=None, Faces=F, smoothing=False)
    else:
        return o3d.io.read_triangle_mesh(file_path)


def form_mesh(V, T, Faces, smoothing=False):
    """ Form Open3D mesh object
    inputs
        - V: vertices [N, 3]
        - T: vertex colors [N, 3]
        - Faces: tri-mesh faces [F, 3]
    returns
        - mesh: Open3D mesh object
    """
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)      # dtype vector3d (float)
    mesh.triangles = o3d.utility.Vector3iVector(Faces) # dtype vector3i (int)
    
    if smoothing:
        # smoothing the mesh
        mesh = mesh.filter_smooth_laplacian(1, 0.5, 
                                            filter_scope=o3d.geometry.FilterScope.Vertex)

    if T is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(T) # dtype vector3i (int)
        
    # computing normal will give specular effect while rendering
    mesh.compute_vertex_normals()
    
    return mesh



def render(mesh, width=512, height=512, smooth_shading=True, show_normal=True, show_depth=False):
    """ Render mesh through Open3D
    inputs
        - mesh: Open3D mesh
        - width, height: render size
    returns
        - ret: dict{'image','depth'(optional)}
    """
    
    #mesh.compute_triangle_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible = False)
    
    opt = vis.get_render_option()
    if opt is None:
        #raise Exception('None Type ERROR with: vis.get_render_option() !!')
        print('None Type ERROR with: vis.get_render_option() !!')
        print('Usually because of conda environment ')
        print('SOLUTION: conda install -c conda-forge libstdcxx-ng=12')
        #print('sudo ln -s /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/')
        #print('sudo ln -s /usr/lib/x86_64-linux-gnu/dri/radeonsi_dri.so /usr/lib/dri/')
        #exit()

    #if T is None:
    if len(np.asarray(mesh.vertex_colors)) == 0 and show_normal:
        # use surface normal as texture if no texture exists
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal
    
    vis.add_geometry(mesh)

    # smooth shading
    if smooth_shading:
        opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color

    ret = {}
      
    # render image
    ret['image'] = vis.capture_screen_float_buffer(True)

    if show_depth:
        # render depth map
        ret['depth'] = vis.capture_depth_float_buffer(True)

    return ret



def save_mesh(mesh, save_path):
    """ 
    Save Open3D mesh to a given path (.obj)
    """
    mesh.compute_vertex_normals() # compute vertex normals
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    VN = np.asarray(mesh.vertex_normals) 
    with open(save_path, 'w') as f:
        f.write("# OBJ file\n")
        f.write("# Vertices: {}\n".format(len(V)))
        f.write("# Faces: {}\n".format(len(F)))
        for vid in range(len(V)):
            v = V[vid]
            vn = VN[vid]
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            f.write("vn {} {} {}\n".format(vn[0], vn[1], vn[2]))
        for p in F:
            f.write("f")
            for i in p:
                f.write(" {}".format((i + 1)))
            f.write("\n")
    #print('Mesh saved to ' + save_path)

    

def display_side_by_side(meshes, spacing, notebook=False):
    """ Display multiple meshes side by side
    inputs
        - meshes: List of Open3D triangle mesh objects
        - spacing: float, the spacing between two nearest meshes
        - notebook: Boolean,
            True for displaying via plotly in Jupyter notebook
            False for displaying via pop-up window on screen
    """
    assert isinstance(meshes, list)
    
    # make copies
    meshes_copy = []
    for mesh in meshes:
        meshes_copy.append(copy.deepcopy(mesh))

    # find the index of mesh at the middle
    middle_idx = len(meshes_copy) // 2
        
    # arrange the meshes at the left side
    x_offset = 0
    idx = middle_idx - 1
    while idx >= 0:
        x_offset -= spacing
        meshes_copy[idx].translate([x_offset, 0, 0])
        bbox = meshes_copy[idx].get_axis_aligned_bounding_box()
        width = bbox.max_bound[0] - bbox.min_bound[0]
        x_offset -= width
        idx -= 1

    # arrange the meshes at the right side
    x_offset = 0
    idx = middle_idx
    while idx < len(meshes_copy):
        x_offset += spacing
        meshes_copy[idx].translate([x_offset, 0, 0])
        bbox = meshes_copy[idx].get_axis_aligned_bounding_box()
        width = bbox.max_bound[0] - bbox.min_bound[0]
        x_offset += width
        idx += 1
    
    # display
    if notebook:
        ## NOTE: currently not working well when there are more than two meshes!!
        width = 800
        height = 400
        o3d.visualization.draw_plotly(meshes_copy, width=width, height=height, zoom=0.5)
    else:
        o3d.visualization.draw_geometries(meshes_copy)
        


def values_to_colors(values, colormap='viridis'):
    """ Converts mesh vertex values to RGB colors using a colormap.
    :param offsets: A numpy array of shape [N] representing vertex offsets.
    :param colormap: A string representing the matplotlib colormap to use.
    :return: A numpy array of shape [N, 3] representing RGB colors.
    """

    # Normalize the offsets to be in the range [0, 1]
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))

    # Get the colormap
    cmap = plt.get_cmap(colormap)

    # Map normalized offsets to colors using the colormap
    colors = cmap(normalized_values)[:, :3]  # Discard the alpha channel

    return colors



