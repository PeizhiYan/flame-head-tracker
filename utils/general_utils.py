#
# General utility functions.     
# Author: Peizhi Yan                
# Copyright. 2025
#

import numpy as np
import cv2
import torch
from pytorch3d.renderer import (
    OrthographicCameras, MeshRenderer, MeshRasterizer,
    SoftPhongShader, RasterizationSettings, PointLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from utils.graphics_utils import compute_vertex_normals

def check_nan_in_dict(array_dict):
    # Function to check for NaN in dictionary of NumPy arrays
    nan_detected = {}
    flag = False # True: there exist at least a NaN value
    for key, array in array_dict.items():
        if np.isnan(array).any():
            nan_detected[key] = True
            flag = True
        else:
            nan_detected[key] = False
    return nan_detected, flag


def plot_landmarks(img, lmks, color=(0, 255, 0), radius=2, thickness=-1, copy=True):
    """
    Draw 2D landmarks on an image.

    Args:
        img: np.ndarray, shape (H, W, 3) or (H, W), dtype uint8 or float32
        lmks: np.ndarray, shape (68, 2), float32 or float64
        color: tuple, color of the points (B, G, R)
        radius: int, dot radius
        thickness: int, -1 for filled
        copy: bool, whether to copy image before drawing
    Returns:
        img_draw: image with landmarks
    """
    if copy:
        img_draw = img.copy()
    else:
        img_draw = img

    for (x, y) in lmks:
        cv2.circle(img_draw, (int(round(x)), int(round(y))), radius, color, thickness, lineType=cv2.LINE_AA)

    return img_draw


def draw_landmarks(img, landmarks_2d_screen, eye_landmarks2d_screen, ear_landmarks2d_screen, blendweight=0.8):
    """
    Draw landmarks on the image.
    Args:
        img: The image on which to draw landmarks. [H, W, 3] uint8
        landmarks_2d_screen: 2D landmarks in screen coordinates. [68, 2]
        eye_landmarks2d_screen: Eye landmarks in screen coordinates. [10, 2]
        ear_landmarks2d_screen: Ear landmarks in screen coordinates. [40, 2]
        blendweight: how much to keep the original image
    """
    H = img.shape[0]
    img = cv2.addWeighted(img, blendweight, np.full_like(img, 255), 1-blendweight, 0) # blend with white filter
    for j, coords in enumerate(landmarks_2d_screen):
        coords = np.clip(coords, 0, H-1).astype(np.uint8)
        #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
        if j < 17 // 2: color = (0, 100, 0)
        elif j < 17: color = (0, 255, 0)
        else: color = (150, 255, 0)
        cv2.circle(img, (coords[0], coords[1]), radius=0, color=color, thickness=2)  # Green color, filled circle

    # draw eye landmarks as blue dots
    for j, coords in enumerate(eye_landmarks2d_screen):
        color = (20, 20, 255)
        coords = np.clip(coords, 0, H-1).astype(np.uint8)
        #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
        cv2.circle(img, (coords[0], coords[1]), radius=0, color=color, thickness=2)  # Blue color, filled circle

    # draw ear landmarks as pink dots
    for j, coords in enumerate(ear_landmarks2d_screen):
        if j < 20: color = (100, 0, 100)
        else: color = (255, 0, 255)
        coords = np.clip(coords, 0, H-1).astype(np.uint8)
        #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
        cv2.circle(img, (coords[0], coords[1]), radius=0, color=color, thickness=2)
    return img


@torch.no_grad()
def render_geometry(vertices, verts_ndc_3d, faces, device, render_size=256):
    """ 
    For visualizing the FLAME mesh using PyTorch3D.
    Args:
        vertices: [V, 3] vertices of the mesh              numpy
        verts_ndc_3d: [V, 3] vertices in NDC coordinates   numpy
        faces: [F, 3] triangle faces of the mesh           numpy
    Returns:
        rendered_img: [H, W, 3] rendered image in uint8 RGB format
        foreground_mask: [H, W] binary mask of the rendered image
    """
    faces = torch.from_numpy(faces).to(device)   # [F, 3] faces of the template mesh
    vertices = torch.from_numpy(vertices).to(device)  # [V, 3] vertices of the mesh
    verts_ndc_3d = torch.from_numpy(verts_ndc_3d).to(device)  # [V, 3] vertices in NDC coordinates
    mesh = Meshes(verts=verts_ndc_3d[None], faces=faces[None])

    # Compute vertex normals
    normals = compute_vertex_normals(verts=vertices, faces=faces)  # [V, 3]

    # Set up lighting and fake camera
    light_pos = torch.tensor([0.0, 0.0, 10.0], device=vertices.device)  # In front of the mesh
    camera_pos = torch.tensor([0.0, 0.0, 10.0], device=vertices.device)
    ambient_color  = torch.tensor([0.1, 0.1, 0.1], device=vertices.device)  # [3], RGB
    diffuse_color  = torch.tensor([0.8, 0.8, 0.8], device=vertices.device)
    specular_color = torch.tensor([0.0, 0.0, 0.0], device=vertices.device)
    shininess = 30.0  # Specular exponent
    base_color = torch.ones_like(vertices) 

    v = vertices     # [V, 3]
    n = normals      # [V, 3]

    # Light direction (from vertex to light)
    light_dir = (light_pos - v)  # [V, 3]
    light_dir = light_dir / (light_dir.norm(dim=-1, keepdim=True) + 1e-8)

    # View direction (from vertex to camera)
    view_dir = (camera_pos - v)
    view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)

    # Reflection direction for specular
    reflect_dir = 2 * (n * (n * light_dir).sum(-1, keepdim=True)) - light_dir
    reflect_dir = reflect_dir / (reflect_dir.norm(dim=-1, keepdim=True) + 1e-8)

    # Ambient term: just base color * ambient
    ambient = ambient_color * base_color  # [V, 3]

    # Diffuse term: base color * diffuse * max(0, dot(normal, light_dir))
    diff = torch.clamp((n * light_dir).sum(-1, keepdim=True), min=0.0)
    diffuse = diffuse_color * base_color * diff  # [V, 3]

    # Specular term: specular color * [max(0, dot(reflect_dir, view_dir))^shininess]
    spec = torch.clamp((reflect_dir * view_dir).sum(-1, keepdim=True), min=0.0)
    specular = specular_color * (spec ** shininess)  # [V, 3]

    # Final color per vertex
    vertex_color = ambient + diffuse + specular
    vertex_color = torch.clamp(vertex_color, 0.0, 1.0)  # Ensure in [0,1]

    mesh.textures = TexturesVertex(verts_features=vertex_color[None])
    
    # No camera transform: this is a hack; set camera at the origin with identity rotation
    R=torch.eye(3)
    R[0,0] = -1.0
    R[1,1] = -1.0
    R = R[None].to(device)  # Add batch dimension and move to device
    T=torch.tensor([0,0,0])[None]  # Place the camera at z=1.0
    cameras = OrthographicCameras(device=device, R=R, T=T)

    # Standard rasterization/rendering setup
    raster_settings = RasterizationSettings(
        image_size=512,  # e.g., 224
        blur_radius=0.0,
        faces_per_pixel=1
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]],
                        ambient_color=[[1.0, 1.0, 1.0]],
                        diffuse_color=[[0.0, 0.0, 0.0]],
                        specular_color=[[0.0, 0.0, 0.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    # Render the mesh
    image = renderer(mesh)
    alpha = image[0, ..., 3].cpu().numpy() 
    foreground_mask = (alpha > 0).astype(np.uint8)
    image = image[0, ..., :3].cpu().numpy()  # Convert to numpy array and remove alpha channel

    image = (image * 255).astype(np.uint8)  # Convert to uint8 RGB format
    image = np.clip(image, 0, 255)

    image = cv2.resize(image, (render_size, render_size))
    foreground_mask = cv2.resize(foreground_mask, (render_size, render_size))

    return image, foreground_mask


@torch.no_grad()
def prepare_batch_visualized_results(vertices, faces, in_dict, verts_ndc_3d, RENDER_SIZE, 
                                     landmarks_2d_screen, eye_landmarks2d_screen, ear_landmarks2d_screen):
    """
    Returns:
        img_rendered_batch: [N, 256, 256, 3]
        mesh_rendered_batch: [N, 256, 256, 3]
    """

    img_rendered_list = []   # to store blended+landmark images
    mesh_rendered_list = []  # to store raw mesh rendering

    for n in range(vertices.shape[0]):
        img_input = np.copy(in_dict['img_resized'][n])  # [256, 256, 3]
        mesh_rendered, fg_mask = render_geometry(
            vertices[n].detach().cpu().numpy(), 
            verts_ndc_3d[n].detach().cpu().numpy(), 
            faces=np.copy(faces),
            device=vertices.device,
            render_size=RENDER_SIZE
        )
        # Blend mesh with original image
        img_rendered = cv2.addWeighted(img_input, 0.4, mesh_rendered, 0.6, 0)
        # Draw landmarks on blended image
        img_rendered = draw_landmarks(
            img_rendered,
            landmarks_2d_screen[n],
            eye_landmarks2d_screen[n],
            ear_landmarks2d_screen[n],
            blendweight=1.0
        )
        img_rendered_list.append(img_rendered)
        mesh_rendered_list.append(mesh_rendered)

    # Convert to batched numpy arrays: [N, 256, 256, 3]
    img_rendered_batch = np.stack(img_rendered_list, axis=0)
    mesh_rendered_batch = np.stack(mesh_rendered_list, axis=0)

    return img_rendered_batch, mesh_rendered_batch
