import torch
try:
    import nvdiffrast.torch as dr
except :
    print("nvdiffrast are not installed. Please install them to use the mesh renderer.")
from easydict import EasyDict as edict
from ..representations.mesh import MeshExtractResult
import torch.nn.functional as F


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


class MeshRenderer:
    """
    Renderer for the Mesh representation.

    Args:
        rendering_options (dict): Rendering options.
        glctx (nvdiffrast.torch.RasterizeGLContext): RasterizeGLContext object for CUDA/OpenGL interop.
    """
    def __init__(self, rendering_options={}, device='cuda'):
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1
        })
        self.rendering_options.update(rendering_options)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device = device
        
    def render(
            self,
            mesh : MeshExtractResult,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            return_types = ["color", "normal", "nocs", "depth"]
        ) -> edict:
        """
        Render the mesh.

        Args:
            mesh : meshmodel
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            return_types (list): list of return types, can be "mask", "depth", "normal", "color", "nocs"

        Returns:
            edict based on return_types containing:
                color (torch.Tensor): [3, H, W] rendered color image
                depth (torch.Tensor): [H, W] rendered depth image
                normal (torch.Tensor): [3, H, W] rendered normal image in camera space
                mask (torch.Tensor): [H, W] rendered mask image
                nocs (torch.Tensor): [3, H, W] rendered NOCS coordinates
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((1, resolution, resolution, 3), dtype=torch.float32, device=self.device)
            ret_dict = {k : default_img if k in ['normal', 'normal_map', 'color'] else default_img[..., :1] for k in return_types}
            return ret_dict
        
        perspective = intrinsics_to_projection(intrinsics, near, far)
        
        RT = extrinsics.unsqueeze(0)
        full_proj = (perspective @ extrinsics).unsqueeze(0)
        
        vertices = mesh.vertices.unsqueeze(0)

        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        vertices_camera = torch.bmm(vertices_homo, RT.transpose(-1, -2))
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
        faces_int = mesh.faces.int()
        rast, _ = dr.rasterize(
            self.glctx, vertices_clip, faces_int, (resolution * ssaa, resolution * ssaa))
        
        out_dict = edict()
        for type in return_types:
            img = None
            try:
                if type == "mask":
                    img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
                elif type == "depth":
                    img = dr.interpolate(vertices_camera[..., 2:3].contiguous(), rast, faces_int)[0]
                elif type == "normal":
                    # Transform face normals to camera space
                    rotation = RT[..., :3, :3]  # [1, 3, 3]
                    face_normals = mesh.face_normal.view(1, -1, 3)  # [1, N, 3]
                    camera_space_normals = torch.matmul(face_normals, rotation.transpose(-1, -2))
                    camera_space_normals = F.normalize(camera_space_normals, dim=-1)
                    
                    img = dr.interpolate(
                        camera_space_normals.reshape(1, -1, 3), rast,
                        torch.arange(mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int).reshape(-1, 3)
                    )[0]
                    # normalize norm pictures to [0,1] range
                    img = (-img + 1) / 2
                elif type == "color":
                    img = dr.interpolate(mesh.vertex_attrs[:, :3].contiguous(), rast, faces_int)[0]
                    img = dr.antialias(img, rast, vertices_clip, faces_int)
                elif type == "nocs":
                    img = dr.interpolate(vertices[..., :3].contiguous(), rast, faces_int)[0]
                    img = img + 0.5

                if ssaa > 1:
                    if type == 'color':
                        img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
                        img = img.squeeze()
                    else:
                        img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='nearest')
                        img = img.squeeze()
                else:
                    img = img.permute(0, 3, 1, 2).squeeze()
            except Exception as e:
                print(f"Error rendering {type}: {str(e)}")
                # Return a blank image of appropriate shape in case of error
                if type in ['normal', 'color', 'nocs', 'depth']:
                    img = torch.zeros((3, resolution, resolution), dtype=torch.float32, device=self.device)
                else:
                    img = torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device)
                    
            out_dict[type] = img

        return out_dict
