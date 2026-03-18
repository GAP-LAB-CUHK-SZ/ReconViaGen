import torch
from easydict import EasyDict as edict
from typing import Tuple, Optional
from diso import DiffDMC
from .cube2mesh import MeshExtractResult
from .utils_cube import *
from ...modules.sparse import SparseTensor

class EnhancedMarchingCubes:
    def __init__(self, device="cuda"):
        self.device = device
        self.diffdmc = DiffDMC(dtype=torch.float32)

    def __call__(self,
                 voxelgrid_vertices: torch.Tensor,
                 scalar_field: torch.Tensor,
                 voxelgrid_colors: Optional[torch.Tensor] = None,
                 training: bool = False
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Enhanced Marching Cubes implementation using DiffDMC that handles deformations and colors
        """
        if scalar_field.dim() == 1:
            grid_size = int(round(scalar_field.shape[0] ** (1 / 3)))
            scalar_field = scalar_field.reshape(grid_size, grid_size, grid_size)
        elif scalar_field.dim() > 3:
            scalar_field = scalar_field.squeeze()

        if scalar_field.dim() != 3:
            raise ValueError(f"Expected 3D array, got shape {scalar_field.shape}")

        # Normalize coordinates for DiffDMC
        scalar_field = scalar_field.to(self.device)
        
        # Get deformation field if provided
        deform_field = None
        if voxelgrid_vertices is not None:
            if voxelgrid_vertices.dim() == 2:
                grid_size = int(round(voxelgrid_vertices.shape[0] ** (1 / 3)))
                voxelgrid_vertices = voxelgrid_vertices.reshape(grid_size, grid_size, grid_size, 3)
            deform_field = voxelgrid_vertices.to(self.device)

        # Run DiffDMC
        vertices, faces = self.diffdmc(
            scalar_field,
            deform_field,
            isovalue=0.0
        )

        # Handle colors if provided
        colors = None
        if voxelgrid_colors is not None:
            voxelgrid_colors = torch.sigmoid(voxelgrid_colors)
            if voxelgrid_colors.dim() == 2:
                grid_size = int(round(voxelgrid_colors.shape[0] ** (1/3)))
                voxelgrid_colors = voxelgrid_colors.reshape(grid_size, grid_size, grid_size, -1)

            grid_positions = vertices.clone() * grid_size
            grid_coords = grid_positions.long()
            local_coords = grid_positions - grid_coords.float()
            
            # Clamp coordinates to grid bounds
            grid_coords = torch.clamp(grid_coords, 0, voxelgrid_colors.shape[0] - 1)
            
            # Trilinear interpolation for colors
            colors = self._interpolate_color(grid_coords, local_coords, voxelgrid_colors)
            
        vertices = vertices * 2 - 1 # Normalize vertices to [-1, 1]
        vertices /= 2.0  # Normalize vertices to [-0.5, 0.5]

        # Compute deviation loss for training
        deviation_loss = torch.tensor(0.0, device=self.device)
        if training and deform_field is not None:
            # Compute deviation between original and deformed vertices
            deviation_loss = self._compute_deviation_loss(vertices, deform_field)

        # faces = faces.flip(dims=[1])  # Maintain consistent face orientation

        return vertices, faces, deviation_loss, colors

    def _interpolate_color(self, grid_coords: torch.Tensor,
                          local_coords: torch.Tensor,
                          color_field: torch.Tensor) -> torch.Tensor:
        """
        Interpolate colors using trilinear interpolation
        Args:
            grid_coords: (N, 3) integer grid coordinates
            local_coords: (N, 3) fractional positions within grid cells
            color_field: (res, res, res, C) color values
        """
        x, y, z = local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]
        
        # Get corner values for each vertex
        c000 = color_field[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]]
        c001 = color_field[grid_coords[:, 0], grid_coords[:, 1],
               torch.clamp(grid_coords[:, 2] + 1, 0, color_field.shape[2] - 1)]
        c010 = color_field[grid_coords[:, 0],
               torch.clamp(grid_coords[:, 1] + 1, 0, color_field.shape[1] - 1),
               grid_coords[:, 2]]
        c011 = color_field[grid_coords[:, 0],
               torch.clamp(grid_coords[:, 1] + 1, 0, color_field.shape[1] - 1),
               torch.clamp(grid_coords[:, 2] + 1, 0, color_field.shape[2] - 1)]
        c100 = color_field[torch.clamp(grid_coords[:, 0] + 1, 0, color_field.shape[0] - 1),
               grid_coords[:, 1], grid_coords[:, 2]]
        c101 = color_field[torch.clamp(grid_coords[:, 0] + 1, 0, color_field.shape[0] - 1),
               grid_coords[:, 1],
               torch.clamp(grid_coords[:, 2] + 1, 0, color_field.shape[2] - 1)]
        c110 = color_field[torch.clamp(grid_coords[:, 0] + 1, 0, color_field.shape[0] - 1),
               torch.clamp(grid_coords[:, 1] + 1, 0, color_field.shape[1] - 1),
               grid_coords[:, 2]]
        c111 = color_field[torch.clamp(grid_coords[:, 0] + 1, 0, color_field.shape[0] - 1),
               torch.clamp(grid_coords[:, 1] + 1, 0, color_field.shape[1] - 1),
               torch.clamp(grid_coords[:, 2] + 1, 0, color_field.shape[2] - 1)]
        
        # Interpolate along x axis
        c00 = c000 * (1 - x)[:, None] + c100 * x[:, None]
        c01 = c001 * (1 - x)[:, None] + c101 * x[:, None]
        c10 = c010 * (1 - x)[:, None] + c110 * x[:, None]
        c11 = c011 * (1 - x)[:, None] + c111 * x[:, None]
        
        # Interpolate along y axis
        c0 = c00 * (1 - y)[:, None] + c10 * y[:, None]
        c1 = c01 * (1 - y)[:, None] + c11 * y[:, None]
        
        # Interpolate along z axis
        colors = c0 * (1 - z)[:, None] + c1 * z[:, None]
        
        return colors

    def _compute_deviation_loss(self, vertices: torch.Tensor,
                               deform_field: torch.Tensor) -> torch.Tensor:
        """Compute deviation loss for training"""
        # Since DiffDMC already handles the deformation, we compute the loss
        # based on the magnitude of the deformation field
        return torch.mean(deform_field ** 2)

class SparseFeatures2MCMesh:
    def __init__(self, device="cuda", res=128, use_color=True):
        super().__init__()
        self.device = device

        self.res = res

        self.mesh_extractor = EnhancedMarchingCubes(device=device)
        self.sdf_bias = -1.0 / res
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)
        self.reg_v = verts.to(self.device)
        self.use_color = use_color
        self._calc_layout()

    def _calc_layout(self):
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if self.use_color:
            '''
            6 channel color including normal map
            '''
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start

    def get_layout(self, feats: torch.Tensor, name: str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name][
            'shape'])

    def __call__(self, cubefeats: SparseTensor, training=False):
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats

        sdf, deform, color, weights = [self.get_layout(feats, name)
                                       for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1),
                                                     training=training)

        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res + 1, sdf_init=True)

        if self.use_color:
            sdf_d, deform_d, colors_d = (v_attrs_d[..., 0], v_attrs_d[..., 1:4],
                                         v_attrs_d[..., 4:])
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None

        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)

        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            voxelgrid_colors=colors_d,
            training=training
        )

        mesh = MeshExtractResult(vertices=vertices, faces=faces,
                                 vertex_attrs=colors, res=self.res)

        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:, :20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]

        return mesh