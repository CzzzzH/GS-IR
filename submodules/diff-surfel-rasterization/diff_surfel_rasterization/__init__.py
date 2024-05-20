#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    albedo,
    roughness,
    metallic,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    
    # print("C++ Forward")
    # print("means3D", means3D.shape, means3D.dtype)
    # print("means2D", means2D.shape, means2D.dtype)
    # print("sh", sh.shape, sh.dtype)
    # print("colors_precomp", colors_precomp.shape, colors_precomp.dtype)
    # print("opacities", opacities.shape, opacities.dtype)
    # print("albedo", albedo.shape, albedo.dtype)
    # print("roughness", roughness.shape, roughness.dtype)
    # print("metallic", metallic.shape, metallic.dtype)
    # print("scales", scales.shape, scales.dtype)
    # print("rotations", rotations.shape, rotations.dtype)
    # print("cov3Ds_precomp", cov3Ds_precomp.shape, cov3Ds_precomp.dtype)
    
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        albedo,
        roughness,
        metallic,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    inference : bool

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        albedo,
        roughness,
        metallic,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            albedo,
            roughness,
            metallic,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.inference,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, albedo_map, roughness_map, metallic_map, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, albedo_map, roughness_map, metallic_map, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, albedo_map, roughness_map, metallic_map, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, albedo_map, roughness_map, metallic_map, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_out_albedo, grad_out_roughness, grad_out_metallic, grad_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, albedo, roughness, metallic, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # print(f"num_rendered {num_rendered}")
        # print(f"grad_out_color {grad_out_color.shape} {grad_out_color.min()} {grad_out_color.max()} {grad_out_color.mean()} {grad_out_color.std()}")
        # grad_depth_no_nan = torch.where(torch.isnan(grad_depth), torch.zeros_like(grad_depth), grad_depth)
        # print(f"grad_depth {grad_depth_no_nan.shape} {grad_depth_no_nan.min()} {grad_depth_no_nan.max()} {grad_depth_no_nan.mean()} {grad_depth_no_nan.std()}")
        # print(f"colors_precomp {colors_precomp}")
        # print(f"means3D {means3D.shape} {means3D.min()} {means3D.max()} {means3D.mean()} {means3D.std()}")
        # print(f"scales {scales.shape} {scales.min()} {scales.max()} {scales.mean()} {scales.std()}")
        # print(f"rotations {rotations.shape} {rotations.min()} {rotations.max()} {rotations.mean()} {rotations.std()}")
        # print(f"radii {radii.shape} {radii.min()} {radii.max()} {radii.float().mean()} {radii.float().std()}")
        # print(f"sh {sh.shape} {sh.min()} {sh.max()} {sh.mean()} {sh.std()}")
        # print(f"geomBuffer {geomBuffer.shape} {geomBuffer.min()} {geomBuffer.max()} {geomBuffer.float().mean()} {geomBuffer.float().std()}")
        # print(f"binningBuffer {binningBuffer.shape} {binningBuffer.min()} {binningBuffer.max()} {binningBuffer.float().mean()} {binningBuffer.float().std()}")
        # print(f"imgBuffer {imgBuffer.shape} {imgBuffer.min()} {imgBuffer.max()} {imgBuffer.float().mean()} {imgBuffer.float().std()}")
        
        # print("Rasterize Settings:")
        # print(f"bg {raster_settings.bg.shape} {raster_settings.bg.min()} {raster_settings.bg.max()} {raster_settings.bg.mean()} {raster_settings.bg.std()}")
        # print(f"scale_modifier {raster_settings.scale_modifier}")
        # print(f"viewmatrix {raster_settings.viewmatrix.min()} {raster_settings.viewmatrix.max()} {raster_settings.viewmatrix.mean()} {raster_settings.viewmatrix.std()}")
        # print(f"projmatrix {raster_settings.projmatrix.min()} {raster_settings.projmatrix.max()} {raster_settings.projmatrix.mean()} {raster_settings.projmatrix.std()}")
        # print(f"tanfovx {raster_settings.tanfovx}")
        # print(f"tanfovy {raster_settings.tanfovy}")
        # print(f"sh_degree {raster_settings.sh_degree}")
        # print(f"campos {raster_settings.campos.shape} {raster_settings.campos.min()} {raster_settings.campos.max()} {raster_settings.campos.mean()} {raster_settings.campos.std()}")

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp,
                albedo,
                roughness,
                metallic,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_albedo,
                grad_out_roughness,
                grad_out_metallic,
                grad_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_albedo, grad_roughness, grad_metallic, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_albedo, grad_roughness, grad_metallic, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_albedo,
            grad_roughness,
            grad_metallic,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )
        
        # print(f"grad_means3D {grad_means3D.shape} {grad_means3D.min()} {grad_means3D.max()} {grad_means3D.mean()} {grad_means3D.std()}")
        # print(f"grad_means2D {grad_means2D.shape} {grad_means2D.min()} {grad_means2D.max()} {grad_means2D.mean()} {grad_means2D.std()}")
        # print(f"grad_sh {grad_sh.shape} {grad_sh.min()} {grad_sh.max()} {grad_sh.mean()} {grad_sh.std()}")
        # print(f"grad_colors_precomp {grad_colors_precomp.shape} {grad_colors_precomp.min()} {grad_colors_precomp.max()} {grad_colors_precomp.mean()} {grad_colors_precomp.std()}")
        # print(f"grad_opacities {grad_opacities.shape}  {grad_opacities.min()} {grad_opacities.max()} {grad_opacities.mean()} {grad_opacities.std()}")
        # print(f"grad_albedo {grad_albedo.shape} {grad_albedo.min()} {grad_albedo.max()} {grad_albedo.mean()} {grad_albedo.std()}")
        # print(f"grad_roughness {grad_roughness.shape} {grad_roughness.min()} {grad_roughness.max()} {grad_roughness.mean()} {grad_roughness.std()}")
        # print(f"grad_metallic {grad_metallic.shape} {grad_metallic.min()} {grad_metallic.max()} {grad_metallic.mean()} {grad_metallic.std()}")
        # print(f"grad_scales {grad_scales.shape} {grad_scales.min()} {grad_scales.max()} {grad_scales.mean()} {grad_scales.std()}")
        # print(f"grad_rotations {grad_rotations.shape} {grad_rotations.min()} {grad_rotations.max()} {grad_rotations.mean()} {grad_rotations.std()}")
        # print(f"grad_cov3Ds_precomp {grad_cov3Ds_precomp.shape} {grad_cov3Ds_precomp.min()} {grad_cov3Ds_precomp.max()} {grad_cov3Ds_precomp.mean()} {grad_cov3Ds_precomp.std()}")
        # grads.stop()

        return grads
    
class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, albedo, roughness, metallic, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()

        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).cuda()
        
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            albedo,
            roughness,
            metallic,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )
