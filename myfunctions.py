import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import numpy as np
from plyfile import PlyData

def MatrixComp(A, B):
    return np.allclose(A, B, rtol=1e-05, atol=1e-08)


def quaternion2rotation(q):
    """
    Convert quaternion to rotation matrix
    Args:
        q: (4, ) =
    Returns:
        R: (3, 3)
    """
    QW, QX, QY, QZ = q[0], q[1], q[2], q[3]
    R = 2 * np.array([[1/2 - (QY**2 + QZ**2), QX*QY - QW*QZ,         QX*QZ + QW*QY        ],
                      [QX*QY + QW*QZ,         1/2 - (QX**2 + QZ**2), QY*QZ - QW*QX        ],
                      [QX*QZ - QW*QY,         QY*QZ + QW*QX,         1/2 - (QX**2 + QY**2)]])
    return R


def loadply(filename, requires_grad=True, device='cuda'):
    plydata = PlyData.read(filename)
    means    = np.array([plydata.elements[0]['x'], plydata.elements[0]['y'], plydata.elements[0]['z']]).T
    normals  = np.array([plydata.elements[0]['nx'], plydata.elements[0]['ny'], plydata.elements[0]['nz']]).T
    opacitys = np.array([plydata.elements[0]['opacity']]).T
    scales   = np.array([plydata.elements[0]['scale_0'], plydata.elements[0]['scale_1'], plydata.elements[0]['scale_2']]).T
    rots     = np.array([plydata.elements[0]['rot_0'], plydata.elements[0]['rot_1'], plydata.elements[0]['rot_2'], plydata.elements[0]['rot_3']]).T
    f_dcs    = np.array([plydata.elements[0]['f_dc_0'], plydata.elements[0]['f_dc_1'], plydata.elements[0]['f_dc_2']]).T
    # f_rests  = np.array([plydata.elements[0]['f_rest_0'], plydata.elements[0]['f_rest_1'], plydata.elements[0]['f_rest_2'], plydata.elements[0]['f_rest_3'], plydata.elements[0]['f_rest_4'], plydata.elements[0]['f_rest_5'], plydata.elements[0]['f_rest_6'], plydata.elements[0]['f_rest_7'], plydata.elements[0]['f_rest_8'], plydata.elements[0]['f_rest_9'], plydata.elements[0]['f_rest_10'], plydata.elements[0]['f_rest_11'], plydata.elements[0]['f_rest_12'], plydata.elements[0]['f_rest_13'], plydata.elements[0]['f_rest_14'], plydata.elements[0]['f_rest_15'], plydata.elements[0]['f_rest_16'], plydata.elements[0]['f_rest_17'], plydata.elements[0]['f_rest_18'], plydata.elements[0]['f_rest_19'], plydata.elements[0]['f_rest_20'], plydata.elements[0]['f_rest_21'], plydata.elements[0]['f_rest_22'], plydata.elements[0]['f_rest_23'], plydata.elements[0]['f_rest_24'], plydata.elements[0]['f_rest_25'], plydata.elements[0]['f_rest_26'], plydata.elements[0]['f_rest_27'], plydata.elements[0]['f_rest_28'], plydata.elements[0]['f_rest_29'], plydata.elements[0]['f_rest_30'], plydata.elements[0]['f_rest_31'], plydata.elements[0]['f_rest_32'], plydata.elements[0]['f_rest_33'], plydata.elements[0]['f_rest_34'], plydata.elements[0]['f_rest_35'], plydata.elements[0]['f_rest_36'], plydata.elements[0]['f_rest_37'], plydata.elements[0]['f_rest_38'], plydata.elements[0]['f_rest_39'], plydata.elements[0]['f_rest_40'], plydata.elements[0]['f_rest_41'], plydata.elements[0]['f_rest_42'], plydata.elements[0]['f_rest_43'], plydata.elements[0]['f_rest_44']]).T
    f_rests = np.array([plydata.elements[0][f"f_rest_{i}"] for i in range(45)]).T
    f_rests = f_rests.reshape(-1, 3, 15).transpose(0, 2, 1)
    means    = torch.tensor(means, dtype=torch.float32)
    normals  = torch.tensor(normals, dtype=torch.float32)
    opacitys = torch.tensor(opacitys, dtype=torch.float32)
    scales   = torch.tensor(scales, dtype=torch.float32)
    rots     = torch.tensor(rots, dtype=torch.float32)
    f_dcs    = torch.tensor(f_dcs, dtype=torch.float32).reshape(-1, 1, 3)
    # f_rests  = torch.tensor(f_rests, dtype=torch.float32).reshape(-1, 15, 3)
    f_rests  = torch.tensor(f_rests, dtype=torch.float32)
    if requires_grad:
        means.requires_grad    = True
        normals.requires_grad  = True
        opacitys.requires_grad = True
        scales.requires_grad   = True
        rots.requires_grad     = True
        f_dcs.requires_grad    = True
        f_rests.requires_grad  = True

    # to device
    means    = means.to(device)
    normals  = normals.to(device)
    opacitys = opacitys.to(device)
    scales   = scales.to(device)
    rots     = rots.to(device)
    f_dcs    = f_dcs.to(device)
    f_rests  = f_rests.to(device)

    return means, normals, opacitys, scales, rots, f_dcs, f_rests



def cull_and_project_point(mean3D, projmatrix, viewmatrix):
    p_origin_homo = torch.cat([mean3D, torch.ones((1, 1), device='cuda:0')], dim=1)
    p_view = p_origin_homo @ viewmatrix
    p_view = p_view[:, :3]

    # culling
    if (p_view[0, 2] < 0.2):
        return True, torch.zeros(1,3), torch.zeros(1,3)

    p_hom         = p_origin_homo @ projmatrix
    p_w = 1.0 / (p_hom[0, 3] + 0.0000001)
    mean2D = p_hom[:, :3] * p_w
    return False, p_view, mean2D

def computeCov3D(scale: torch.Tensor, scale_modifier: float, rot: torch.Tensor) -> torch.Tensor:
    """
    Compute a 3D covariance matrix from a scaling vector and a quaternion rotation.
    
    Parameters:
      scale (torch.Tensor): Tensor of shape [1, 3].
      scale_modifier (float): A scaling modifier.
      rot (torch.Tensor): Tensor of shape [1, 4] representing a quaternion (r, x, y, z).
      
    Returns:
      torch.Tensor: A tensor of shape [1, 6] containing the upper triangular elements
                    of the symmetric covariance matrix:
                    [Sigma[0,0], Sigma[0,1], Sigma[0,2], Sigma[1,1], Sigma[1,2], Sigma[2,2]]
    """
    device = scale.device
    dtype = scale.dtype
    # Create a 3x3 scaling matrix S (diagonal matrix)
    S = torch.eye(3, device=device, dtype=dtype)
    S[0, 0] = scale_modifier * scale[0, 0]
    S[1, 1] = scale_modifier * scale[0, 1]
    S[2, 2] = scale_modifier * scale[0, 2]
    # Use the quaternion directly (assumed order: (r, x, y, z))
    q = rot[0]  # q has shape (4,)
    r, x, y, z = q[0], q[1], q[2], q[3]
    # Compute the rotation matrix R from the quaternion
    R = torch.empty((3, 3), device=device, dtype=dtype)
    R[0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[0, 1] = 2.0 * (x * y - r * z)
    R[0, 2] = 2.0 * (x * z + r * y)
    R[1, 0] = 2.0 * (x * y + r * z)
    R[1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[1, 2] = 2.0 * (y * z - r * x)
    R[2, 0] = 2.0 * (x * z - r * y)
    R[2, 1] = 2.0 * (y * z + r * x)
    R[2, 2] = 1.0 - 2.0 * (x * x + y * y)
    # Compute the combined matrix M = S * R
    # M = torch.matmul(S, R)
    M = torch.matmul(R, S)
    # Compute the covariance matrix Sigma = transpose(M) * M
    Sigma = torch.matmul(M, M.t())
    # Extract the upper triangular part (order: [Sigma00, Sigma01, Sigma02, Sigma11, Sigma12, Sigma22])
    cov_elements = torch.empty(6, device=device, dtype=dtype)
    cov_elements[0] = Sigma[0, 0]
    cov_elements[1] = Sigma[0, 1]
    cov_elements[2] = Sigma[0, 2]
    cov_elements[3] = Sigma[1, 1]
    cov_elements[4] = Sigma[1, 2]
    cov_elements[5] = Sigma[2, 2]
    return cov_elements.unsqueeze(0)


def computeCov2D(mean3D: torch.Tensor, focal_x: float, focal_y: float,
                 tan_fovx: float, tan_fovy: float,
                 cov3D: torch.Tensor, viewmatrix: torch.Tensor) -> torch.Tensor:
    """
    Computes the 2D screen-space covariance from the 3D covariance.
    
    Parameters:
      mean3D: Tensor of shape (1,3) representing the 3D point.
      focal_x, focal_y: Focal lengths.
      tan_fovx, tan_fovy: Tangents of half the FOV in x and y.
      cov3D: Tensor of shape (1,6) containing the unique elements of the symmetric 3x3 covariance matrix:
             [v00, v01, v02, v11, v12, v22].
      viewmatrix: Tensor of shape (4,4) representing the view transformation.
      
    Returns:
      A tensor of shape (1,3) containing [cov_xx, cov_xy, cov_yy] for the 2D covariance.
    """
    device = mean3D.device
    dtype = mean3D.dtype

    # Transform mean3D to camera space.
    # Append a 1 to make it homogeneous.
    p_origin_homo = torch.cat([mean3D, torch.ones((1, 1), dtype=dtype, device=device)], dim=1)  # (1,4)
    t_homo = torch.matmul(p_origin_homo, viewmatrix)  # (1,4)
    t = t_homo[:, :3]  # (1,3)

    # Extract t components (as scalars)
    t_x = t[0, 0]
    t_y = t[0, 1]
    t_z = t[0, 2]

    # Clamp the normalized x and y values according to limits.
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t_x / t_z
    tytz = t_y / t_z
    clamped_txtz = torch.clamp(txtz, min=-limx, max=limx)
    clamped_tytz = torch.clamp(tytz, min=-limy, max=limy)
    # Multiply back by t_z to get corrected t.x and t.y.
    t_x = clamped_txtz * t_z
    t_y = clamped_tytz * t_z
    # (We still use t_z as is.)

    # Build the 3x3 Jacobian J.
    # Note: the third row is zeros.
    J = torch.tensor([
            [focal_x / t_z,         0.0, - (focal_x * t_x) / (t_z * t_z)],
            [0.0,         focal_y / t_z, - (focal_y * t_y) / (t_z * t_z)],
            [0.0,         0.0,                   0.0]
        ], dtype=dtype, device=device)
    
    # Extract the rotation part from viewmatrix.
    # In the original C++ code (using GLM in column-major order),
    # the 3x3 matrix W is constructed as:
    #   [ viewmatrix[0], viewmatrix[4], viewmatrix[8];
    #     viewmatrix[1], viewmatrix[5], viewmatrix[9];
    #     viewmatrix[2], viewmatrix[6], viewmatrix[10] ]
    # For a torch viewmatrix (assumed row-major), the upper-left 3x3 is viewmatrix[:3, :3].
    # Taking its transpose gives the same ordering.
    W = viewmatrix[:3, :3].T  # (3,3)

    # Compute T = W * J.
    T = torch.matmul(J, W)  # (3,3)

    # Step 6: Construct the 3D covariance matrix Vrk from cov3D.
    # cov3D is given as [v00, v01, v02, v11, v12, v22]
    v00 = cov3D[0, 0]
    v01 = cov3D[0, 1]
    v02 = cov3D[0, 2]
    v11 = cov3D[0, 3]
    v12 = cov3D[0, 4]
    v22 = cov3D[0, 5]
    Vrk = torch.tensor([
            [v00, v01, v02],
            [v01, v11, v12],
            [v02, v12, v22]
        ], dtype=dtype, device=device)
    # === Step 7: Project the 3D covariance to 2D ===
    # Although the C code computes: cov = transpose(T) * transpose(Vrk) * T (i.e. Tᵀ * Vrk * T),
    # our rough estimates show that using:
    #     cov = T * Vrk * Tᵀ
    # produces values that match the C output.
    cov = torch.matmul(T, torch.matmul(Vrk, T.T))
    # print("cov:", cov)

    # Extract the upper-left 2x2 block (unique elements)
    cov_xx = cov[0, 0]
    cov_xy = cov[0, 1]
    cov_yy = cov[1, 1]
    cov2D = torch.stack([cov_xx, cov_xy, cov_yy]).unsqueeze(0)  # shape (1,3)
    
    return cov2D


def project_points(means, normals, opacitys, scales, rots, f_dcs, f_rests, projmatrix, viewmatrix, focal_x, focal_y, tan_fovx, tan_fovy):
    # pass
    N = means.shape[0]
    for idx in range(N):
        _, p_view, mean2D = cull_and_project_point(means[idx, :].view(-1, 3), projmatrix, viewmatrix)
        conv3D            = computeCov3D(scales[idx, :].view(-1, 3), 1.0, rots[idx, :].view(-1, 4))
        conv2D            = computeCov2D(means[idx, :].view(-1, 3), focal_x, focal_y, tan_fovx, tan_fovy, conv3D, viewmatrix)
        
        # culling and projection
        # print("culling: ", _)
        # print("p_view ", p_view)
        # print("mean2D ", mean2D)

        # compute 3D covariance matrix
        # scale = scales[idx, :].view(-1, 3)
        # rot   = rots[idx, :].view(-1, 4)
        # print("scale: ", scale)
        # print("rot: ", rot)
        # print("scale.shape:", scale.shape)
        # print("rot.shape ", rot.shape)
        # print("conv3D: ", conv3D)

        # compute 2D covariance matrix
        # float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
        # mean3D  = means[idx, :].view(-1, 3)
        # print("mean3D: ", mean3D)
        # print("focal_x: ", focal_x)
        # print("focal_y: ", focal_y)
        # print("tan_fovx: ", tan_fovx)
        # print("tan_fovy: ", tan_fovy)
        # print("conv3D: ", conv3D)
        # print("viewmatrix: ", viewmatrix)
        print("conv2D: ", conv2D)


