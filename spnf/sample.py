import torch

def rand_ortho(dim, size, device='cuda', dtype=torch.float32, use_analytic=True):
    """
    Generates random orthonormal matrices (O(N) or SO(N)).
    
    Args:
        dim (int): Dimension N (2, 3, or >3).
        size (int): Batch size.
        device (str): Device to generate on.
        use_analytic (bool): If True, uses fast analytic formulas for 2D/3D.
                             (Note: Analytic 2D/3D generates SO(N) rotations. 
                              For covariance matrices Σ=RΛR^T, SO(N) == O(N)).
    """
    # 1. Fast Analytic 2D (Rotation Angle)
    if use_analytic and dim == 2:
        theta = torch.rand(size, device=device, dtype=dtype) * 2 * torch.pi
        c, s = torch.cos(theta), torch.sin(theta)
        # Construct [[c, -s], [s, c]]
        row1 = torch.stack([c, -s], dim=1)
        row2 = torch.stack([s,  c], dim=1)
        return torch.stack([row1, row2], dim=1)

    # 2. Fast Analytic 3D (Quaternions)
    elif use_analytic and dim == 3:
        # Sample random quaternions (4D Gaussian) and normalize
        q = torch.randn(size, 4, device=device, dtype=dtype)
        q = q / q.norm(dim=1, keepdim=True)
        r, i, j, k = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Quaternion to Matrix conversion
        return torch.stack([
            1 - 2*(j**2 + k**2),     2*(i*j - k*r),     2*(i*k + j*r),
                2*(i*j + k*r), 1 - 2*(i**2 + k**2),     2*(j*k - i*r),
                2*(i*k - j*r),     2*(j*k + i*r), 1 - 2*(i**2 + j**2)
        ], dim=1).reshape(size, 3, 3)

    # 3. Generic ND (QR Decomposition with Mezzadri correction)
    else:
        H = torch.randn(size, dim, dim, device=device, dtype=dtype)
        Q, R = torch.linalg.qr(H)
        # Fix signs to ensure uniformity (Mezzadri algorithm)
        signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
        return Q * signs.unsqueeze(-2)


def logrand(log_start, log_end, size, device='cuda', dtype=torch.float32):
    """
    Samples values uniformly in log-space between 10^log_start and 10^log_end
    """
    return torch.pow(
        10, torch.rand(size, device=device, dtype=dtype) * (log_end - log_start) + log_start)
    

def construct_covariance(q_matrix, eigenvalues):
    """
    Constructs Σ = QΛQ^T efficiently using broadcasting.
    
    Args:
        q_matrix: (B, D, D) orthonormal matrix
        eigenvalues: (B, D) variances
    """
    # 1. Scale columns of Q by eigenvalues (Q @ Λ)
    # q_matrix is (B, D, D)
    # eigenvalues.unsqueeze(1) is (B, 1, D)
    # Broadcasting (B, D, D) * (B, 1, D) applies element-wise multiplication
    # to every row, effectively scaling each column j by eigenvalue j.
    q_scaled = q_matrix * eigenvalues.unsqueeze(1)
    
    # 2. Multiply by Q^T
    # (B, D, D) @ (B, D, D)
    sigma = torch.bmm(q_scaled, q_matrix.transpose(1, 2))
    
    return sigma


def sample_gaussian_delta(q_matrix, eigenvalues):
    """
    Samples delta values from a Gaussian distribution defined by covariance Σ = QΛQ^T.
    
    Args:
        q_matrix (torch.Tensor): Orthonormal rotation matrices Q. Shape (B, D, D).
        eigenvalues (torch.Tensor): Principal variances Λ (diagonal elements). Shape (B, D).
        
    Returns:
        torch.Tensor: Delta values Δx with shape (B, D).
    """
    # 1. Sample standard normal noise u ~ N(0, I)
    # Shape: (B, D, 1) to allow for batch matrix multiplication later
    u = torch.randn(q_matrix.shape[0], q_matrix.shape[1], 1, 
                    device=q_matrix.device, dtype=q_matrix.dtype)
    
    # 2. Scale by standard deviation (sqrt of eigenvalues)
    # eigenvalues shape is (B, D), we unsqueeze to (B, D, 1) for broadcasting
    # z = Λ^(1/2) * u
    # Note: Ensure eigenvalues are variances (σ^2). If they are standard deviations (σ), remove sqrt.
    scale = torch.sqrt(eigenvalues).unsqueeze(-1)
    z = u * scale
    
    # 3. Rotate into world frame: Δx = Q * z
    # (B, D, D) @ (B, D, 1) -> (B, D, 1)
    delta = torch.bmm(q_matrix, z)
    
    # Remove the last singleton dimension to return (B, D)
    return delta.squeeze(-1)


def sample_ellipsoid_delta(q_matrix, eigenvalues):
    """
    Samples delta values uniformly from an ellipsoid defined by x^T Σ^-1 x <= 1,
    where Σ = QΛQ^T.
    
    Args:
        q_matrix (torch.Tensor): Orthonormal rotation matrices Q. Shape (B, D, D).
        eigenvalues (torch.Tensor): Principal variances Λ. Shape (B, D).
        
    Returns:
        torch.Tensor: Delta values Δx with shape (B, D).
    """
    batch_size, dim, _ = q_matrix.shape
    device = q_matrix.device
    dtype = q_matrix.dtype

    # 1. Sample random directions on the unit sphere
    # Gaussian sampling + normalization gives uniform distribution on sphere surface
    u = torch.randn(batch_size, dim, 1, device=device, dtype=dtype)
    u = u / torch.norm(u, dim=1, keepdim=True)
    
    # 2. Sample random radius for uniform volume density
    # PDF(r) ~ r^(d-1) -> CDF(r) ~ r^d -> Inverse CDF sample: r = U^(1/d)
    random_u = torch.rand(batch_size, 1, 1, device=device, dtype=dtype)
    radius = torch.pow(random_u, 1.0 / dim)
    
    # Combine to get uniform samples inside the unit ball
    z_unit = u * radius
    
    # 3. Scale by the ellipsoid axes lengths (sqrt of eigenvalues)
    # The ellipsoid boundary is defined by axes of length sqrt(lambda)
    scale = torch.sqrt(eigenvalues).unsqueeze(-1)
    z_scaled = z_unit * scale
    
    # 4. Rotate into world frame: Δx = Q * z_scaled
    delta = torch.bmm(q_matrix, z_scaled)
    
    return delta.squeeze(-1)

def get_deltas(q_matrix, eigenvalues, method='gaussian'):
    """
    Wrapper to get delta samples using specified method.
    """
    if method == 'gaussian':
        return sample_gaussian_delta(q_matrix, eigenvalues)
    elif method == 'uniform_ellipsoid':
        return sample_ellipsoid_delta(q_matrix, eigenvalues)
    else:
        raise ValueError(f"Unknown sampling method: {method}")