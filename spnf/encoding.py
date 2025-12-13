from itertools import product
import math
from math import pi, gamma
import numpy as np
from scipy.integrate import quad

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import HalfNormal
from torch.quasirandom import SobolEngine

try:
    from torch._dynamo import disable as _dynamo_disable
except Exception:  # PyTorch < 2.0 or no dynamo
    def _dynamo_disable(fn): return fn


# --- core integral: no caching, pure Python/NumPy/SciPy ---
def lanczos_radial_mass(a, d, R=None, detSigma=1.0, absolute=True, tol=1e-9):
    """Returns (value, error) of |Σ|^{1/2} S_{d-1} ∫_0^R [|]sinc(r)sinc(r/a)[|] r^{d-1} dr."""
    R = 2*np.sqrt(a) if R is None else float(R)
    S = 2 * pi**(d/2) / gamma(d/2)               # S_{d-1}
    f = (lambda r: abs(np.sinc(r)*np.sinc(r/a))*r**(d-1)) if absolute \
        else (lambda r: (np.sinc(r)*np.sinc(r/a))*r**(d-1))
    val, err = quad(f, 0.0, R, epsabs=tol, epsrel=tol, limit=1000)
    scale = np.sqrt(detSigma) * S
    return scale*val, scale*err

# --- small cache + JIT/Dynamo-safe wrapper ---
LANCZOS_CACHE = {}  # key -> float (cached value)

@torch.jit.ignore           # TorchScript won't try to compile; calls stay in Python
@_dynamo_disable            # torch.compile won't trace/compile this function
def get_lanczos_mass(a, d, R=None, detSigma=1.0, absolute=True, tol=1e-9):
    """Cached Python-side accessor: returns a float; never traced/compiled by JIT/torch.compile."""
    key = (float(a), int(d), None if R is None else float(R),
           float(detSigma), bool(absolute), float(tol))
    if key not in LANCZOS_CACHE:
        LANCZOS_CACHE[key] = float(lanczos_radial_mass(a, d, R, detSigma, absolute, tol)[0])
    return LANCZOS_CACHE[key]


class FourierEncoding(nn.Module):
    """
    Code taken from Neural Gaussian Scale Space Fields (NGSSF) repository and modified slightly.
    Random Fourier Features encoding from:
    M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron,
    R. Ng. "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains."
    Advances in Neural Information Processing Systems (NeurIPS). 2020.

    Cube-to-ball mapping from:
    J. A. Griepentrog, W. Höppner, H. C. Kaiser, J. Rehberg.
    "A bi-Lipschitz continuous, volume preserving map from the unit ball onto a cube." Note di Matematica, 28(1). 2008.
    """
    def __init__(
        self,
        coords: int,
        embed: int,
        noise: str = "sobol",
        length_distribution: str = "folded_normal",
        length_distribution_param: float = 20,
        amplitude: float = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.kwargs = kwargs

        half_embed = embed // 2
        if noise == "white":
            directions = F.normalize(torch.randn(half_embed, coords))
            lengths_01 = torch.rand(half_embed, 1)
        elif noise == "sobol":
            drawn = half_embed
            if coords > 3:
                # Just a heuristic on how many more samples we need for our rejection sampling approach.
                drawn *= 2 ** (coords - 1)
            while True:
                directions = SobolEngine(coords, scramble=True).draw(drawn) * 2 - 1
                if (directions != 0).any(dim=1).all():
                    break
            if coords <= 3:
                self._cube_to_ball(directions)
            else:
                directions = directions[directions.square().sum(dim=1) <= 1][:half_embed]
            lengths_01 = directions.norm(dim=1, keepdim=True)
            directions /= lengths_01
            # This squaring makes lengths_01 uniformly distributed (instead of "triangularly").
            lengths_01.square_()
        else:
            raise ValueError(f"Unknown noise: {noise}")

        if length_distribution == "uniform":
            A = directions * (lengths_01 * length_distribution_param)
        elif length_distribution == "folded_normal":
            A = directions * HalfNormal(math.sqrt(length_distribution_param)).icdf(lengths_01)
        else:
            raise ValueError(f"Unknown length distribution: {length_distribution}")

        self.register_buffer("A", A)
        self.register_buffer("amplitude", torch.tensor(amplitude))

    @staticmethod
    def _cube_to_ball(X):
        # Notice that we can ignore the special 0-point case because that's already avoided by the sampling code.
        N, D = X.shape
        for i, d in product(range(N), range(1, D)):
            xi_len_sq = X[i, :d].square().sum().item()
            eta = X[i, d].item()
            eta_sq = eta * eta
            not_in_cone = eta_sq <= xi_len_sq
            if d == 1:
                xi = X[i, 0].item()
                if not_in_cone:
                    a = (math.pi * eta) / (4 * xi)
                    X[i, 0] = xi * math.cos(a)
                    X[i, 1] = xi * math.sin(a)
                else:
                    a = (math.pi * xi) / (4 * eta)
                    X[i, 0] = eta * math.sin(a)
                    X[i, 1] = eta * math.cos(a)
            elif d == 2:
                if not_in_cone:
                    X[i, :2] *= math.sqrt(1 - (4 * eta_sq) / (9 * xi_len_sq))
                    X[i, 2] = (2 / 3) * eta
                else:
                    X[i, :2] *= math.sqrt(2 / 3 - xi_len_sq / (9 * eta_sq))
                    X[i, 2] = eta - xi_len_sq / (3 * eta)
            else:
                raise ValueError(f"The cube to ball mapping does not support {D}-d yet.")
        return X

    def forward(self, X):
        M = (X @ self.A.T) * (2 * torch.pi)
        return torch.cat([M.cos(), M.sin()], dim=-1) * self.amplitude

    def extra_repr(self) -> str:
        return f"coords={self.A.shape[1]}, embed={2 * self.A.shape[0]}, amplitude={self.amplitude.item()}"
    

class SPNFExactFourierEncoding(FourierEncoding):
    """
    Fourier feature encoding to follow exact spectral prefiltering of neural fields.
    """
    def forward(self, X, covariances=None, filter_type="gaussian"):
        x_nodownweight = super().forward(X)
        
        if covariances is None:
            return x_nodownweight
        
        # Get the quad form
        quad_form = (self.A.T * (covariances @ self.A.T)).sum(dim=-2).clamp(0)
        dims = covariances.shape[-1] if len(covariances.shape) > 1 else 1
        
        filter_type = filter_type.lower()
        if filter_type == "gaussian":
            exponent = 2 * (torch.pi ** 2) * quad_form
            exponent = (-exponent).exp()
        elif filter_type == "uniform_ellipsoid":
            eps, threshold = self.kwargs.get("eps", 1e-16), self.kwargs.get("threshold", 1e-4)
            sqrt_quad = torch.sqrt(quad_form + eps)
            
            # Compute x = 2π * sqrt_quad
            bessel_input = 2.0 * torch.pi * sqrt_quad
            if dims == 1:
                # 1D box (interval): frequency response = sin(x)/x with x = 2π*sqrt(b^TΣb)
                exponent = torch.where(
                    bessel_input < threshold,
                    1.0,
                    torch.sin(bessel_input) / bessel_input
                )
            elif dims == 2:
                # Compute Bessel J1 and handle division for non-masked elements
                j1_x = torch.special.bessel_j1(bessel_input)
                # Calculate the result using the limit for small x and the formula otherwise
                exponent = torch.where(
                    bessel_input < threshold,
                    1.0,
                    (2.0 * j1_x) / bessel_input
                )
            elif dims == 3:
                # Explicitly compute J_{3/2}(x) using sin and cos
                sin_x = torch.sin(bessel_input)
                cos_x = torch.cos(bessel_input)
                j32_x = torch.sqrt(2.0 / (torch.pi * bessel_input)) * (sin_x / bessel_input - cos_x)
                
                # Calculate the result
                exponent = torch.where(
                    bessel_input < threshold,
                    1.0,
                    (j32_x / bessel_input**1.5) * (3 * (torch.pi / 2.0) ** 0.5)
                )
        elif filter_type == "lanczos":
            eps, threshold = self.kwargs.get("eps", 1e-16), self.kwargs.get("threshold", 1e-4)
            lanczos_a = self.kwargs.get("lanczos_a", 1)
            sqrt_quad = torch.sqrt(quad_form + eps)
            
            lo = (1.0 - 1.0 / lanczos_a) * 0.5   # pass-band edge
            hi = (1.0 + 1.0 / lanczos_a) * 0.5   # stop-band edge

            # initialise with zeros (stop-band)
            Lambda = torch.zeros_like(sqrt_quad)

            # flat pass-band (gain = 1)
            Lambda = torch.where(sqrt_quad <= lo, 1.0, Lambda)

            # linear roll-off region
            mid = (sqrt_quad > lo) & (sqrt_quad < hi)
            Lambda = torch.where(mid, (lanczos_a + 1) * 0.5 - lanczos_a * sqrt_quad, Lambda)
            
            # Normalize by the mass
            Lambda = Lambda / get_lanczos_mass(lanczos_a, dims)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
                
        exponent = exponent.tile(2)

        return x_nodownweight * exponent
    
class IdentityEncoding(nn.Module):
    """
    Identity encoding (no encoding).
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, X, *args, **kwargs):
        return X