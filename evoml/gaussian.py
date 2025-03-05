import numpy as np
import scipy.stats
import matplotlib

from .common_types import RandomVariablePopulationGroup

__all__ = [
    'GaussianPopulationGroup',
    'gaussian_confidence_ellipse',
    'rot_2d',
]

rot_2d = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
rot_vec = lambda v, theta: v@rot_2d(theta).T
rot_mat = lambda m, theta: rot_2d(theta)@m@rot_2d(theta).T

class GaussianPopulationGroup(RandomVariablePopulationGroup):
    def __init__(self, mu_vec, sigma_vec, p):
        self.rv_type = (
            scipy.stats.multivariate_normal if np.ndim(mu_vec[0])>0
            else scipy.stats.norm
        )
        super().__init__(
            rv_vec = [
                self.rv_type(mu, sigma)
                for mu, sigma in zip(mu_vec, sigma_vec)
            ],
            p=p,
        )
        self.mu_vec = mu_vec
        self.sigma_vec = sigma_vec

    @classmethod
    def bivariate_from_angle_offset(cls, theta, d, offset, sigma_x, sigma_y, balance_p=0.5):
        return cls(
            mu_vec=rot_vec(offset+np.array([[-d,0],[d,0]]), theta),
            sigma_vec=[rot_mat(np.diag([sigma_x,sigma_y]),theta)]*2,
            p=[1-balance_p,balance_p],
        )

    @classmethod
    def univariate_from_offset(cls, d, offset, sigma, balance_p=0.5):
        return cls(
            mu_vec=np.array([-d,d])+offset,
            sigma_vec=[sigma]*2,
            p=[1-balance_p,balance_p],
        )

def gaussian_confidence_ellipse(ax,rv,n_std,**kwargs):
    # https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    cov = rv.cov
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = matplotlib.patches.Ellipse(
        (0, 0),
        width=ell_radius_x*2,
        height=ell_radius_y*2,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = rv.mean[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = rv.mean[1]

    transf = (
        matplotlib.transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

