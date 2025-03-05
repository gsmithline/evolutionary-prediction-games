import functools
import numpy as np
import scipy.stats

__all__ = [
    'PopulationGroup',
    'RandomVariablePopulationGroup',
    'Population',
    'MixtureRandomVariable',
    'TransformedRV',
    'RigidPostProcessingRV',
    'ProductRV',
    'DiscreteOneDimensionalRV',
]

basis = lambda n,i: (np.arange(n)==i).astype(int)

def assert_sum1(p):
    assert np.isclose(sum(p),1), f'Sum of probability vector is {sum(p)}!=1'

def sample_from_nvals(nvals, samplers, n_features, sample_type, random_state):
    n = sum(nvals)
    rng = np.random.default_rng(random_state)
    X = np.zeros((n,n_features))
    y = np.zeros(n)
    g = np.zeros(n)-1
    ind = 0
    for g_i, n_i in enumerate(nvals):
        if n_i==0:
            continue
        s = samplers[g_i](n=n_i, random_state=rng)
        if sample_type=='X':
            X[ind:ind+n_i] = s if n_features>1 else s.reshape(-1,1)
            y[ind:ind+n_i] = g_i
        elif sample_type=='Xy':
            X[ind:ind+n_i]= s[0] if n_features>1 else s[0].reshape(-1,1)
            y[ind:ind+n_i] = s[1]
            g[ind:ind+n_i] = g_i
        ind += n_i
    ind = np.arange(n)
    rng.shuffle(ind)
    if sample_type=='X':
        return X[ind], y[ind]
    elif sample_type=='Xy':
        return X[ind], y[ind], g[ind]


class PopulationGroup:
    def sample(self, n, random_state=None):
        raise NotImplementedError

    @property
    def n_features(self):
        raise NotImplementedError


class RandomVariablePopulationGroup(PopulationGroup):
    def __init__(self, rv_vec, p):
        """
        rv_vec: List of random variables, each one representing p(x|y) for some y
        p: Label distribution p(y)
        """
        assert_sum1(p)
        assert len(p)==len(rv_vec)
        self.rv_vec = rv_vec
        self.p = p
        self.k = len(rv_vec)

    def _sampler(self, rv):
        return lambda n,random_state: rv.rvs(size=n, random_state=random_state)

    def sample(self, n, random_state=None):
        rng = np.random.default_rng(random_state)
        nvals = rng.multinomial(n=n, pvals=self.p)
        X,y = sample_from_nvals(
            nvals=nvals,
            samplers=[
                self._sampler(rv)
                for rv in self.rv_vec
            ],
            sample_type='X',
            n_features=self.n_features,
            random_state=random_state,
        )
        return X,y

    @property
    def n_features(self):
        return getattr(self.rv_vec[0], 'dim', 1)

    def pdf(self, x):
        return np.stack([
        p_k*rv_k.pdf(x)
        for rv_k,p_k in zip(self.rv_vec, self.p)
    ])


class TransformedRV(scipy.stats.rv_continuous):
    def __init__(self, rv, transform, *args, **kwargs):
        super().__init__(
            name=f'TransformedRV({rv})',
        )
        self.__transform = transform
        self.__rv = rv

    def _rvs(self, *args, **kwargs):
        x = self.__rv.rvs(*args, **kwargs)
        return self.__transform(x)


class Population:
    def __init__(self, groups):
        """
        groups: List of PopulationGroups
        """
        self.groups = groups

    @property
    def n_groups(self):
        return len(self.groups)

    def _sampler(self, g):
        return lambda n,random_state: g.sample(n=n, random_state=random_state)

    def _sample_from_nvals(self, nvals, random_state):
        return sample_from_nvals(
            nvals=nvals,
            samplers=[
                self._sampler(g)
                for g in self.groups
            ],
            n_features=self.n_features,
            sample_type='Xy',
            random_state=random_state,
        )

    def sample_from_mixture(self, n, p, random_state=None):
        """
        n: sample size
        p: group mixture coefficients (list of size n_groups)
        random_state: random seed
        """
        assert_sum1(p)
        assert len(p)==self.n_groups
        rng = np.random.default_rng(random_state)
        nvals = rng.multinomial(n=n, pvals=p)
        return self._sample_from_nvals(nvals, random_state=rng)

    def sample_from_uniform_mixture(self, n, random_state=None):
        return self.sample_from_mixture(
            n,
            p=np.ones(self.n_groups)/self.n_groups,
            random_state=random_state,
        )

    def sample_from_hypergeometric(self, n, m, random_state=None):
        """
        n: sample size
        m: number of samples from each group (list of size n_groups)
        random_state: random seed
        """
        assert n<=sum(m)
        nvals = scipy.stats.multivariate_hypergeom.rvs(m=m, n=n, random_state=random_state)
        return self._sample_from_nvals(nvals, random_state=random_state)

    def sample_from_group(self, n, g, random_state=None):
        """
        n: sample size
        g: group index (e.g. 0 or 1 in the case of two groups)
        random_state: random seed
        """
        assert 0<=g<self.n_groups, f'Invalid group index i={i}'
        return self.sample_from_mixture(
            n=n,
            p=(np.arange(self.n_groups)==g).astype(float),
            random_state=random_state,
        )[:2]

    def mixture_pdf(self, x, p):
        return np.stack([g.pdf(x) for g in self.groups],axis=-1)@p

    @property
    def n_features(self):
        return self.groups[0].n_features


class MixtureRandomVariable:
    def __init__(self,rv_vec,p):
        """
        rv_vec: List of random variables
        p: List of mixture coefficients
        """
        assert_sum1(p)
        assert len(p)==len(rv_vec)
        self.rv_vec = rv_vec
        self.p = np.array(p)
        self.k = len(p)

    def rvs(self,size=1,random_state=None):
        out = np.zeros(size)
        rng = np.random.default_rng(random_state)
        choice = rng.choice(
            a=range(self.k),
            size=size,
            p=self.p,
        )
        for i in range(self.k):
            out[choice==i] = self.rv_vec[i].rvs(
                size=(choice==i).sum(),
                random_state=rng,
            )
        return out

    def pdf(self, x):
        pdfs = np.stack([
            rv.pdf(x)
            for rv in self.rv_vec
        ])
        return self.p@pdfs


class RigidPostProcessingRV:
    def __init__(self, rv, f, f_inv):
        """
        rv: Random variable
        f: Rigid transformation function (assumed to have jacobian=1)
        f_inv: Inverse transformation
        """
        self.rv = rv
        self.f = f
        self.f_inv = f_inv

    def rvs(self, *args, **kwargs):
        X = self.rv.rvs(*args, **kwargs).reshape(-1,self.rv.dim)
        return self.f(X)

    @property
    def dim(self):
        return self.rv.dim

    def pdf(self, x, *args, **kwargs):
        return self.rv.pdf(self.f_inv(x),*args,**kwargs)


class ProductRV:
    def __init__(self, rv_vec):
        """
        rv_vec: List of random variables
        """
        self.rv_vec = rv_vec

    def rvs(self, *args, **kwargs):
        return np.stack(
            [
                rv.rvs(*args, **kwargs)
                for rv in self.rv_vec
            ],
            axis=-1,
        )

    def pdf(self, x):
        out = np.ones(x.shape[:-1])
        j = 0
        for rv in self.rv_vec:
            dim = getattr(rv,'dim',1)
            out *= np.squeeze(rv.pdf(x[...,j:j+dim]),axis=-1)
            j += dim
        return out

    @property
    def dim(self):
        return sum(getattr(rv,'dim',1) for rv in self.rv_vec)


class DiscreteOneDimensionalRV:
    def __init__(self, x, p):
        self.x = np.array(x)
        self.p = np.array(p)

    def rvs(self, size, random_state=None):
        rng = np.random.default_rng(random_state)
        return rng.choice(
            a=self.x,
            size=size,
            replace=True,
            p=self.p/self.p.sum(),
        )

    def pdf(self, x):
        return np.interp(
            x=x,
            xp=self.x,
            fp=self.p,
        )
