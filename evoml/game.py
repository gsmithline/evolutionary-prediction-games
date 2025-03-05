import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def from_fitness_functions(f_A, f_B, p_vec):
    return (
        pd.DataFrame([
            {
                'p_A': 1-p_B,
                'p_B': p_B,
                'acc_A': f_A(p_B),
                'acc_B': (f_B or (lambda p: f_A(1-p)))(p_B),
            }
            for p_B in p_vec
        ])
        .assign(acc_p=lambda df: df['acc_A']*df['p_A'] + df['acc_B']*df['p_B'])
        .set_index('p_B')
    )

def from_population_1d_rv(pop, clf_factory, x, p_vec):
    X = np.expand_dims(np.hstack([x,x]),-1)
    y = np.hstack([np.zeros_like(x), np.ones_like(x)])
    
    results = []

    # for p in tqdm(p_vec, desc='Population game'):
    for p in p_vec:
        w = np.hstack(pop.mixture_pdf(x, p=[1-p,p]))
        clf = clf_factory()
        clf.fit(X, y, w)
        results.append({
            'p_B': p,
            'clf': clf,
            'acc_A': clf.score(
                X,
                y,
                np.hstack(pop.mixture_pdf(x, p=[1,0])),
            ),
            'acc_B': clf.score(
                X,
                y,
                np.hstack(pop.mixture_pdf(x, p=[0,1])),
            ),
        })
        results[-1]['acc_p'] = (1-p)*results[-1]['acc_A'] + p*results[-1]['acc_B']
    
    return pd.DataFrame(results).set_index('p_B').sort_index()

def from_population_sample(pop, clf_factory, p_vec, n_train, n_test, n_reps, random_state):   
    results = []
    for p in tqdm(p_vec, desc='Population game'):
        for rep in range(n_reps):
            clf = clf_factory()
            clf.fit(*pop.sample_from_mixture(
                n=n_train,
                p=[1-p,p],
                random_state=random_state,
            )[:2])
            results.append({
                'rep': rep,
                'p_A': 1-p,
                'p_B': p,
                'clf': clf,
                'acc_A': clf.score(*pop.sample_from_group(n_test,0,random_state)),
                'acc_B': clf.score(*pop.sample_from_group(n_test,1,random_state)),
            })
            results[-1]['acc_p'] = (1-p)*results[-1]['acc_A'] + p*results[-1]['acc_B']
    
    return pd.DataFrame(results).set_index('p_B').sort_index()

