
import numpy as np
import scipy.optimize
from typing import Tuple, Callable

from .. import tools

def average_payload(
    *,
    ps: Tuple[np.ndarray] = None,
    e: float = None,
    lbda: float = None,
    rhos: Tuple[np.ndarray] = None,
    add_zero: bool = True,
) -> Tuple[Tuple[np.ndarray], float]:
    """"""
    assert (
        (ps is not None and rhos is None) or (ps is None and rhos is not None)
    ), 'exactly one of ps or rhos must be given'
    assert (
        lbda is not None or ps is not None
    ), 'lambda can be specified only with rhos'
    #
    if ps is None:
        # denominator (forced left-associativity)
        ps = [
            np.exp(-lbda * rho)
            for rho in rhos
        ]
        denum = 1 if add_zero else 0
        for p in ps:
            denum += p
        denum[denum == 0] = tools.EPS
        #
        ps = [
            ps[i] / denum
            for i in range(len(rhos))
        ]
        # ps = [
        #     get_p(lbda, rhos[i], *rhos[:i], *rhos[i+1:], add_zero=add_zero)
        #     for i in range(len(rhos))
        # ]

    # Imperfect coding - given embedding efficiency
    if e is not None:
        raise NotImplementedError('to be implemented')
        H_hat = np.sum(ps) * e

    # Perfect coding - upper bound efficiency
    else:
        H_hat = tools.entropy(*ps)

    # print(lbda, H_hat, rhos.shape)

    return ps, H_hat


def probability(
    rhos: Tuple[np.ndarray],
    alpha: float,
    n: int,
    objective: Callable = None,
    stack_axis: int = None
) -> Tuple[Tuple[np.ndarray], float]:
    """"""
    # objective
    if objective is None:
        objective = average_payload

    # lambda search
    message_length = int(np.round(alpha * n))
    lbda = scipy.optimize.fminbound(
        lambda lbda: (
            objective(rhos=rhos, lbda=lbda)[1] - message_length
        )**2,
        0, 1000,
        xtol=1,
        maxfun=100,
        # disp=3,
    )

    # get probabilites
    ps, _ = objective(lbda=lbda, rhos=rhos)
    if stack_axis is not None:
        ps = np.stack(ps, axis=stack_axis)
    return ps, lbda


def simulate(
    ps: Tuple[np.ndarray],
    deltas: np.ndarray,
    add_zero: bool = True,
    seed: int = None,
) -> np.ndarray:
    """"""
    # add no change
    if add_zero:
        ps = np.concatenate([
            np.cumsum(ps, axis=0),
            np.ones_like(ps[:1]),  # no change
        ], axis=0)
        deltas = np.concatenate([
            deltas,
            [[0]*deltas.shape[1]],  # no change
        ], axis=0)
        # print(ps.shape, deltas.shape)
    # do not assume 0
    else:
        assert 2**deltas.shape[0] == deltas.shape[1]

    # draw sample ~ U
    rng = np.random.default_rng(seed=seed)
    rand_change = rng.random(ps.shape[1:])

    # convert to deltas
    idx = np.argmax(ps > rand_change[None], axis=0)
    return deltas[idx]


def get_deltas(rhos: np.ndarray) -> np.ndarray:
    q = len(rhos)
    deltas = np.arange(1, 2**q)
    deltas = (((deltas[:, None] & (1 << np.arange(q)))) > 0).astype(int)
    return deltas


def qary(
    rhos: np.ndarray,
    alpha: float,
    n: int,
    *,
    objective: Callable = None,
    deltas: np.ndarray = None,
    **kw,
) -> np.ndarray:
    """"""
    # construct delta matrix
    if deltas is None:
        deltas = get_deltas(rhos)
    #
    ps, _ = probability(
        rhos=rhos,
        alpha=alpha,
        n=n,
        objective=objective,
    )
    # print('after probability:', [p.shape for p in ps])
    # print(average_payload(ps=ps, lbda=lbda)[1] / n)
    return simulate(
        ps=ps,
        deltas=deltas,
        **kw,
    )


def q_to_pow2q(
    rhos: np.ndarray,
    *,
    deltas: np.ndarray = None,
) -> np.ndarray:
    """"""
    if deltas is None:
        deltas = get_deltas(rhos)
    #
    rhos8 = (deltas @ rhos.reshape(len(rhos), -1)).reshape(len(deltas), *rhos.shape[1:])
    #
    return rhos8, deltas
