"""Performative-stability analysis on the EGTA decoupled-PR tensor.

Given M in R^{T x K x K} from egta_matrix.py, intervention i is *performatively
stable at time t* iff

    i in argmin_j M_t[i, j]

(no in-set deviation reduces loss in i's own induced world). The deviation gain

    Delta_t[i] = M_t[i, i] - min_j M_t[i, j]  >= 0

is zero exactly when i is stable. This is the empirical analog of Perdomo's
pi_ST in argmin_pi E_{z ~ D(pi_ST)} L(pi).

Outputs (alongside the input .npz):
  * <stem>_equilibria.npz with arrays {deviation_gain, is_stable, best_response,
    tol}.
  * <stem>_stability_strip.png -- binary heatmap over (t, intervention) showing
    when each intervention is performatively stable.
  * <stem>_deviation_gain.png -- per-intervention Delta_t trajectory + tol line.

Usage:
    python -m LLM_experiments.equilibrium /path/egta_sft_sweep_with_static_s0.npz \
        --tol 0.01 --out-dir /path/figs_egta/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize

from LLM_experiments.egta_wandb import WandbSession, add_wandb_args

try:
    from notebook_env import *  # noqa: F401,F403
    import evoml.analysis as evan
    _HAVE_NOTEBOOK_ENV = True
except Exception as e:
    print(f"[equilibrium] notebook_env import failed: {e}", file=sys.stderr)
    matplotlib.rcParams["text.usetex"] = False
    evan = None
    _HAVE_NOTEBOOK_ENV = False


def deviation_gain(M: np.ndarray) -> np.ndarray:
    """Delta[t, i] = M[t, i, i] - min_j M[t, i, j]. Shape (T, K). Always >= 0."""
    diag = np.einsum("tii->ti", M)
    return diag - M.min(axis=2)


def best_response_map(M: np.ndarray) -> np.ndarray:
    """BR[t, i] = argmin_j M[t, i, j]. Shape (T, K). Ties broken by lowest index."""
    return M.argmin(axis=2)


def is_stable(M: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Boolean (T, K): True iff intervention i is performatively stable at t."""
    return deviation_gain(M) <= tol


def _grid(ax):
    if evan is not None:
        ax.grid(**evan.background_line_style)
    else:
        ax.grid(alpha=0.25)


def nash_minimax(M_t: np.ndarray):
    """Solve min_q max_p p'Mq for cost matrix M_t (K x K).

    Returns (p_star, q_star, v_star) with both as Delta^K probability vectors.
    Two LPs to recover both sides; v_star equals max from both by strong duality.
    """
    K = M_t.shape[0]
    # Column player (min over q): min_q max_i (Mq)_i = min t s.t. Mq <= t*1, q>=0, sum q = 1.
    c = np.zeros(K + 1)
    c[-1] = 1.0
    A_ub = np.hstack([M_t, -np.ones((K, 1))])
    b_ub = np.zeros(K)
    A_eq = np.zeros((1, K + 1)); A_eq[0, :K] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * K + [(None, None)]
    res_q = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res_q.success:
        raise RuntimeError(f"column-player LP failed: {res_q.message}")
    q_star = np.clip(res_q.x[:K], 0.0, None)
    s = q_star.sum()
    q_star = q_star / s if s > 0 else np.full(K, 1.0 / K)
    v_q = float(res_q.x[-1])

    # Row player (max over p): max_p min_j (p'M)_j = max u s.t. p'M >= u*1, p>=0, sum p = 1.
    c = np.zeros(K + 1)
    c[-1] = -1.0  # maximize u
    A_ub = np.hstack([-M_t.T, np.ones((K, 1))])  # -M'p + u <= 0
    b_ub = np.zeros(K)
    A_eq = np.zeros((1, K + 1)); A_eq[0, :K] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * K + [(None, None)]
    res_p = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res_p.success:
        raise RuntimeError(f"row-player LP failed: {res_p.message}")
    p_star = np.clip(res_p.x[:K], 0.0, None)
    s = p_star.sum()
    p_star = p_star / s if s > 0 else np.full(K, 1.0 / K)
    v_p = float(res_p.x[-1])

    if abs(v_p - v_q) > 1e-6:
        print(f"[equilibrium] WARNING duality gap |v_p - v_q| = {abs(v_p - v_q):.2e}", file=sys.stderr)
    return p_star, q_star, 0.5 * (v_p + v_q)


def nash_minimax_per_t(M: np.ndarray):
    """Apply nash_minimax to each time slice. Returns p_traj, q_traj, v_traj."""
    T, K, _ = M.shape
    p_traj = np.empty((T, K))
    q_traj = np.empty((T, K))
    v_traj = np.empty(T)
    for tt in range(T):
        p_traj[tt], q_traj[tt], v_traj[tt] = nash_minimax(M[tt])
    return p_traj, q_traj, v_traj


def _neg_entropy(x):
    safe = np.maximum(x, 1e-30)
    return float(np.sum(x * np.log(safe)))


def _neg_entropy_jac(x):
    safe = np.maximum(x, 1e-30)
    return np.log(safe) + 1.0


def max_entropy_q(M_t: np.ndarray, v_star: float, *, slack: float = 1e-7) -> np.ndarray:
    """Among NE for column player (q with max_i (Mq)_i <= v*), return the
    maximum-entropy one. Tiebreaker over the degenerate optimal face of the LP.
    """
    K = M_t.shape[0]
    x0 = np.full(K, 1.0 / K)
    constraints = [
        {"type": "eq", "fun": lambda q: q.sum() - 1.0, "jac": lambda q: np.ones_like(q)},
        {"type": "ineq", "fun": lambda q, M=M_t, v=v_star, s=slack: -(M @ q) + v + s,
         "jac": lambda q, M=M_t: -M},
    ]
    bounds = [(0.0, 1.0)] * K
    res = minimize(_neg_entropy, x0, jac=_neg_entropy_jac, method="SLSQP",
                   constraints=constraints, bounds=bounds,
                   options={"maxiter": 300, "ftol": 1e-10})
    if not res.success:
        return x0
    q = np.clip(res.x, 0.0, 1.0)
    s = q.sum()
    return q / s if s > 0 else x0


def max_entropy_p(M_t: np.ndarray, v_star: float, *, slack: float = 1e-7) -> np.ndarray:
    """Among NE for row player (p with min_j (p'M)_j >= v*), return the
    maximum-entropy one."""
    K = M_t.shape[0]
    x0 = np.full(K, 1.0 / K)
    constraints = [
        {"type": "eq", "fun": lambda p: p.sum() - 1.0, "jac": lambda p: np.ones_like(p)},
        {"type": "ineq", "fun": lambda p, M=M_t, v=v_star, s=slack: (M.T @ p) - v + s,
         "jac": lambda p, M=M_t: M.T},
    ]
    bounds = [(0.0, 1.0)] * K
    res = minimize(_neg_entropy, x0, jac=_neg_entropy_jac, method="SLSQP",
                   constraints=constraints, bounds=bounds,
                   options={"maxiter": 300, "ftol": 1e-10})
    if not res.success:
        return x0
    p = np.clip(res.x, 0.0, 1.0)
    s = p.sum()
    return p / s if s > 0 else x0


def max_entropy_ne_per_t(M: np.ndarray, v_traj: np.ndarray):
    """Apply max_entropy_{p,q} to each time slice. Returns p_me, q_me."""
    T, K, _ = M.shape
    p_me = np.empty((T, K))
    q_me = np.empty((T, K))
    for tt in range(T):
        q_me[tt] = max_entropy_q(M[tt], float(v_traj[tt]))
        p_me[tt] = max_entropy_p(M[tt], float(v_traj[tt]))
    return p_me, q_me


def plot_nash_stack(q_traj: np.ndarray, labels, t_axis, *, out_path: Path, side: str):
    fig, ax = plt.subplots(figsize=(9, 4.0), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    ax.stackplot(t_axis, q_traj.T, labels=[str(x) for x in labels],
                 colors=[cmap(i % 10) for i in range(q_traj.shape[1])], alpha=0.85)
    ax.set_xlabel("Time ($t$)")
    ax.set_ylabel("Probability")
    title = "Robust deployment mixture $q^*_t$" if side == "q" else "Adversarial world mixture $p^*_t$"
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_nash_value(v_traj: np.ndarray, M: np.ndarray, t_axis, *, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4.0), constrained_layout=True)
    diag = np.einsum("tii->ti", M)
    best_pure = diag.min(axis=1)
    worst_pure = diag.max(axis=1)
    ax.plot(t_axis, v_traj, color="black", linewidth=1.8, label=r"NE value $v^*_t$")
    ax.plot(t_axis, best_pure, color="tab:green", linewidth=1.3, linestyle="--",
            label=r"$\min_i M_t[i,i]$ (best pure)")
    ax.plot(t_axis, worst_pure, color="tab:red", linewidth=1.3, linestyle="--",
            label=r"$\max_i M_t[i,i]$ (worst pure)")
    ax.set_xlabel("Time ($t$)")
    ax.set_ylabel("Risk")
    ax.set_title(r"Nash value $v^*_t = \min_q \max_p p^\top M_t q$ vs pure-strategy bounds")
    _grid(ax)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_stability_strip(stab: np.ndarray, labels, t_axis, *, out_path: Path, tol: float):
    K = stab.shape[1]
    fig, ax = plt.subplots(figsize=(8.5, 0.6 + 0.5 * K), constrained_layout=True)
    ax.imshow(stab.T.astype(float), aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0,
              interpolation="nearest", extent=[t_axis[0] - 0.5, t_axis[-1] + 0.5, K - 0.5, -0.5])
    ax.set_yticks(range(K))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time ($t$)")
    ax.set_title(rf"Performative stability strip ($\Delta_t(i) \leq {tol:g}$)")
    ax.invert_yaxis()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_deviation_gain(dev: np.ndarray, labels, t_axis, *, out_path: Path, tol: float):
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for i, label in enumerate(labels):
        ax.plot(t_axis, dev[:, i], color=cmap(i % 10), linewidth=1.6, label=str(label))
    ax.axhline(tol, color="black", linestyle=":", alpha=0.6, label=rf"tol $={tol:g}$")
    ax.set_xlabel("Time ($t$)")
    ax.set_ylabel(r"$\Delta_t(i) = M_t[i,i] - \min_j M_t[i,j]$")
    ax.set_title("Performative deviation gain (0 = stable)")
    _grid(ax)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _terminal_summary(M, labels, t_axis, *, tol, dev, stab, br):
    T = M.shape[0]
    tT = T - 1
    print(f"\n[equilibrium] terminal (t={int(t_axis[tT])}) snapshot:")
    print(f"  diag M[t,i,i]:    {dict(zip(labels, np.einsum('ii->i', M[tT]).round(3)))}")
    print(f"  min_j M[t,i,j]:   {dict(zip(labels, M[tT].min(axis=1).round(3)))}")
    print(f"  Delta_t:          {dict(zip(labels, dev[tT].round(3)))}")
    print(f"  BR_t:             {dict(zip(labels, [labels[j] for j in br[tT]]))}")
    print(f"  stable (tol={tol:g}): {[labels[i] for i in range(len(labels)) if stab[tT, i]]}")
    return


def _coverage_summary(stab, labels):
    coverage = stab.mean(axis=0)
    print("[equilibrium] fraction of t with intervention stable:")
    print({lab: float(round(c, 3)) for lab, c in zip(labels, coverage)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path, help=".npz from egta_matrix.py")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="Tolerance on deviation gain to count as stable (default 0.01).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory for PNGs and equilibria .npz (default: alongside the input .npz).")
    ap.add_argument("--do-nash", action="store_true",
                    help="Also compute per-t Nash minimax saddle point (p*, q*, v*).")
    add_wandb_args(ap, default_project="egta-analysis")
    args = ap.parse_args()

    if not args.npz.exists():
        print(f"[equilibrium] .npz not found: {args.npz}", file=sys.stderr)
        sys.exit(2)
    data = np.load(args.npz, allow_pickle=False)
    M = data["M"]
    t_axis = data["t"]
    labels = [str(x) for x in data["labels"]]

    dev = deviation_gain(M)
    if (dev < -1e-12).any():
        bad = np.argwhere(dev < -1e-12)[:5]
        print(f"[equilibrium] WARNING: negative deviation gain at {bad.tolist()} -- numerics?", file=sys.stderr)

    br = best_response_map(M)
    stab = is_stable(M, tol=args.tol)

    out_dir = args.out_dir or args.npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.npz.stem
    eq_npz = out_dir / f"{stem}_equilibria.npz"
    np.savez(eq_npz,
             deviation_gain=dev, is_stable=stab, best_response=br, tol=np.asarray(args.tol),
             t=t_axis, labels=np.asarray(labels))
    stab_strip_png = out_dir / f"{stem}_stability_strip.png"
    dev_gain_png = out_dir / f"{stem}_deviation_gain.png"
    plot_stability_strip(stab, labels, t_axis, out_path=stab_strip_png, tol=args.tol)
    plot_deviation_gain(dev, labels, t_axis, out_path=dev_gain_png, tol=args.tol)

    _terminal_summary(M, labels, t_axis, tol=args.tol, dev=dev, stab=stab, br=br)
    _coverage_summary(stab, labels)
    print(f"[equilibrium] wrote {stem}_equilibria.npz and 2 PNGs to {out_dir}")

    wb = WandbSession(
        enabled=args.wandb,
        project=args.wandb_project,
        name=args.wandb_name or f"{stem}_eq",
        config={
            "stage": "equilibrium",
            "input_npz": str(args.npz),
            "labels": labels,
            "K": len(labels),
            "T": int(M.shape[0]),
            "tol": float(args.tol),
            "do_nash": bool(args.do_nash),
        },
    )
    if wb.active:
        K = len(labels)
        for tt in range(M.shape[0]):
            wb.log_metrics({
                "eq/dev_gain_min": float(dev[tt].min()),
                "eq/dev_gain_max": float(dev[tt].max()),
                "eq/dev_gain_mean": float(dev[tt].mean()),
                "eq/n_stable": int(stab[tt].sum()),
            }, step=int(t_axis[tt]))
        tT = M.shape[0] - 1
        terminal_stable = [labels[i] for i in range(K) if stab[tT, i]]
        wb.log_metrics({
            "eq/terminal_dev_gain_max": float(dev[tT].max()),
            "eq/terminal_n_stable": int(stab[tT].sum()),
            "eq/terminal_stable_set": "|".join(terminal_stable) if terminal_stable else "(none)",
        })
        wb.log_artifact(eq_npz, artifact_type="egta-equilibria")
        wb.log_image(stab_strip_png, key="eq/stability_strip")
        wb.log_image(dev_gain_png, key="eq/deviation_gain")

    if args.do_nash:
        p_traj, q_traj, v_traj = nash_minimax_per_t(M)
        p_me, q_me = max_entropy_ne_per_t(M, v_traj)
        nash_npz = out_dir / f"{stem}_nash.npz"
        np.savez(nash_npz,
                 p_star=p_traj, q_star=q_traj, v_star=v_traj,
                 p_star_me=p_me, q_star_me=q_me,
                 t=t_axis, labels=np.asarray(labels))
        nash_q_png = out_dir / f"{stem}_nash_qstar.png"
        nash_p_png = out_dir / f"{stem}_nash_pstar.png"
        nash_q_me_png = out_dir / f"{stem}_nash_qstar_maxent.png"
        nash_p_me_png = out_dir / f"{stem}_nash_pstar_maxent.png"
        nash_v_png = out_dir / f"{stem}_nash_value.png"
        plot_nash_stack(q_traj, labels, t_axis, out_path=nash_q_png, side="q")
        plot_nash_stack(p_traj, labels, t_axis, out_path=nash_p_png, side="p")
        plot_nash_stack(q_me, labels, t_axis, out_path=nash_q_me_png, side="q")
        plot_nash_stack(p_me, labels, t_axis, out_path=nash_p_me_png, side="p")
        plot_nash_value(v_traj, M, t_axis, out_path=nash_v_png)
        T = M.shape[0]
        tT = T - 1
        diag_T = np.einsum("ii->i", M[tT])
        print(f"\n[equilibrium] terminal Nash (t={int(t_axis[tT])}):")
        print(f"  v* = {v_traj[tT]:.4f}   (best pure diag: {diag_T.min():.4f}, worst pure diag: {diag_T.max():.4f})")
        print(f"  LP-vertex q*: {[(lab, round(float(q), 3)) for lab, q in zip(labels, q_traj[tT]) if q > 1e-3]}")
        print(f"  LP-vertex p*: {[(lab, round(float(p), 3)) for lab, p in zip(labels, p_traj[tT]) if p > 1e-3]}")
        print(f"  max-ent  q*: {[(lab, round(float(q), 3)) for lab, q in zip(labels, q_me[tT]) if q > 1e-3]}")
        print(f"  max-ent  p*: {[(lab, round(float(p), 3)) for lab, p in zip(labels, p_me[tT]) if p > 1e-3]}")
        print(f"[equilibrium] wrote {stem}_nash.npz and 5 Nash PNGs.")

        if wb.active:
            for tt in range(T):
                wb.log_metrics({
                    "eq/nash_v": float(v_traj[tt]),
                    "eq/nash_q_argmax": labels[int(q_me[tt].argmax())],
                    "eq/nash_p_argmax": labels[int(p_me[tt].argmax())],
                    "eq/nash_q_entropy": float(-(q_me[tt] * np.log(np.maximum(q_me[tt], 1e-30))).sum()),
                    "eq/nash_p_entropy": float(-(p_me[tt] * np.log(np.maximum(p_me[tt], 1e-30))).sum()),
                }, step=int(t_axis[tt]))
            q_support_T = [lab for lab, q in zip(labels, q_me[tT]) if q > 1e-3]
            p_support_T = [lab for lab, p in zip(labels, p_me[tT]) if p > 1e-3]
            wb.log_metrics({
                "eq/terminal_nash_v": float(v_traj[tT]),
                "eq/terminal_q_support": "|".join(q_support_T) if q_support_T else "(none)",
                "eq/terminal_p_support": "|".join(p_support_T) if p_support_T else "(none)",
            })
            wb.log_artifact(nash_npz, artifact_type="egta-nash")
            wb.log_image(nash_q_png, key="nash/qstar_lp")
            wb.log_image(nash_p_png, key="nash/pstar_lp")
            wb.log_image(nash_q_me_png, key="nash/qstar_maxent")
            wb.log_image(nash_p_me_png, key="nash/pstar_maxent")
            wb.log_image(nash_v_png, key="nash/value_vs_t")

    if wb.active:
        wb.finish()


if __name__ == "__main__":
    main()
