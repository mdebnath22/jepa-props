"""
numerical_opt.py — Pure numerical function optimization benchmark suite,
matching ProPS Appendix A exactly: 5 benchmark functions (Ackley,
Rastrigin, Levy, Weierstrass, Salomon) at D in {2, 4, 8, 16}, with random
per-task offset to prevent trivial origin-exploitation, compared against
Gradient Descent, Adam, Nelder-Mead, Random Search, and an LLM-iterative
optimizer.

This is intentionally a SEPARATE, self-contained module from envs.py /
baselines.py: there is no environment rollout here, just direct function
evaluation, matching ProPS's own separation (their Appendix A runs fully
independently of the RL task suite in the main paper).
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


def ackley(x, a=20.0, b=0.2, c=2 * np.pi):
    term1 = -a * np.exp(-b * np.sqrt(np.mean(x ** 2)))
    term2 = -np.exp(np.mean(np.cos(c * x)))
    return term1 + term2 + a + np.exp(1)


def rastrigin(x):
    D = len(x)
    return 10 * D + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def levy(x):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    return term1 + term2 + term3


def weierstrass(x, a=0.5, b=3.0, k_max=20):
    D = len(x)
    k = np.arange(k_max + 1)
    ak = a ** k
    bk = b ** k
    total = 0.0
    for xi in x:
        total += np.sum(ak * np.cos(2 * np.pi * bk * (xi + 0.5)))
    correction = D * np.sum(ak * np.cos(np.pi * bk))
    return total - correction


def salomon(x):
    norm = np.sqrt(np.sum(x ** 2))
    return 1 - np.cos(2 * np.pi * norm) + 0.1 * norm


BENCHMARK_FUNCTIONS = {
    "ackley": ackley,
    "rastrigin": rastrigin,
    "levy": levy,
    "weierstrass": weierstrass,
    "salomon": salomon,
}

DIMENSIONS = [2, 4, 8, 16]


def make_shifted_objective(fn, dim, seed):
    """Wraps a benchmark function with a random offset o ~ U(0,20)^D,
    matching ProPS's exact shift protocol: f'(x) = f(x - o)."""
    rng = np.random.RandomState(seed)
    offset = rng.uniform(0, 20, size=dim)

    def shifted(x):
        return fn(np.asarray(x) - offset)

    return shifted, offset


def _numerical_gradient(objective, x, eps):
    grad = np.zeros_like(x)
    f0 = objective(x)
    for i in range(len(x)):
        x_eps = x.copy()
        x_eps[i] += eps
        grad[i] = (objective(x_eps) - f0) / eps
    return grad


def run_gradient_descent(objective, x0, n_steps=100, lr=0.005, eps=1e-4):
    x = x0.copy()
    trace = [objective(x)]
    for _ in range(n_steps - 1):
        grad = _numerical_gradient(objective, x, eps)
        x = x - lr * grad
        trace.append(objective(x))
    return x, trace


def run_adam(objective, x0, n_steps=100, lr=0.5, eps=1e-4,
             beta1=0.9, beta2=0.999, adam_eps=1e-8):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    trace = [objective(x)]
    for t in range(1, n_steps):
        grad = _numerical_gradient(objective, x, eps)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + adam_eps)
        trace.append(objective(x))
    return x, trace


def run_nelder_mead(objective, x0, n_steps=100):
    from scipy.optimize import minimize
    trace = []

    def tracked_objective(x):
        val = objective(x)
        trace.append(val)
        return val

    minimize(tracked_objective, x0, method="Nelder-Mead",
              options={"maxfev": n_steps, "xatol": 1e-10, "fatol": 1e-10})

    while len(trace) < n_steps:
        trace.append(trace[-1] if trace else objective(x0))
    return None, trace[:n_steps]


def run_random_search_numerical(objective, x0, n_steps=100, sigma=0.3, seed=0):
    """Local random search: perturb current best, accept if improved.
    Matches ProPS Appendix A.2.2's exact Random Search definition."""
    rng = np.random.RandomState(seed)
    x = x0.copy()
    best_val = objective(x)
    trace = [best_val]
    for _ in range(n_steps - 1):
        candidate = x + rng.randn(len(x)) * sigma
        val = objective(candidate)
        if val < best_val:
            x, best_val = candidate, val
        trace.append(best_val)
    return x, trace


LLM_NUMERICAL_SYSTEM_PROMPT = """You are an optimization assistant, helping me find the global minimum of a mathematical function.
I will give you the function evaluation and the current iteration number at each step.
Your goal is to propose input values that efficiently lead us to the global minimum within a limited number of iterations ({n_steps}).

Here's how we'll interact:
1. I will provide the iteration number and history of previous x values and their f(x) results.
2. You will respond with ONLY a JSON array of {dim} floats representing the next x to try.
   Do not include any explanation, markdown formatting, or additional text.

Remember:
- Do not propose previously seen x values exactly.
- Use the trend in f(x) values to guide your search toward lower values.
- Explore systematically; the function may have many local minima.
"""


def run_llm_numerical_optimizer(objective, x0, n_steps=100, dim=None,
                                  ollama_model="qwen2.5:7b",
                                  ollama_host="http://localhost:11434",
                                  warmup_steps=2, timeout=240, seed=0):
    """Matches ProPS Appendix A.1: Adam warmup for warmup_steps, then
    LLM-guided iteration for the remainder, one LLM call per step."""
    import requests

    dim = dim or len(x0)
    rng = np.random.RandomState(seed)

    x = x0.copy()
    history_x, history_f = [], []
    trace = []

    m, v = np.zeros_like(x), np.zeros_like(x)
    for t in range(1, warmup_steps + 1):
        f_val = objective(x)
        trace.append(f_val)
        history_x.append(x.copy())
        history_f.append(f_val)
        grad = _numerical_gradient(objective, x, 1e-4)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        x = x - 0.5 * m_hat / (np.sqrt(v_hat) + 1e-8)

    system_prompt = LLM_NUMERICAL_SYSTEM_PROMPT.format(n_steps=n_steps, dim=dim)
    n_parse_failures = 0

    for step in range(warmup_steps, n_steps):
        f_val = objective(x)
        trace.append(f_val)
        history_x.append(x.copy())
        history_f.append(f_val)

        hist_lines = [f"x={list(np.round(hx, 4))} f(x)={hf:.4f}"
                      for hx, hf in list(zip(history_x, history_f))[-20:]]
        user_prompt = (
            f"Iteration {step + 1}/{n_steps}\n" + "\n".join(hist_lines) +
            "\nPropose the next x array (JSON array only):"
        )

        try:
            resp = requests.post(
                f"{ollama_host}/api/chat",
                json={"model": ollama_model,
                      "messages": [{"role": "system", "content": system_prompt},
                                   {"role": "user", "content": user_prompt}],
                      "options": {"temperature": 1.0}, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json()["message"]["content"]
        except (Exception,) as e:
            if "Connection" in str(type(e)) or "Timeout" in str(type(e)):
                raise RuntimeError(f"Ollama call failed: {e}")
            log.warning(f"LLM call failed (non-network): {e}")
            text = ""

        new_x = _parse_numerical_response(text, dim)
        if new_x is None:
            n_parse_failures += 1
            best_idx = int(np.argmin(history_f))
            new_x = history_x[best_idx] + rng.randn(dim) * 0.3
        x = new_x

    final_f = objective(x)
    trace.append(final_f)
    return x, trace[:n_steps], {"n_parse_failures": n_parse_failures}


def _parse_numerical_response(text, dim):
    import json, re
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, list) and len(obj) == dim:
            return np.array([float(v) for v in obj])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = re.search(r"\[([^\]]+)\]", text)
    if m:
        try:
            vals = [float(v.strip().strip('"').strip("'")) for v in m.group(1).split(",")]
            if len(vals) == dim:
                return np.array(vals)
        except ValueError:
            pass
    return None


OPTIMIZER_REGISTRY = {
    "gd": run_gradient_descent,
    "adam": run_adam,
    "nelder_mead": run_nelder_mead,
    "random": run_random_search_numerical,
}


def run_full_benchmark_grid(
    optimizers=("gd", "adam", "nelder_mead", "random"),
    functions=tuple(BENCHMARK_FUNCTIONS.keys()),
    dimensions=tuple(DIMENSIONS),
    n_runs=5, n_steps=100, seed_base=0,
    include_llm=False, ollama_model="qwen2.5:7b",
    ollama_host="http://localhost:11434",
):
    """Replicates ProPS Table 5's grid: functions x dims x optimizers x
    n_runs, sharing the SAME initial points per (function, dim) across all
    optimizers (matching ProPS's stated fair-comparison protocol)."""
    results = {}

    for fn_name in functions:
        fn = BENCHMARK_FUNCTIONS[fn_name]
        results[fn_name] = {}

        for dim in dimensions:
            results[fn_name][dim] = {}
            shifted_fn, offset = make_shifted_objective(fn, dim, seed=seed_base)

            init_rng = np.random.RandomState(seed_base + 1000)
            init_points = [init_rng.uniform(-5, 5, size=dim) for _ in range(n_runs)]

            for opt_name in optimizers:
                opt_fn = OPTIMIZER_REGISTRY[opt_name]
                final_vals = []
                for run_idx in range(n_runs):
                    x0 = init_points[run_idx].copy()
                    if opt_name == "random":
                        _, trace = opt_fn(shifted_fn, x0, n_steps=n_steps,
                                          seed=seed_base + run_idx)
                    else:
                        _, trace = opt_fn(shifted_fn, x0, n_steps=n_steps)
                    final_vals.append(trace[-1])
                results[fn_name][dim][opt_name] = final_vals
                log.info(f"{fn_name} D={dim} {opt_name}: "
                         f"{np.mean(final_vals):.3f} +/- {np.std(final_vals):.3f}")

            if include_llm:
                final_vals = []
                diag_total = {"n_parse_failures": 0}
                for run_idx in range(n_runs):
                    x0 = init_points[run_idx].copy()
                    _, trace, diag = run_llm_numerical_optimizer(
                        shifted_fn, x0, n_steps=n_steps, dim=dim,
                        ollama_model=ollama_model, ollama_host=ollama_host,
                        seed=seed_base + run_idx,
                    )
                    final_vals.append(trace[-1])
                    diag_total["n_parse_failures"] += diag["n_parse_failures"]
                results[fn_name][dim]["llm"] = final_vals
                log.info(f"{fn_name} D={dim} llm: "
                         f"{np.mean(final_vals):.3f} +/- {np.std(final_vals):.3f} "
                         f"(parse_failures={diag_total['n_parse_failures']})")

    return results


def format_results_table(results):
    lines = []
    for fn_name, dims in results.items():
        for dim, optimizers in dims.items():
            row = [f"{fn_name:<12} D={dim:<3}"]
            for opt_name, vals in optimizers.items():
                row.append(f"{opt_name}={np.mean(vals):.2f}+/-{np.std(vals):.2f}")
            lines.append(" | ".join(row))
    return "\n".join(lines)
