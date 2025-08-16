import numpy as np
from scipy.optimize import minimize

def pso_optimize(mu, Sigma, prev_weights, cardinality=CARDINALITY):
    """PSO optimization with transaction cost and diversity penalties"""
    n = len(mu)
    mu = np.array(mu, dtype=float)
    Sigma = np.array(Sigma, dtype=float)
    def objective(w):
        portfolio_return = mu @ w
        portfolio_risk = np.sqrt(w @ Sigma @ w)
        sharpe = (portfolio_return - RISK_FREE_RATE/12) / (portfolio_risk + 1e-10)
        to = sum(abs(w[i] - prev_weights.get(i, 0)) for i in range(n)) if prev_weights else 0
        tc_penalty = TRANSACTION_COST * to
        hhi = np.sum(w**2)
        diversity_penalty = 0.1 * hhi
        card_penalty = 0.1 * abs(sum(w > 0.01) - cardinality)
        return -sharpe + tc_penalty + diversity_penalty + card_penalty
    particles = np.random.uniform(0, MAX_WEIGHT, (NUM_PARTICLES, n))
    particles = particles / particles.sum(axis=1)[:, None]
    velocities = np.random.uniform(-0.1, 0.1, (NUM_PARTICLES, n))
    pbest = particles.copy()
    pbest_scores = np.array([objective(p) for p in pbest])
    gbest = pbest[np.argmin(pbest_scores)]
    gbest_score = min(pbest_scores)
    for _ in range(MAX_ITERATIONS):
        w1, w2, c = 0.5, 0.5, 0.5
        r1, r2 = np.random.random((2, NUM_PARTICLES, n))
        velocities = (w1 * velocities +
                     w2 * r1 * (pbest - particles) +
                     c * r2 * (gbest - particles))
        particles = np.clip(particles + velocities, 0, MAX_WEIGHT)
        particles = particles / particles.sum(axis=1)[:, None]
        scores = np.array([objective(p) for p in particles])
        update_mask = scores < pbest_scores
        pbest[update_mask] = particles[update_mask]
        pbest_scores[update_mask] = scores[update_mask]
        if min(scores) < gbest_score:
            gbest = particles[np.argmin(scores)]
            gbest_score = min(scores)
    w = np.maximum(gbest, 0)
    w[w < 0.01] = 0
    return w / (w.sum() + 1e-10)

def mvo_optimize(mu, Sigma):
    """Standard MVO optimization"""
    n = len(mu)
    mu = np.array(mu, dtype=float)
    Sigma = np.array(Sigma, dtype=float)
    Sigma += np.eye(n) * 1e-6
    def objective(w):
        return float(w @ Sigma @ w) - 0.5 * mu @ w
    bounds = [(0.0, MAX_WEIGHT)] * n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    x0 = np.ones(n)/n
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
    w = np.maximum(res.x, 0)
    w = w / (w.sum() + 1e-10)
    w[w < 0.01] = 0
    return w / w.sum()

def select_by_momentum(prices_train, top_k=CARDINALITY):
    """Select assets based on momentum"""
    returns = prices_train.pct_change().dropna()
    momentum = (prices_train.ffill().rolling(3).mean() /
               prices_train.ffill().rolling(12).mean() - 1).iloc[-1]
    df = pd.DataFrame({'momentum': momentum}).dropna()
    df = df.sort_values('momentum', ascending=False)
    return list(df.index[:top_k])
