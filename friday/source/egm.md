---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Endogenous Grid Method


## Overview

In this lecture we use the endogenous grid method (EGM) to solve a basic income
fluctuation (optimal savings) problem.


```python hide-output=false
# !pip install quantecon  # Uncomment if necessary
```

Imports:

```python hide-output=false
import quantecon as qe
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import numba
from time import time
from typing import NamedTuple
```

Let’s check the GPU we are running

```python hide-output=false
!nvidia-smi
```

We use 64 bit floating point numbers for extra precision.

```python hide-output=false
jax.config.update("jax_enable_x64", True)
```

## Model class

```python hide-output=false
class Model(NamedTuple):
    β: float          # discount factor
    R: float          # gross interest rate
    γ: float          # CRRA preference parameter
    μ: float          # income location parameter
    ν: float          # income volatility parameter
    s_array: jax.Array  # savings grid
    y_draws: jax.Array  # income draws for Monte Carlo
```

## Setup

We consider a household that chooses a state-contingent consumption plan $ \{c_t\}_{t \geq 0} $ to maximize

$$
    \mathbb{E} \, \sum_{t=0}^{\infty} \beta^t u(c_t)
$$

subject to

$$
    a_{t+1} \leq  R(a_t - c_t)  + Y_{t+1},
    \quad c_t \geq 0,
    \quad a_t \geq 0
    \quad t = 0, 1, \ldots
$$

Here $ R = 1 + r $ where $ r $ is the interest rate.

To make our life simple, the income process $ \{Y_t\} $ will be IID and
lognormal:

$$
    Y_t = \exp(\mu + \sigma Z_t) 
    \quad \text{with} \quad 
    Z_t ~ N(0,1)
$$

Utility has the CRRA specification

$$
u(c) = \frac{c^{1 - \gamma}} {1 - \gamma}
$$

The following function creates a model instance with default parameter values.

```python hide-output=false
def create_model(R=1.01,             # gross interest rate
                 β=0.96,             # discount factor
                 γ=1.5,              # CRRA preference parameter
                 μ=-1.0,             # income location parameter
                 ν=0.2,              # income volatility parameter
                 s_max=16,           # savings grid max
                 s_size=200,         # savings grid size
                 y_size=1_000,       # number of income draws
                 seed=42):           # random seed

    # require R β < 1 for convergence
    assert R * β < 1, "Stability condition failed."

    # Create savings grid
    s_array = jnp.linspace(0, s_max, s_size)

    # Generate income draws for Monte Carlo integration
    key = jax.random.PRNGKey(seed)
    z_draws = jax.random.normal(key, (y_size,))
    y_draws = jnp.exp(μ + ν * z_draws)

    # Return Model instance
    return Model(β=β, R=R, γ=γ, μ=μ, ν=ν,
                 s_array=s_array, y_draws=y_draws)
```

## Solution method

The state $a_t$ takes values in  $S := \mathbb R_+$.

We aim to compute an optimal consumption policy $ \sigma^* \colon S \to S$, under which dynamics are given by

$$
    c_t = \sigma^*(a_t)
    \quad \text{and} \quad
    a_{t+1} = R (a_t - c_t) + Y_{t+1}
$$

In this section we discuss how we intend to solve for this policy.

### Euler equation

The Euler equation for the optimization problem is

$$
u' (c_t)
    = \max \left\{
        \beta R \,  \mathbb{E}_t  u'(c_{t+1})  \,,\;  u'(a_t)
    \right\}
$$

An explanation for this expression can be found [here](https://python.quantecon.org/ifp.html#value-function-and-euler-equation).

We rewrite the Euler equation in functional form

$$
    (u' \circ \sigma)  (y)
    = \max 
    \left\{
        \beta R \, \mathbb E (u' \circ \sigma)
            [R (a - \sigma(y)) + Y]
            \; , \; u'(a)
    \right\}
$$

where $ (u' \circ \sigma)(y) := u'(\sigma(y)) $ and $ \sigma $ is a consumption
policy.

For given consumption policy $ \sigma $, we define $ (K \sigma) (y) $ as the unique $ c \in [0, a] $ that solves


$$
    u'(c)
    = \max \left\{
               \beta R \, \mathbb E (u' \circ \sigma) \,
               [R (a - c) + Y]
               \, , \;
               u'(a)
         \right\} 
$$

It [can be shown that](https://python.quantecon.org/ifp.html)

1. iterating with $ K $ computes an optimal policy and  
1. if $ \sigma $ is increasing, then so is $ K\sigma $  


Hence below we always assume that $ \sigma $ is increasing.

The EGM is a technique for computing the update $ K\sigma $ given $ \sigma $ along a grid of asset values.

Notice that the second term in the max above dominates for sufficiently small $ a $.

We have $ c=a $ for all such $ a $.

Hence, for sufficiently small $ a $,

$$
u'(a) \geq
   \beta R \, \mathbb E (u' \circ \sigma)  (Y)
$$

Equality holds at $ \bar a(y) $ given by

$$
\bar a (y) =
   (u')^{-1}
   \left\{
       \beta R \, \mathbb E (u' \circ \sigma)  (Y)
   \right\}
$$

We can now write

$$
u'(c)
    = \begin{cases}
        \beta R \, \mathbb E (u' \circ \sigma) \, [R (a - c) + Y]
               & \text{if } a > \bar a (y) \\
        u'(a)  & \text{if } a \leq \bar a (y)
    \end{cases}
$$

Equivalently, we can state that the $ c $ satisfying $ c = (K\sigma)(y) $ obeys


$$
    c = \begin{cases}
            (u')^{-1}
            \left\{
                \beta R \, \mathbb E (u' \circ \sigma) \,
                   [R (a - c) + Y]
            \right\}
                   & \text{if } a > \bar a (y) \\
                a  & \text{if } a \leq \bar a (y)
        \end{cases} 
$$

We begin with an *exogenous* grid of saving values $ 0 = s_0 < \ldots < s_{N-1} $

Using the exogenous savings grid, we create an *endogenous* asset grid
$ a_0, \ldots, a_{N-1} $ and a consumption grid $ c_0, \ldots, c_{N-1} $ as follows.

First we set $ a_0 = c_0 = 0 $, since zero consumption is an optimal (in fact the only) choice when $ a=0 $.

Then, for $ i > 0 $, we compute

<!-- #region -->
$$
    c_i
    = (u')^{-1}
    \left\{ 
        \beta R \, \mathbb E (u' \circ \sigma) \, [R s_i + Y]
     \right\}
     \quad \text{for all } i 
$$

and we set

$$
    a_i = s_i + c_i
$$


We are now ready to iterate with $ K $.
<!-- #endregion -->

### JAX version

First we define a vectorized operator $ K $ based on the EGM.

Notice in the code below that

- we avoid all loops and any mutation of arrays  
- the function is pure (no globals, no mutation of inputs)  

```python hide-output=false
@jax.jit
def K_egm(a_in, σ_in, model):
    """
    The vectorized operator K using EGM.

    """

    # Unpack
    β, R, γ, μ, ν = model.β, model.R, model.γ, model.μ, model.ν
    s_array = model.s_array
    y_draws = model.y_draws
    
    def u_prime(c):
        return c**(-γ)

    def u_prime_inv(u):
            return u**(-1/γ)

    # Linearly interpolate σ
    def σ(a):
        return jnp.interp(a, a_in, σ_in)

    # Broadcast and vectorize
    def expectation(s):
        # For each savings level s, compute E[u'(σ(R*s + Y))]
        a_next = R * s + y_draws  # shape (y_size,)
        c_next = σ(a_next)  # shape (y_size,)
        return jnp.mean(u_prime(c_next))

    # Vectorize over s_array
    E = jax.vmap(expectation)(s_array)
    c = u_prime_inv(β * R * E)

    # The policy is computed consumption with the first element set to zero
    σ_out = c.at[0].set(0.0)

    # Compute a_out by a = s + c
    a_out = s_array + σ_out
    
    return a_out, σ_out
```

Next we define a successive approximator that repeatedly applies $ K $.

```python hide-output=false
def successive_approx_jax(model,
            tol=1e-5,
            max_iter=100_000,
            verbose=True,
            print_skip=25):

    # Initial condition is to consume all in every state
    σ_init = model.s_array
    a_init = jnp.copy(σ_init)
    a_vec, σ_vec = a_init, σ_init

    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        a_new, σ_new = K_egm(a_vec, σ_vec, model)
        error = jnp.max(jnp.abs(σ_vec - σ_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")
        a_vec, σ_vec = jnp.copy(a_new), jnp.copy(σ_new)

    if error > tol:
        print("Failed to converge!")
    else:
        print(f"\nConverged in {i} iterations.")

    return a_new, σ_new
```

## Solutions

Here we solve the IFP with JAX and Numba.

We will compare both the outputs and the execution time.


### Outputs

```python
model = create_model()
```

Here’s a first run of the JAX code.

```python hide-output=false
%%time
a_star, c_star = successive_approx_jax(model,
                                       print_skip=1000)
```

Now let's check the outputs in a plot to make sure they are the same.

```python hide-output=false
# Compute optimal savings
s_star = a_star - c_star

fig, ax = plt.subplots()

ax.plot(
   a_star,
   c_star,
   label="consumption policy"
)
ax.plot(
   a_star,
   s_star,
   label="savings policy"
)
ax.plot(a_star, a_star, 'k--', alpha=0.5, label="45 degree line")

ax.set_xlabel('assets')
ax.set_ylabel('consumption / savings')
plt.legend()
plt.show()
```

## Simulation

Having computed the optimal consumption policy, we can now simulate the model forward.

We start by simulating a single household over time to see how consumption and assets
evolve in response to random income shocks.

The household uses the optimal policy $\sigma^*$ to choose consumption at each date,
given current assets:

$$
    c_t = \sigma^*(a_t)
$$

Assets then evolve according to

$$
    a_{t+1} = R(a_t - c_t) + Y_{t+1}
$$

where $Y_{t+1}$ is the random income realization.

```python hide-output=false
def simulate(model, a_grid, c_policy, a_0=1.0, T=100, seed=123):
    """
    Simulate the optimal consumption path.

    Parameters:
    -----------
    model : Model
        The model instance
    a_grid : array
        Grid of asset values from solution
    c_policy : array
        Optimal consumption policy on the grid
    a_0 : float
        Initial asset level
    T : int
        Number of periods to simulate
    seed : int
        Random seed for income shocks
    """

    # Generate income shocks for simulation
    key = jax.random.PRNGKey(seed)
    z_sim = jax.random.normal(key, (T,))
    y_sim = jnp.exp(model.μ + model.ν * z_sim)

    # Initialize arrays
    a_path = jnp.zeros(T+1)
    c_path = jnp.zeros(T)

    a_path = a_path.at[0].set(a_0)

    # Simulate forward
    for t in range(T):
        # Interpolate consumption policy
        c_t = jnp.interp(a_path[t], a_grid, c_policy)
        c_path = c_path.at[t].set(c_t)

        # Update assets
        s_t = a_path[t] - c_t
        a_path = a_path.at[t+1].set(model.R * s_t + y_sim[t])

    return a_path, c_path, y_sim

# Run simulation
a_path, c_path, y_path = simulate(model, a_star, c_star)

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot consumption and income
ax1.plot(c_path, label='consumption')
ax1.plot(y_path, alpha=0.5, label='income')
ax1.set_xlabel('time')
ax1.set_ylabel('value')
ax1.set_title('Consumption and Income Over Time')
ax1.legend()

# Plot assets
ax2.plot(a_path)
ax2.set_xlabel('time')
ax2.set_ylabel('assets')
ax2.set_title('Asset Evolution')

plt.tight_layout()
plt.show()
```

## Cross-sectional distribution

The plot above shows the time series for a single household.

Now let's simulate a large cross-section of households to study the stationary
wealth distribution.

We simulate $N = 5000$ households forward for $T = 500$ periods.

All households start with the same initial assets but face independent income
shocks drawn from the lognormal distribution.

Each household follows the same optimal policy $\sigma^*$, but the different
income realizations lead to heterogeneity in the wealth distribution.

The final cross-section of assets provides an approximation to the stationary
distribution of wealth under the optimal policy.

```python hide-output=false
def simulate_cross_section(model, a_grid, c_policy, a_0=1.0, T=500, N=5_000, seed=456):
    """
    Simulate a cross-section of households.

    Parameters:
    -----------
    model : Model
        The model instance
    a_grid : array
        Grid of asset values from solution
    c_policy : array
        Optimal consumption policy on the grid
    a_0 : float
        Initial asset level for all households
    T : int
        Number of periods to simulate
    N : int
        Number of households
    seed : int
        Random seed for income shocks
    """

    # Generate income shocks for all households
    key = jax.random.PRNGKey(seed)
    z_sim = jax.random.normal(key, (N, T))
    y_sim = jnp.exp(model.μ + model.ν * z_sim)

    # Initialize asset holdings
    a_current = jnp.ones(N) * a_0

    # Simulate forward
    for t in range(T):
        # Interpolate consumption policy for all households
        c_current = jnp.interp(a_current, a_grid, c_policy)

        # Update assets
        s_current = a_current - c_current
        a_current = model.R * s_current + y_sim[:, t]

    return a_current

# Run cross-sectional simulation
print("Simulating cross-section...")
a_final = simulate_cross_section(model, a_star, c_star)

# Plot histogram with kernel density
from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(a_final, bins=50, alpha=0.5, edgecolor='black', density=True, label='histogram')

# Add kernel density estimate
kde = gaussian_kde(a_final)
a_range = jnp.linspace(jnp.min(a_final), jnp.max(a_final), 200)
ax.plot(a_range, kde(a_range), 'r-', linewidth=2, label='kernel density')

ax.set_xlabel('assets')
ax.set_ylabel('density')
ax.set_title('Cross-Sectional Distribution of Assets (N=5,000, T=500)')
ax.legend()
plt.show()
```


