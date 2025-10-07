---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## Quick scientific Python introduction

**Prepared for the Bank of Portugal Computational Economics Course (Oct 2025)**

**Author:** [John Stachurski](https://johnstachurski.net)

This notebook does a "whirlwind tour" of the functionality that the scientific Python stack exposes.

+++

## Numpy

```{code-cell} ipython3
import numpy as np
```

NumPy is one of the core scientific libraries used in Python.

It introduces an "array" type. 

This array type allows for users to represent vectors, matrices, and higher dimensional arrays.

```{code-cell} ipython3
x = np.array([0.0, 1.0, 2.0])

y = np.array([[0.0, 1.0], 
              [2.0, 3.0], 
              [4.0, 5.0]])
```

```{code-cell} ipython3
x
```

```{code-cell} ipython3
y
```

```{code-cell} ipython3
y.shape
```

```{code-cell} ipython3
y.T
```

```{code-cell} ipython3
y.T.shape
```

**Indexing**

+++

We can select elements out of the array by indexing into the arrays

```{code-cell} ipython3
x[0]
```

```{code-cell} ipython3
y[:, 1]  # Select column 1 
```

### Special array creation methods

+++

**Create an empty array**

```{code-cell} ipython3
np.empty((5, 2))
```

**Create an array filled with zeros**

```{code-cell} ipython3
np.zeros(10)
```

**Create an array filled with ones**

```{code-cell} ipython3
np.ones((2, 5))
```

**Create a vector filled with numbers from i to n**

```{code-cell} ipython3
np.arange(1, 7)
```

**Create a vector filled with n evenly spaced numbers**

```{code-cell} ipython3
n = 11
np.linspace(0, 5, n)
```

**Create a vector filled with U(0, 1)**

```{code-cell} ipython3
np.random.rand(2, 3)
```


```{code-cell} ipython3
np.random.rand(2, 3)   # Redraw
```

**Create a vector filled with N(0, 1)**

```{code-cell} ipython3
np.random.randn(2, 2, 3)
```

```{code-cell} ipython3
np.random.randn(2, 2, 3)  # Redraw
```

### Operations on Arrays

Operations on arrays are typically element by element.

```{code-cell} ipython3
z = np.full(3, 10.0)
print(f"    x = {x}")
print(f"    z = {z}")
print(f"z + x = {z + x}")
print(f"z - x = {z - x}")
print(f"z * x = {z * x}")
print(f"z / x = {z / x}")
print(f"z**x  = {z ** x}")
```

**Operations between scalars and arrays**

These operations do mostly what you would expect -- They apply the scalar operation to each individual element of the array.

```{code-cell} ipython3
x
```

```{code-cell} ipython3
x + 1
```

```{code-cell} ipython3
x * 3
```

```{code-cell} ipython3
x - 3
```

#### Operations between arrays of different sizes

In general, for pointwise operations between arrays of different shape, NumPy uses *broadcasting* rules.

```{code-cell} ipython3
z = np.ones((3, 1)) * 10
```

```{code-cell} ipython3
z
```

```{code-cell} ipython3
y
```

```{code-cell} ipython3
z + y
```

#### Matrix Multiplication

```{code-cell} ipython3
print(y)
print(y @ y.T)
```

We can use `@` for inner products too, even for "flat" arrays.

```{code-cell} ipython3
z = np.random.randn(10)
print(f"The shape of z is {z.shape}")
print("Explicit inner product:", np.sum(z * z))
print("Using @", z @ z)
```

### Reductions

In NumPy-speak, *reductions* are functions that map an array into a single value.

Here we demonstrate a few of the most common array functions and some reductions:

```{code-cell} ipython3
np.cumsum(x)
```

```{code-cell} ipython3
np.mean(z)
```

```{code-cell} ipython3
np.var(z)
```

```{code-cell} ipython3
np.std(np.random.randn(100_000))
```

### Universal functions

Universal functions ("ufuncs") are maps scalar-to-scalar maps
that can also act on n-dimensional arrays, acting element-by-element.

```{code-cell} ipython3
np.sin(x)
```

```{code-cell} ipython3
np.exp(x)
```

## Matplotlib

The "default" plotting package for most of the Python world is `matplotlib`.

It is a very flexible package and allows for creating very good looking graphs (in spite of relatively simple defaults)

```{code-cell} ipython3
import matplotlib.pyplot as plt
```

### Figure/Axis

The main pieces of a graph in `matplotlib` are a "figure" and an "axis". We’ve found that the easiest way for us to distinguish between the figure and axis objects is to think about them as a framed painting.

The axis is the canvas; it is where we “draw” our plots.

The figure is the entire framed painting (which inclues the axis itself!).

We can see this difference by setting certain elements of the figure to different colors.

```{code-cell} ipython3
fig, ax = plt.subplots()

fig.set_facecolor("green")
ax.set_facecolor("blue")
```

### More

+++

**Scatter plots**

```{code-cell} ipython3
x = np.random.randn(5_000)
y = np.random.randn(5_000)

fig, ax = plt.subplots()

ax.scatter(x, y, color="DarkBlue", alpha=0.05, s=25);
```

**Line plots**

```{code-cell} ipython3
x = np.linspace(0, 10)
y = np.sin(x)

fig, ax = plt.subplots()

ax.plot(x, y, linestyle="-", color="k")
ax.plot(x, 2*y, linestyle="--", color="k")

# Bonus - Fill between two lines
ax.fill_between(x, y, 2*y, color="LightBlue", alpha=0.3);
```

**Bar plots**

```{code-cell} ipython3
x = np.arange(10)
y = np.cos(x)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x, y);
```

**Histograms**

```{code-cell} ipython3
x = np.random.randn(5000)
fig, ax = plt.subplots()
ax.hist(x, bins=25, density=True, edgecolor='k', alpha=0.5);
```

**Plotting piecewise linear interpolation**

```{code-cell} ipython3
x = np.linspace(0.25, 10.0, 15)  # interpolation points on x axis
y = np.sin(x)                    # interpolation points on y axis

x_interp = np.linspace(0.0, 11, 100)    # x points where we seek interpolated values
y_interp = np.interp(x_interp, x, y)    # interpolated values

fig, ax = plt.subplots()
ax.scatter(x, y, color="r", s=20, label='interpolation points')
ax.plot(x_interp, y_interp, alpha=0.5, lw=2, label='interpolated values')
ax.set_yticks((-1, 0, 1))
ax.legend();
```

## Scipy

`scipy` is a package that is closely related to `numpy`.

While `numpy` introduces the array type and some basic functionality on top of that array, `scipy` extends these arrays further by providing higher level functionality with access to a variety of useful tools for science.

+++

### Interpolation

NumPy provides basic linear interpolation, as shown above.

The function `scipy.interpolate` provides more options

```{code-cell} ipython3
import scipy.interpolate as interp
```

**Piecewise cubic**

```{code-cell} ipython3
f = interp.interp1d(x, y, kind="cubic", fill_value="extrapolate")
y_interp = f(x_interp)

fig, ax = plt.subplots()
ax.scatter(x, y, color="r", s=20)
ax.plot(x_interp, y_interp, lw=2, alpha=0.5);
```

**Other**

```{code-cell} ipython3
f = interp.PchipInterpolator(x, y, extrapolate=True)
y_interp = f(x_interp)

fig, ax = plt.subplots()
ax.scatter(x, y, color="r", s=20)
ax.plot(x_interp, y_interp, linewidth=2, alpha=0.5)
```

### Linear algebra

Linear algebra is a core component of many toolkits. `numpy` itself has a small set of core operations that are within a package called `numpy.linalg` but `scipy.linalg` contains a superset of those operations.

```{code-cell} ipython3
import scipy.linalg as la
```

**Lots of your standard linear algebra tools**

```{code-cell} ipython3
X = np.array([
    [0.5, 0.3, 0.0],
    [0.3, 0.5, 0.4],
    [0.0, 0.4, 0.75]
])
```

Here's the Cholesky decomposition

```{code-cell} ipython3
L = la.cholesky(X)
```

```{code-cell} ipython3
L.T @ L
```

Here's how we solve $X b = y$ for $b$

```{code-cell} ipython3
y = np.array((0.0, 0.5, 0.3))
la.solve(X, y)
```

Get eigenvalues:

```{code-cell} ipython3
la.eigvals(X)
```

QR decomposition:

```{code-cell} ipython3
Q, R = la.qr(X)
print(Q, R)
```

```{code-cell} ipython3
X
```

```{code-cell} ipython3
Q @ R
```

Here's how to compute an inverse:

```{code-cell} ipython3
la.inv(X) @ X
```

```{code-cell} ipython3
np.allclose(la.inv(X) @ X, np.identity(3))
```

### Statistics

We often want to work with various probability distributions. 

We could code up the pdf or a sampler ourselves but this work is largely done for us within `scipy.statistics`.

```{code-cell} ipython3
import scipy.stats as st
```

```{code-cell} ipython3
# location specifies the mean / scale specifies the standard deviation
d = st.norm(loc=2.0, scale=4.0)
```

```{code-cell} ipython3
# Draw random samples
d.rvs(25)
```

```{code-cell} ipython3
# Probability density function
d.pdf(0.5)
```

```{code-cell} ipython3
# Cumulative density function
d.cdf(2.0)
```

```{code-cell} ipython3
# Fit a normal rv to N(0, 1) data
st.norm.fit(12 + 4 * np.random.randn(100_000))
```

SciPy's API for working with probability distributions is a bit weird but the code is stable and well-written.

+++

## Numba

Numba is a powerful package that brings "just-in-time" (JIT) compilation technology to Python.

```{code-cell} ipython3
import numpy as np


def calculate_pi_python(n=1_000_000):
    """
    Approximates π as follows:

    For a circle of radius 1/2, area = π r^2 = π / 4 

    Hence π = 4 area.

    We estimate the area of a circle C of radius 1/2 inside the unit square S = [0, 1] x [0, 1].

        area = probability that a uniform distribution on S assigns to C
        area is approximately fraction of uniform draws in S that fall in C

    Then we estimate π using the formula above.
    """
    in_circ = 0

    for i in range(n):
        # Draw (x, y) uniformly on S
        x = np.random.random()
        y = np.random.random()
        # Increment counter if (x, y) falls in C
        if np.sqrt((x - 0.5)**2 + (y - 0.5)**2) < 1/2:
            in_circ += 1

    approximate_area = in_circ / n

    return 4 * approximate_area
```

```{code-cell} ipython3
%%timeit

calculate_pi_python(1_000_000)
```

To get a idea of how fast this is, let's compare it to a Fortran version.

The following code allows us to call Fortran code from within Jupyter.

```{code-cell} ipython3
#pip install meson ninja  # Uncomment if you wish
```

```{code-cell} ipython3
#pip install fortran-magic  # Uncomment if you wish
```

```{code-cell} ipython3
#sudo apt install gfortran  # Uncomment for Ubuntu or search for instructions for your OS
```

```{code-cell} ipython3
%load_ext fortranmagic
```

```{code-cell} ipython3
%%fortran

subroutine calculate_pi_fortran(n, pi_approx)
    implicit none
    integer, intent(in) :: n
    real, intent(out) :: pi_approx
    
    integer :: in_circ, i
    real :: x, y, distance
    real :: approximate_area

    in_circ = 0
    
    CALL RANDOM_SEED
    DO i = 1, n
        ! Draw (x, y) uniformly on unit square [0,1] x [0,1]
        CALL RANDOM_NUMBER(x)
        CALL RANDOM_NUMBER(y)
        
        ! Calculate distance from center (0.5, 0.5)
        distance = SQRT((x - 0.5)**2 + (y - 0.5)**2)
        
        ! Increment counter if (x, y) falls in circle of radius 1/2
        IF (distance < 0.5) in_circ = in_circ + 1
    END DO

    ! Estimate area and then π
    approximate_area = REAL(in_circ) / REAL(n)
    pi_approx = 4.0 * approximate_area
end subroutine calculate_pi_fortran

```

```{code-cell} ipython3
%%timeit

calculate_pi_fortran(1_000_000)
```

Clearly Fortran is much faster that the Python code.

So should we be using Fortran?

In general, no --- the next section explains.

+++

**JIT compilation**

JIT is a relatively modern development which has the goal of bridging some of the gaps between compiled and interpreted.

Rather than compile the code ahead of time or interpreting line-by-line, JIT compiles small chunks of the code right before it runs them.

For example, recall the function `mc_approximate_pi_python` (that we wrote earlier) that approximates the value of pi using Monte-carlo methods... We might even want to run this function multiple times to average across the approximations. The way that JIT works is,

1. Check the input types to the function
2. The first time it sees particular types of inputs to the function, it compiles the function assuming those types as inputs and stores this compiled code
3. The computer then runs the function using the compiled code -- If it has seen these inputs before, it can jump directly to this step.

`numba` is a package that will empower Python with "JIT super powers"

+++

**What works within Numba?**

* Almost all core Python objects. including: lists, tuples, dictionaries, integers, floats, strings
* Python logic, including: `if.. elif.. else`, `while`, `for .. in`, `break`, `continue`
* NumPy arrays
* Many (but not all!) NumPy functions 

```{code-cell} ipython3
import numba
```

```{code-cell} ipython3
calculate_pi_numba = numba.jit(calculate_pi_python)
```

```{code-cell} ipython3
%%time

calculate_pi_numba(1_000_000)
```

```{code-cell} ipython3
%%timeit

calculate_pi_numba(1_000_000)
```

**Writing parallel code with numba**

```{code-cell} ipython3
@numba.jit(parallel=True)
def calculate_pi_parallel(n=1_000_000):
    in_circ = 0

    for i in numba.prange(n):
        # Draw (x, y) uniformly on S
        x = np.random.random()
        y = np.random.random()
        # Increment counter if (x, y) falls in C
        if np.sqrt((x - 0.5)**2 + (y - 0.5)**2) < 1/2:
            in_circ += 1

    approximate_area = in_circ / n

    return 4 * approximate_area
```

```{code-cell} ipython3
calculate_pi_parallel(1_000_000)
```

```{code-cell} ipython3
%%timeit

calculate_pi_parallel(1_000_000)
```

**Writing GPU code with numba**

```{code-cell} ipython3
import numpy as np

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
```

```{code-cell} ipython3
@cuda.jit
def compute_pi(rng_states, n, out):
    thread_id = cuda.grid(1)

    # Compute pi by drawing random (x, y) points and finding
    # the fraction that lie inside the unit circle
    inside = 0
    for i in range(n):
        x = xoroshiro128p_uniform_float32(rng_states, thread_id)
        y = xoroshiro128p_uniform_float32(rng_states, thread_id)
        if x**2 + y**2 <= 1.0:
            inside += 1

    out[thread_id] = 4.0 * inside / n
```

```{code-cell} ipython3
%%time

threads_per_block = 64
blocks = 32

n = 500

rng_states = create_xoroshiro128p_states(threads_per_block*blocks, seed=3252024)
out = np.zeros(threads_per_block*blocks, dtype=np.float32)

compute_pi[blocks, threads_per_block](rng_states, n, out)
```

```{code-cell} ipython3
print("As if we sampled: ", threads_per_block*blocks*n)
print("Pi: ", out.mean())
```
