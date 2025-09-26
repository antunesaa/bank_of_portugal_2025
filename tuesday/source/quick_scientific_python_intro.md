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

`numpy` is one of the core scientific libraries used in Python.

The main object that it introduces is an "array" type. This array type allows for users to represent vectors, matrices, and higher dimensional arrays.

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
# Python indexing starts at 0
x[0]
```

```{code-cell} ipython3
y[:, 1]  # Select column 1 (not col 0!)
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

**Create a vector filled with n evenly spaced numbers from x to z**

```{code-cell} ipython3
np.linspace(0, 5, 11)
```

**Create a vector filled with U(0, 1)**

```{code-cell} ipython3
np.random.rand(2, 3)
```

**Create a vector filled with N(0, 1)**

```{code-cell} ipython3
np.random.randn(2, 2, 3)
```

### Operations on Arrays

Operations on arrays are typically element by element.

```{code-cell} ipython3
z = np.full(3, 10.0)
print(f"    x = {x}")
print(f"    z = {z}")
print(f"z + x = {z + x}")
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

```{code-cell} ipython3
z = np.ones((3, 1))* 10
print(z)
print()
print(y)
print()
print(z + y)
```

```{code-cell} ipython3
# Matrix multiplication
y @ y.T
```

### Numpy array functions

We are often interested in computing various functions of a single array -- Something like `sum` or `mean`.

`numpy` has many array functions built in. Many of these functions are something known as a "reduction" -- This just means it takes many inputs and returns a single output (think about what computing the mean does). Reductions can often be applied either to the entire array or to a single axis. If you apply it to a single axis, that axis will get collapsed into a single value.

Here we demonstrate a few of the most common array functions and some reductions:

```{code-cell} ipython3
# Cumulative sum
np.cumsum(x)
```

```{code-cell} ipython3
# One element differences
np.diff(x)
```

```{code-cell} ipython3
# Mean (vector)
np.mean(x)
```

```{code-cell} ipython3
# Mean (matrix)
np.mean(y)
```

```{code-cell} ipython3
# Mean on a matrix but collapsing the rows
# y is size (3, 2) and np.mean(y, axis=0) is size (2,)
np.mean(y, axis=0)
```

```{code-cell} ipython3
# Standard deviation on a 3 dimensional array collapsing on the
# 3rd dimension
np.std(z, axis=2)
```

### Universal functions

One of the powerful tools that `numpy` opens to users is "universal functions" (ufuncs).

These are functions that operate directly on n-dimensional arrays in an element-by-element fashion.

Not only does this make it simple to apply a function to an entire array but, behind the scenes, there is a significant amount of optimization and multithreading happening. There are lots of ufuncs available to you in `numpy` -- Take a peek at [the documentation](https://docs.scipy.org/doc/numpy/reference/ufuncs.html?highlight=ufunc#available-ufuncs) for a list

```{code-cell} ipython3
# Computes sin(x) for each element of x
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

fig.set_facecolor("red")
ax.set_facecolor("blue")
```

### More

+++

**Scatter plots**

```{code-cell} ipython3
x = np.random.randn(5_000)
y = np.random.randn(5_000)

fig, ax = plt.subplots()

ax.scatter(x, y, color="DarkBlue", alpha=0.05, s=25)
```

**Line plots**

```{code-cell} ipython3
x = np.linspace(0, 10)
y = np.sin(x)

fig, ax = plt.subplots()

ax.plot(x, y, linestyle="-", color="k")
ax.plot(x, 2*y, linestyle="--", color="k")

# Bonus - Fill between two lines
ax.fill_between(x, y, 2*y, color="LightBlue", alpha=0.3)
```

**Bar plots**

```{code-cell} ipython3
x = ["Red", "Blue", "Green"]
y = [5, 10, 15]

fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(x, y, color=["r", "b", "g"])
```

**Histograms**

```{code-cell} ipython3
x = np.random.randn(5000)

fig, ax = plt.subplots()

ax.hist(x, bins=25, density=True)
```

## Scipy

`scipy` is a package that is closely related to `numpy`.

While `numpy` introduces the array type and some basic functionality on top of that array, `scipy` extends these arrays further by providing higher level functionality with access to a variety of useful tools for science.

+++

### Interpolation

Within economics, we often need to approximate functions where we only observe a finite set of values that the function can take.

We "interpolate" between these points in a number of ways. `scipy.interpolate` provides a convenient interface to perform this interpolation in many of the most common interpolation routines.

```{code-cell} ipython3
import scipy.interpolate as interp
```

```{code-cell} ipython3
x = np.linspace(0.25, 10.0, 15)
y = np.log(x)
```

**Piecewise linear interpolation**

```{code-cell} ipython3
# Can do piecewise linear with numpy
x_interp = np.linspace(0.0, 11, 100)
y_interp = np.interp(x_interp, x, y)

fig, ax = plt.subplots()

ax.scatter(x, y, color="r", s=20)
ax.plot(x_interp, y_interp, color="k", linewidth=0.5)
```

**Piecewise cubic**

```{code-cell} ipython3
f = interp.interp1d(x, y, kind="cubic", fill_value="extrapolate")

y_interp = f(x_interp)

fig, ax = plt.subplots()

ax.scatter(x, y, color="r", s=20)
ax.plot(x_interp, y_interp, color="k", linewidth=0.5)
```

**Other**

```{code-cell} ipython3
f = interp.PchipInterpolator(x, y, extrapolate=True)

y_interp = f(x_interp)

fig, ax = plt.subplots()

ax.scatter(x, y, color="r", s=20)
ax.plot(x_interp, y_interp, color="k", linewidth=0.5)
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

```{code-cell} ipython3
la.cholesky(X)
```

```{code-cell} ipython3
la.solve(X, np.array([0.0, 0.5, 0.3]))

# Check out all of the other solve options based
# on the shape of the matrix
# la.solve_circulant
# la.solve_toeplitz
# la.solve_banded
# la.solve_triangular
```

```{code-cell} ipython3
la.eigvals(X)
```

```{code-cell} ipython3
Q, R = la.qr(X)
print(Q, R)
```

```{code-cell} ipython3
X
```

```{code-cell} ipython3
Q@R
```

```{code-cell} ipython3
la.inv(X)@X
```

### Statistics

We often want to work with various probability distributions. We could code up the pdf or a sampler ourselves but this work is largely done for us within `scipy.statistics`.

The one warning we include here is that sometimes `scipy.stats` uses a non-canonical representation of the distribution.

```{code-cell} ipython3
import scipy.stats as st
```

**Normal distribution**

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
st.norm.fit(np.random.randn(5000))
```

**Exponential**

Note: `scipy.stats` classifies this in terms of a "standardized form"

$$f(x) = \exp(-x)$$

If you want to use the "non-standardized form" then you use the `loc` and `scale` parameters

Let $y = \frac{(x - \text{loc})}{\text{scale}}$. Then `expon(loc=loc, scale=scale).pdf(x) = expon().pdf(x) / scale`

The classical characterization (i.e. $f(x) = \exp(-\lambda x)$) is equivalent to choosing `scale = 1 / lambda`.

```{code-cell} ipython3
d = st.expon(scale=2)
```

```{code-cell} ipython3
# Draw random samples
d.rvs(25)
```

```{code-cell} ipython3
# Probability density function
d.pdf(-1.0)
```

```{code-cell} ipython3
# Cumulative density function
d.cdf(5.0)
```

```{code-cell} ipython3
# Fit random data
st.expon.fit(np.array([0.5, 0.25, 0.75, 0.1, 0.2, 1.5]))
```

```{code-cell} ipython3
st.expon.fit(np.array([0.5, 0.25, 0.75, 0.1, 0.2, 1.5]), floc=0)
```

```{code-cell} ipython3
st.expon.fit(np.array([0.5, 0.25, 0.75, 0.0, 0.2, 1.5]))
```

**Pareto**

Note: `scipy.stats` classifies this in terms of a "standardized form"

$$f(x, b) = \frac{b}{x^{b+1}$$

If you want to use the "non-standardized form" then you use the `loc` and `scale` parameters

Let $y = \frac{(x - \text{loc})}{\text{scale}}$. Then `pareto(b, loc=loc, scale=scale).pdf(x) = pareto(b).pdf(y) / scale`

```{code-cell} ipython3
d = st.pareto(b=2.0, loc=0, scale=1)
```

```{code-cell} ipython3
# Draw random samples
d.rvs(25)
```

```{code-cell} ipython3
# Probability density function
d.pdf(5.0)
```

```{code-cell} ipython3
# Cumulative density function
d.cdf(5.0)
```

```{code-cell} ipython3
# Fit random data
st.pareto.fit(np.array([2.5, 4.25, 7.75, 3.1, 1.2, 0.5]), floc=0, fscale=0)
```

## Numba

`numba` is a very exciting, and powerful package, that brings "just-in-time" (JIT) compilation technology to Python.

+++

**Brief background**

You may have heard about the differences between "compiled programming languages" and "interpreted programming languages"

* A compiled language is run in a few steps:
  * Programmer writes the code
  * Compiler converts that code into machine code
  * Computer runs machine code. Note that once the code is compiled, it can be run whenever one wants without the compilation step
* An interpreted language runs code differently:
  * Programmer writes code
  * Computer "runs" the code by
    * An "interpreter" reads the code line-by-line
    * For each line, the interpreter figures out what the inputs are and tries to convert it to machine code
    * Computer runs the machine code

+++

**Pros and cons of compiled**

* Once the compiler has run, the code is already machine code and runs very fast (as fast as possible given the code you wrote)
* For very large programs, compilation requires the upfront cost of compilation which can take minutes/hours
* Compiled programs can only be shared within similar hardware architecture and operating systems (though as long as there's a compiler for the hardware/OS, one could recompile the code)

**Pros and cons of interpreted**

* As long as there is an interpreter for the hardware/operating system, interpreted code can be easily shared
* Significantly slower than compiled code because of the back and forth to read the code line-by-line (which has to be redone each time the code is run!)
* Easier to interact with your code (and more importantly, your data!) because you can run one line at a time

+++

**How different are they in speed?**

+++

_Python_

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

_Fortran_

```{code-cell} ipython3
#pip install mason ninja  # Uncomment if you wish
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

Clearly Fortran is much faster!

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
* Many (but not all!) NumPy functions -- This includes `np.interp`!

For more information, read these sections from the documentation

* [Supported Python features](https://numba.readthedocs.io/en/stable/reference/pysupported.html)
* [Supported NumPy  features](https://numba.readthedocs.io/en/stable/reference/numpysupported.html)

```{code-cell} ipython3
import numba
```

```{code-cell} ipython3
calculate_pi_numba = numba.jit(calculate_pi_python, nopython=True)
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
@numba.jit(nopython=True, parallel=True)
def calculate_pi_parallel(n=1_000_000):
    """
    Approximates pi by drawing two random numbers and
    determining whether the of the sum of their squares
    is less than one (which tells us if the points are
    in the upper-right quadrant of the unit circle). The
    fraction of draws in the upper-quadrant approximates
    the area which we can then multiply by 4 to get the
    area of the circle (which is pi since r=1)
    """

    # Iterate for many sample
    in_circ = 0
    for i in numba.prange(n):
        # Draw random numbers
        x = np.random.random()
        y = np.random.random()

        if (x**2 + y**2) < 1:
            in_circ += 1

    return 4 * (in_circ / n)
```

```{code-cell} ipython3
%%timeit

calculate_pi_parallel(1_000_000)
```

Small side note... A warning to beware of race conditions

Any time that you write parallel code, you must understand how each computation can affect another. If you don't consider this carefully, your output could depend on the order in which each computation finishes! This is known as a "race condition" and is _very very bad_ because it creates a non-determinism in your code.

Why is it so bad? It's possible that your code returns the right answer sometimes and the wrong answer others -- This non-determinism makes it difficult to debug

```{code-cell} ipython3
@numba.jit(parallel=True)
def dumb_parallel_function(n=500_000):
    x = np.zeros(n)
    for i in numba.prange(n):
        x[0] = i
        x[i] = i

    return x
```

```{code-cell} ipython3
dumb_parallel_function(5)
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
