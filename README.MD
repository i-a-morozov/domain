# domain, 2025-2026
#

<p align="center">
  <img width="500" height="250" src="https://github.com/i-a-morozov/domain/blob/main/data/cat.png">
</p>

Computation of center stability **domain** by bounding unstable cloud.

# Install & build

```
$ pip install git+https://github.com/i-a-morozov/domain.git@main
```
or 
```
$ pip install domain -U
```

# Hénon map

$$
\begin{aligned}
q_x' &= \cos(2\pi \nu_x)\, q_x + \sin(2\pi \nu_x)\left(p_x + q_x^2 - q_y^2 + \mu (q_x^3 - 3 q_x q_y^2)\right) \\
q_y' &= \cos(2\pi \nu_y)\, q_y + \sin(2\pi \nu_y)\left(p_y - 2 q_x q_y + \mu (-3 q_x^2 q_y + q_y^3)\right) \\
p_x' &= \cos(2\pi \nu_x)\left(p_x + q_x^2 - q_y^2 + \mu (q_x^3 - 3 q_x q_y^2)\right) - \sin(2\pi \nu_x)\, q_x \\
p_y' &= \cos(2\pi \nu_y)\left(p_y - 2 q_x q_y + \mu (-3 q_x^2 q_y + q_y^3)\right) - \sin(2\pi \nu_y)\, q_y
\end{aligned}
$$

Precomputed [3D projections](https://henon4d.streamlit.app/).

Doesn't work on mobile, requires several GBs of RAM to display 500k+ points in 3D plot.

[3D projection scan video](https://www.youtube.com/watch?v=EDtpyUOtCRg)

# Froschlé map

$$
\begin{aligned}
q_x' &= q_x + p_x \\
q_y' &= q_y + p_y \\
p_x' &= p_x + \frac{k_x}{2\pi}\sin(2\pi  q_x') + k_{xy}\sin\bigl(2\pi( q_x' +  q_y')\bigr) \\
p_y' &= p_y + \frac{k_y}{2\pi}\sin(2\pi  q_y') + k_{xy}\sin\bigl(2\pi( q_x' +  q_y')\bigr)
\end{aligned}
$$

Precomputed [3D projections](https://froeschle4d.streamlit.app/).

Doesn't work on mobile, requires several GBs of RAM to display 500k+ points in 3D plot.

# Moser map

$$
\begin{aligned}
\xi_1' &= \xi_1 + \frac{\delta r_1 - \gamma r_2}{\alpha\delta - \beta\gamma} \\
\xi_2' &= \xi_2 + \frac{-\beta r_1 + \alpha r_2}{\alpha\delta - \beta\gamma} \\
\eta_1' &= \alpha \xi_1 + \beta \xi_2 \\
\eta_2' &= \gamma \xi_1 + \delta \xi_2
\end{aligned}
$$

where

$$
\begin{aligned}
r_1 &= -\eta_1 + \alpha \xi_1 + \beta \xi_2 + \frac{\partial U}{\partial \xi_1} \\
r_2 &= -\eta_2 + \gamma \xi_1 + \delta \xi_2 + \frac{\partial U}{\partial \xi_2}
\end{aligned}
$$

and

$$
\begin{aligned}
U(\xi_1, \xi_2) = a\,\xi_1 + b\,\xi_2 + \frac{1}{2}c\,\xi_1^2 + \mu \xi_1^3 + \xi_1 \xi_2^2
\end{aligned}
$$

Precomputed [3D projections](https://moser4d.streamlit.app/).

Doesn't work on mobile, requires several GBs of RAM to display 500k+ points in 3D plot.
