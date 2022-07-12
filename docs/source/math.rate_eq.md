# Rate Equation

In pump prove time resolved spectroscopy, we assume reaction occurs just after pump pulse. So, for 1st order dynamics, what we should to solve is

\begin{equation*}
\mathbf{y}'(t) = \begin{cases}
0& \text{if $t < 0$}, \\
A\mathbf{y}(t)& \text{if $t>0$}.
\end{cases}
\end{equation*}

with, $\mathbf{y}(0)=\mathbf{y}_0$. The solution of above first order equation is given by

\begin{equation*}
\mathbf{y}(t) = \begin{cases} 
\exp\left(At\right) \mathbf{y}_0 & \text{if $t\geq 0$} \\
\mathbf{0}
\end{cases}
\end{equation*}
, where $\exp\left(At\right)$ is the matrix exponential defined as 

\begin{equation*}
\exp\left(At\right) = 1 + tA + \frac{1}{2!} t^2 A^2 + \frac{1}{3!} t^3 A^3 + \dotsc
\end{equation*}

$\mathbf{0}$ means, at $t<0$ (i.e. before laser irrediation), there is no excited state species. 

Suppose that the rate equation matrix $A$ is diagonalizable. In general, it cannot be diagonalizable.
Then we can write 
\begin{equation*}
A = V \Lambda V^{-1}
\end{equation*}
, where $V$ is eigen matrix of $A$ and $\Lambda = \mathrm{diag}\left(\lambda_1,\dotsc,\lambda_n\right)$.

Then,

\begin{align*}
\mathbf{y}(t) &= \exp\left(At\right) \mathbf{y}_0 \\
&= V \exp\left(\Lambda t \right) V^{-1} \mathbf{y}_0 \\
&= V \mathrm{diag}\left(\exp\left(\lambda_1 t\right),\dotsc,\exp\left(\lambda_n t \right)\right) V^{-1} \mathbf{y}_0
\end{align*}

Define $\mathbf{c}$ as $V\mathbf{c} = \mathbf{y}_0$ then

\begin{equation*}
\mathbf{y}(t) = \sum_i c_i \exp\left(\lambda_i t\right) \mathbf{v}_i
\end{equation*}

To model experimentally observed population, we need to convolve our model population $\mathbf{y}(t)$ to instrumental response function $\mathrm{IRF}$.
Then we can model observed population $\mathbf{y}_{obs}(t)$ as

\begin{equation*}
\mathbf{y}_{obs}(t) = \sum_i c_i (\exp *_h {IRF})(\lambda_i t) \mathbf{v}_i
\end{equation*}

, where $*_h$ is the half convolution operator defined as

\begin{equation*}
(f *_h g)(t) = \int_{0}^{\infty} f(x)g(t-x) \mathrm{d} x
\end{equation*}

Experimental time delay signal is the linear combination of $(y_{obs})_i(t)$. Thus, experimental time delay signal is also represented by the sum of $\{(\exp*_h {IRF})(\lambda_i t)\}$.
