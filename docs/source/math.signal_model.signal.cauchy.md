# Signal, Cauchy IRF

When instrument response function is modeled as normalized cauchy distribution, experimental signal is modeled as convolution of exponentical decay and normalized cauchy distribution.

\begin{align*}
{Signal}_c(t) &= ({model} * {irf})(t) \\
&= \frac{1}{\pi} \int_0^{\infty} \frac{\gamma \exp(-kx)}{(x-t)^2+\gamma^2} \mathrm{d}x \\
&= \frac{1}{\pi} \Im\left(\int_0^{\infty} \frac{\exp(-kx)}{(x-t)-i\gamma} \mathrm{d}x \right)
\end{align*}

Assume $k > 0$, and let $u=k(x-t)-ik\gamma$, then
\begin{align*}
{Signal}_c(t) &= \frac{1}{\pi} \exp(-kt) \Im\left(\exp(-ik\gamma) \int_{-k(t+i\gamma)}^{\infty-ik\gamma} \frac{\exp(-u)}{u} \mathrm{d}u \right) \\
&= \frac{1}{\pi} \exp(-kt) \Im(\exp(-ik\gamma)E_1(-k(t+i\gamma))
\end{align*}

So, experimental signal could be modeled as

\begin{equation*}
{Signal_c}(t) = \begin{cases}
\frac{1}{2} + \frac{1}{\pi}\arctan\left(\frac{t}{\gamma}\right)& \text{if $k=0$}, \\
\frac{\exp(-kt)}{\pi} \Im(\exp(-ik\gamma)E_1(-k(t+i\gamma)))& \text{if $k>0$}.
\end{cases}
\end{equation*}

$E_1(z)$ is exponential integral, see [dlmf section 6.2](https://dlmf.nist.gov/6.2). 

## Implementation Note
At $|kt| > 700$, $E_1$ or $\exp$ term overflows, so, in this region,
the following asymptotic expression is used.

\begin{equation*}
{Signal_c}(t) = -\frac{1}{\pi}\Im\left(\frac{1}{kt+i\gamma}\sum_{i=0}^{10} \frac{i!}{(kt+i\gamma)^i}\right)
\end{equation*}

