# Signal, Cauchy IRF

When instrument response function is modeled to normalized cauchy distribution, experimental signal is modeled to exponentical decay convolved with normalized cauchy distribution.

\begin{align*}
{Signal}_c(t) &= ({model} * {irf})(t) \\
&= \frac{1}{\pi} \int_0^{\infty} \frac{\gamma \exp(-kx)}{(x-t)^2+\gamma^2} \mathrm{d}x \\
&= \frac{1}{\pi} \Im\left(\int_0^{\infty} \frac{\exp(-kx)}{(x-t)-i\gamma} \mathrm{d}x \right)
\end{align*}

Assume $k > 0$, and let $u=k(x-t)-ik\gamma$, then
\begin{align*}
{Signal}_c(t) &= \frac{1}{\pi} \exp(-kt) \Im\left(\exp(-ik\gamma) \int_{-k(t+i\gamma)}^{\infty-ik\gamma} \frac{\exp(-u)}{u} \mathrm{d}u \right) \\
&= \frac{1}{\pi} \exp(-kt) \Im(\exp(-ik\gamma){E1}(-k(t+i\gamma))
\end{align*}

So, experimental signal is modeled to

\begin{equation*}
{Signal_c}(t) = \begin{cases}
\frac{1}{2} + \frac{1}{\pi}\arctan\left(\frac{t}{\gamma}\right)& \text{if $k=0$}, \\
\frac{\exp(-kt)}{\pi} \Im(\exp(-ik\gamma){E1}(-k(t+i\gamma)))& \text{if $k>0$}.
\end{cases}
\end{equation*}

