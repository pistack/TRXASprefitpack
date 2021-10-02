# Signal, Gaussian IRF

When instrument response function is modeled to normalized gaussian distribution, experimental signal is modeled to exponentical decay convolved with normalized gaussian distribution.

\begin{align*}
{Signal}_g(t) &= ({model} * {irf})(t) \\
&= \frac{1}{\sigma \sqrt{2\pi}} \int_0^{\infty} \exp(-kx)\exp\left(-\frac{(x-t)^2}{2\sigma^2}\right) \mathrm{d}x 
\end{align*}
Let $u=(x-t)/(\sigma\sqrt{2})$ then
\begin{align*}
{Signal}_g(t) &= \frac{\exp(-kt)}{\sqrt{\pi}} \int_{-t/(\sigma\sqrt{2})}^{\infty} \exp(-u^2-k\sigma\sqrt{2}u) \mathrm{d} u \\
&= \frac{\exp((k\sigma)^2/2-kt)}{\sqrt{\pi}} \int_{-t/(\sigma\sqrt{2})}^{\infty} \exp\left(-\left(u+\frac{k\sigma}{\sqrt{2}}\right)^2\right) \mathrm{d} u
\end{align*}
Let $v=u+(k\sigma)/\sqrt{2}$ then
\begin{align*}
{Signal}_g(t) &= \frac{\exp((k\sigma)^2/2-kt)}{\sqrt{\pi}} \int_{(k\sigma)/\sqrt{2}-t/(\sigma\sqrt{2})}^{\infty} \exp(-v^2) \mathrm{d} v \\
&= \frac{1}{2}\exp\left(\frac{(k\sigma)^2}{2}-kt\right){erfc}\left(\frac{1}{\sqrt{2}}\left(k\sigma - \frac{t}{\sigma}\right)\right)
\end{align*}

So, experimental signal is modeled to

\begin{equation*}
{Signal}_g(t) = \frac{1}{2}\exp\left(\frac{(k\sigma)^2}{2}-kt\right){erfc}\left(\frac{1}{\sqrt{2}}\left(k\sigma - \frac{t}{\sigma}\right)\right)
\end{equation*}
