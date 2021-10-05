# Signal, Gaussian IRF

When instrument response function is modeled as normalized gaussian distribution, experimental signal is modeled as convolution of exponentical decay and normalized gaussian distribution.

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
&= \frac{1}{2}\exp\left(\frac{(k\sigma)^2}{2}-kt\right)\mathrm{erfc}\left(\frac{1}{\sqrt{2}}\left(k\sigma - \frac{t}{\sigma}\right)\right)
\end{align*}

So, experimental signal could be modeled as

\begin{equation*}
{Signal}_g(t) = \frac{1}{2}\exp\left(\frac{(k\sigma)^2}{2}-kt\right)\mathrm{erfc}\left(\frac{1}{\sqrt{2}}\left(k\sigma - \frac{t}{\sigma}\right)\right)
\end{equation*}

$\mathrm{erfc}(x)$ is complementary error function, see [dlmf section 7.2](https://dlmf.nist.gov/7.2).

This is also equivalent to

\begin{equation*}
{Signal}_g(t) = \frac{1}{2}\exp\left(-\frac{t^2}{2\sigma^2}\right)\mathrm{erfcx}\left(\frac{1}{\sqrt{2}}\left(k\sigma - \frac{t}{\sigma}\right)\right)
\end{equation*}

$\mathrm{erfcx}(x)$ is scaled complementary error function, see [dlmf section 7.2](https://dlmf.nist.gov/7.2).

## Implementation Note
When $x>0$, $\mathrm{erfcx}(x)$ deverges and when $x<0$, $\exp(-x)$ deverges.
To tame such divergency, I use following implementation.

\begin{equation*}
{Signal}_g(t) = \begin{cases}
\frac{1}{2}\exp\left(-t^2/(2\sigma^2)\right)\mathrm{erfcx}\left(\frac{1}{\sqrt{2}}\left(k\sigma - t/\sigma\right)\right) & \text{if $t<k\sigma^2$}, \\
\frac{1}{2}\exp\left((k\sigma)^2/2-kt\right)\mathrm{erfc}\left(\frac{1}{\sqrt{2}}\left(k\sigma - t/\sigma\right)\right) & \text{otherwise}.
\end{cases}
\end{equation*}





