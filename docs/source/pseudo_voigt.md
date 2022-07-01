# Pseudo Voigt

For pseudo voigt IRF function

\begin{equation*}
{pv}(t) = \eta \frac{\gamma}{\pi}\frac{1}{t^2+\gamma^2} + (1-\eta)\frac{1}{\sigma \sqrt{2\pi}}\exp\left(-\frac{t^2}{2\sigma}\right)
\end{equation*}, mixing parameter eta is guessed to

\begin{equation*}
\eta = 1.36603({fwhm}_L/f)-0.47719({fwhm}_L/f)^2+0.11116({fwhm}_L/f)^3
\end{equation*}

where
\begin{align*}
f &= ({fwhm}_G^5+2.69269{fwhm}_G^4{fwhm}_L+2.42843{fwhm}_G^3{fwhm}_L^2 \\
  &+ 4.47163{fwhm}_G^2{fwhm}_L^3+0.07842{fwhm}_G{fwhm}_L^4 \\
  &+ {fwhm}_L^5)^{1/5}
\end{align*}

This guess is according to [J. Appl. Cryst. (2000). **33**, 1311-1316](https://doi.org/10.1107/S0021889800010219)