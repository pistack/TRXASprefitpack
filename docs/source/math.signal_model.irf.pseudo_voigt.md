# Pseudo Voigt

Someone models instrument response function as voigt profile. Since convolution with voigt profile is hard, I approximate voigt profile to linear combination of cauchy and gaussian distribution. Such approximated function is usually called pseudo voigt profile.

\begin{equation*}
{IRF}(t) = \eta C({fwhm}_L, t) + (1-\eta)G({fwhm}_G, t)
\end{equation*}

where, ${fwhm}_L$, ${fwhm}_G$ is full width at half maximum value of cauchy and gaussian distribution, respectively. $C(t)$, $G(t)$ is normalized cauchy and gaussian distribution, each. $\eta$ is mixing parameter ($0 \leq \eta \leq 1$).
