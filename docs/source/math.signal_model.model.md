# Model

In pump-probe time resolved spectroscopy, we usually model the dynamics as sum of the exponential decay. For simplicity, I will consider one exponential decay model.

\begin{equation*}
{Model}(t) = \begin{cases}
0& \text{if $t<0$}, \\
\exp(-kt)& \text{if $t \geq 0$}.
\end{cases}
\end{equation*}

where $k$ is rate constant, inverse of the life time.
