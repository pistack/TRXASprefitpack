# Rate Equation

In pump prove time resolved spectroscopy, we assume reaction occurs just after pump pulse. So, for 1st order dynamics, what we should to solve is

\begin{equation*}
y'(t) = \begin{cases}
0& \text{if $t < 0$}, \\
Ay(t)& \text{if $t>0$}.
\end{cases}
\end{equation*}

with, $y(0)=y_0$.

Ususally, y is modeled as sum of the exponential decays, so we can assume the matrix A could be diagonalizable.

Then, 
\begin{equation*}
y(t) = \begin{cases}
0& \text{if $t < 0$}, \\
\sum_i c_i \exp(\lambda_i t) v_i& \text{if $ t \geq 0$}
\end{cases}
\end{equation*}

Where $\lambda_i$ is $i$th eigenvalue of the matrix $A$, $v_i$ is the eigenvector corresponding to $\lambda_i$ and coefficient $c_i$ are chosen to satisfy $y(0)=y_0$.

To model experimental signal corresponding to the dynamics, we convolve our model $y(t)$ to IRF, then we can model experimental signal as

\begin{equation*}
signal(t) = \sum_i c_i (\exp * {irf})(\lambda_i t) v_i
\end{equation*}
