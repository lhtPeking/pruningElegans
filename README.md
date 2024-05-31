<div  align="center"><img  src="https://github.com/lhtPeking/pruningElegans/blob/main/imgs/connectome.png?raw=true"  width="800"/></div>

# Elegans Inspired Pruning&Rewiring Algorithm

## 📜 Reference

  

[Neural Circuit Policies Enabling Auditable Autonomy (Open Access)](https://publik.tuwien.ac.at/files/publik_292280.pdf).

[Closed-form continuous-time neural networks (Open Access)](https://www.nature.com/articles/s42256-022-00556-7)

  

Traditional **Neural Circuit Policies (NCPs)** are designed sparse recurrent neural networks loosely inspired by the nervous system of the organism [**C. elegans**](http://www.wormbook.org/chapters/www_celegansintro/celegansintro.html). 

In the traditional **RNN** model, the hidden layer state transition could be represented as:

\begin{center}
$\pmb{X}_{t+1}=\pmb{X}_t+f(\pmb{X}_t, \pmb{\theta})$
\end{center}

Where $\pmb{X}$ represents hidden layer state, $\pmb{\theta}$ represents model's parameters.

We can continuous the discretized state transitions, then we have the **Neural Ordinary Differential Equation (ODE)**:
$$
\frac{d\pmb{X}(t)}{dt}=f(\pmb{X}(t),\pmb{I}(t),t,\pmb{\theta})
$$
Where $\pmb{I}(t)$ represents the input matrix at time point $t$.