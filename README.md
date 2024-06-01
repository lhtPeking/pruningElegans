<div  align="center"><img  src="https://github.com/lhtPeking/pruningElegans/blob/main/imgs/connectome.png?raw=true"  width="800"/></div>

# Elegans Inspired Pruning&Rewiring Algorithm

##  Reference

  

[Neural Circuit Policies Enabling Auditable Autonomy (Open Access)](https://publik.tuwien.ac.at/files/publik_292280.pdf).

[Closed-form continuous-time neural networks (Open Access)](https://www.nature.com/articles/s42256-022-00556-7)

  
##  Algorithm
Traditional **Neural Circuit Policies (NCPs)** are designed sparse recurrent neural networks loosely inspired by the nervous system of the organism [**C. elegans**](http://www.wormbook.org/chapters/www_celegansintro/celegansintro.html). And our work attempts to make improvements on the wiring strategy of a single neuron in the NCPs.
</n>
</n>
In the traditional **RNN** model, the hidden layer state transition could be represented as:


<div  align="center"><img  src="https://github.com/lhtPeking/pruningElegans/blob/main/imgs/Formula1.png?raw=true"  width="250"/></div>


Where $\pmb{X}$ represents hidden layer state, $\pmb{\theta}$ represents model's parameters.
</n>
</n>
We can continuous the discretized state transitions, then we have the **Neural Ordinary Differential Equation (ODE)**:

<div  align="center"><img  src="https://github.com/lhtPeking/pruningElegans/blob/main/imgs/Formula2.png?raw=true"  width="300"/></div>


Where $\pmb{I}(t)$ represents the input matrix at time point $t$.
</n>
</n>
Then introduce **time constant** $\tau$ to describe the speed at which the model converges to equilibrium, get the **Continuous-Time RNN (CT-RNN)**:
<div  align="center"><img  src="https://github.com/lhtPeking/pruningElegans/blob/main/imgs/Formula3.png?raw=true"  width="400"/></div>

And $\tau$ does not affect the equilibrium state:
<div  align="center"><img  src="https://github.com/lhtPeking/pruningElegans/blob/main/imgs/Formula4.png?raw=true"  width="250"/></div>


</n>
</n>
Finally we introduce a correction term $\pmb{A}-\pmb{X}(t)$ for the convergence speed, making the convergence speed inversely proportional to the distance between the model itself and the expected state:
<div  align="center"><img  src="https://github.com/lhtPeking/pruningElegans/blob/main/imgs/Formula5.png?raw=true"  width="500"/></div>


Simplifying the equation structure, yields:
<div  align="center"><img  src="https://github.com/lhtPeking/pruningElegans/blob/main/imgs/Formula6.png?raw=true"  width="500"/></div>



Where $\tau_{system}=\frac{\tau}{1+\tau f(\pmb{X}(t),\pmb{I}(t),t,\pmb{\theta})}$, and this model is called **Liquid Time Constant (LTC)** model.
</n>
</n>
In a NCP, every unit (every neuron) is actually a LTC model, so any neuron possesses time-series processing ability (derived from RNN), and the formula for Neuron $i$ is transformed to:

<div  align="center"><img  src="https://github.com/lhtPeking/pruningElegans/blob/main/imgs/Formula.png?raw=true"  width="800"/></div>


Where $w_{ij}$ represents connection weight between neuron $i$ and neuron $j$.  $C_{m_i}$ represents the membrane capacitance of Neuron $i$. $\sigma_i(\pmb{X}_j)$ represents the input from neuron $j$ to neuron $i$, and $\sigma_i$ is the activate function. $E_{ij}$ is the polarity of the connection ($E_{ij}\in\{1,-1\}$)
</n>
</n>
Based on the LTC single neuron, the original algorithm in [Nature Article](https://publik.tuwien.ac.at/files/publik_292280.pdf) defined a wiring system:

<div  align="center"><img  src="https://github.com/lhtPeking/pruningElegans/blob/main/imgs/OriginalWiring.png?raw=true"  width="800"/></div>

(1) Insert four neural layers: sensory neurons $N_s$, inter-neurons $N_i$, command neurons $N_c$ and motor neurons $N_m$
</n>
</n>
(2) Between every two consecutive layers: $\forall$ source neuron, insert $n_{so−t}$ synapses ($n_{so−t}\le{N_t}$), with synaptic polarity $E_{ij}$ ~ $Bernoulli(p_2)$, to $n_{so−t}$ target neurons, randomly selected ~ $Binomial(n_{so−t}, p_1)$. $n_{so−t}$ is the number of synapses from source to target. $p_1$ and $p_2$ are probabilities corresponding to their distributions.
</n>
</n>
(3) Between every two consecutive layers: $\forall$ target neuron $j$ with no synapse, insert $m_{so-t}$ synapses ($m_{so-t}\le{\frac{1}{N_t}\sum_{i\neq{j}}}L_{t_i}$, which means the newly insert number is below average), where $L_{t_i}$ is the number of synapses to target neuron $i$, with synaptic polarity $E_{ij}$ ~ $Bernoulli(p_2)$, from $m_{so-t}$ source neurons, randomly selected from ~ $Binomial(m_{so−t}, p_3)$. $m_{so−t}$ is the number of synapses from source to target neurons with no synaptic connections.
</n>
</n>
(4) Recurrent connections of command neurons: $\forall$ command neuron, insert $l_{so−t}$ synapses ($l_{so−t}\le{N_t}$), with synaptic polarity $E_{ij}$ ~ $Bernoulli(p_2)$, to $l_{so−t}$ target command neurons, randomly selected from ~ $Binomial(l_{so−t}, p_4)$ . $l_{so−t}$ is the number of synapses from one interneuron to target neurons.
</n>
</n>
From the original wiring system we can see that: once the wiring strategy is set up at the very beginning, it won't change during the training process. But it is obvious that the wiring structure won't be optimal through such a stochastic wiring process. This drawback was ignored in the original algorithm.
</n>
</n>
Inspired by biological neural connection pruning system, we developed a new algorithm named "**Pruning&Rewiring**", when the connection weight $w_{ij}$ between any two LTC neurons is lower than a threshold $T$, we cut off this connection ( set $w_{ij}$ to be $0$ ), and meanwhile insert a synapse from the source neuron to its next layer's neuron (expect the target neuron just pruned) to prevent the model from degeneration.

##  Mathematical Justification
We demonstrate the superiority of the mathematical properties of this new algorithm from two aspects：**Stability** and **Astringency**.
</n>
</n>

Given the NCP connection subgraph as shown in the figure, in order to simplify the problem for analysis, only one pruning and its corresponding rewiring process are provided. The state of neurons $j$ and $k$ is directly represented as output (the output matrix $\pmb{X}$ is a 2 × 1 matrix). The connection weights between neurons are as shown in the figure. ($W_{ij}\lt{T}$ and $W_{ik}\gt\gt{W_{ij}}$ )


<div  align="center"><img  src="https://github.com/lhtPeking/pruningElegans/blob/main/imgs/singlePruningAndRewiring.jpg?raw=true"  width="800"/></div>


###  Stability
By analyzing the derivative of Lyapunov Function defined as: $V(\pmb{X})=\pmb{X}^T\pmb{P}\pmb{X}$, we found that the more significant the state values of the newly connected neurons, the more stable the newly connected system becomes (The detailed justification could be seen in the pdf file: **Proof1：The Analysis of Lyapunov Function Stability**).

###  Astringency
By analyzing the Jacobian Matrix of the **Pre-Pruning&Rewiring state** and the **Post-Pruning&Rewiring state**, we found that the convergence speed of the second state will consistently be faster than that of the first state (The detailed justification could be seen in the pdf file: **Proof2：The Analysis of Jacobian matrix Astringency**).

##  Tasks and Reproduction