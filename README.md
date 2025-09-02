# Robô de Reciclagem - _Reinforcement Learning_ - Projeto 1
## Amanda Perez & Lucas Westfal

Repositório criado para implementar um algoritmo de aprendizado por reforço para um robô de reciclagem, como projeto 1 do curso de [Aprendizado por Reforço](https://github.com/fccoelho/Reinforcement-Learning-course/) da FGV EMAp. O enunciado do trabalho encontra-se disponível no arquivo `enunciado.md`. 

A seguir, neste mesmo documento, apresentamos o relatório do projeto, com a descrição do problema e os resultados observados.

> *Ambos os participantes dessa tarefa trabalharam equivalentemente no desenvolvimento do projeto, majoritariamente por meio de programação em par.*

-----

# *Relatório*

## Descrição do problema

Consideramos um robô de reciclagem, conforme apresentado no Exemplo 3.3 (página  52) de [1]. O robô que deve coletar latas para reciclagem. Seu funcionamento depende de uma bateria, que pode estar alta ou baixa, e o robô pode escolher entre andar pelo ambiente em busca de latas, ficar parado esperando que alguém o entregue alguma lata ou ir recarregar. Dessa forma, o espaço de estados $\mathcal{X}$ descreve os níveis da bateria e é dado por:
$$
\mathcal{X} = \{ \textrm{alta}, \textrm{baixa} \},
$$
enquanto o espaço de ações é:
$$
\mathcal{A} = \{ \textrm{buscar}, \textrm{esperar}, \textrm{recarregar} \}
$$

Quando a bateria está baixa, todas as ações são possíveis. Quando a bateria está alta, não é possível recarregar, apenas buscar ou esperar. Nota-se que esse é um processo de decisão de Markov finito e com poucas combinações de espaço-ação, o que possibilita o uso de métodos tabulares.

## Soluções

Usando *temporal difference learning*, o problema pode ser resolvido equivalentemente por diferentes métodos capazes de estimar a política ótima; usamos TD(0) e Q-Learning.

### TD(0)

O TD(0) por si só apenas é um algoritmo de avaliação. Ele nos dá o valor de estar em um estado $s$ sob uma política específica $\pi$, $V(s)$.

Atualizamos TD(0) por:

$$ V(s) \leftarrow V(s) + \alpha \left[ R + \gamma V(s') - V(s) \right] $$

Como a atualização depende do valor do próximo estado, $V(s')$, que foi obtido por seguir a política $\pi$, o método não procura politica ótima mas sim avalia a qualidade da política atual - se fazendo necessário método adicional para evoluir as políticas. 

O código para aprender *tic-tac-toe*, disponibilizado para resolver esse problema, adota essa metodologia.

### Q-Learning

Em contraste, o *Q-learning* aprende diretamente o valor de tomar ação $a$ em um estado $s$, $Q(s, a)$. Com ele podemos encontrar política ótima $\pi^*$ durante jogo, dado o conhecimento acumulado do jogo.

A atualização da função $Q$ é feita pela seguinte equação:

$$ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] $$

Onde com $\max_{a'} Q(s', a')$ encontramos ação $a'$ que maximiza a função $Q$ depois de tomada ação $a$. É atualizado o valor da ação atual $(s, a)$ assumindo que a melhor ação possível será tomada no próximo estado $(s')$, independentemente da ação que a política de exploração realmente escolheu. Isso o torna um método off-policy, pois aprende sobre a política ótima (gananciosa) enquanto segue outra política (por exemplo, epsilon-greedy)

## Implementação

O código implementado encontra-se disponível no arquivo `main.ipynb`. Acompanhando o código, estão alguns comentários para complementar e melhorar a legibilidade. 

Buscando seguir uma estrutura parecida com a do exemplo do jogo da velha, foram definidas as classes:
- `Enviroment`: Lorem ipsum
- `Agent`: Lorem ipsum

### TD(0)

Loren ipsum

### Q-Learning

Lorem ipsum



## Resultados e discussão

Para explorar a influência dos hiperparâmetros no aprendizado de cada método, fixamos valores das recompensas e os parâmetros do ambiente ($\alpha$ e $\beta$) e rodamos os jogos com diferentes valores de $\epsilon$ e $\gamma$. Também comparamos o efeito de tomar ou não políticas exploratórias durante o treinamento.



### Explorando $\epsilon$

- TD(0)

Primeiramente, utilizando o método TD(0), fixamos $\gamma = 0.9$ e tomamos $\varepsilon \in \{0.0, \ 0.1, \ 0.5\}$.

Em todos os cenários, a política ótima aprendida ao fim do treinamento opta por procurar sempre que a bateria está alta e recarregar sempre que a bateria está baixa, não havendo grande diferença entre os *heatmaps*. Entretanto, a evolução dos *rewards* ao longo das épocas apresenta diferenças. 

<p float="left">
  <img src="/figs/rewards_epsilon0p1_td0.png" width="400" />
  <img src="/figs/heatmap_epsilon0p1_td0.png" width="400" />
</p>


<p float="left">
  <img src="/figs/rewards_epsilon0p5_td0.png" width="400" />
  <img src="/figs/heatmap_epsilon0p5_td0.png" width="400" />
</p>

<p float="left">
  <img src="/figs/rewards_greddy_td0.png" width="400" />
  <img src="/figs/heatmap_greedy_td0.png" width="400" />
</p>

<!-- ![FIGURA KAPUTT](figs/rewards_epsilon0p1_td0.png)

![FIGURA KAPUTT](figs/heatmap_epsilon0p1_td0.png) -->

<!-- ![FIGURA KAPUTT](figs/rewards_epsilon0p5_td0.png)

![FIGURA KAPUTT](figs/heatmap_epsilon0p5_td0.png)

![FIGURA KAPUT](figs/rewards_greddy_td0.png)

![FIGURA KAPUT](figs/heatmap_greedy_td0.png) -->


- Q-Learning

<p float="left">
  <img src="/figs/rewards_epsilon0p1_q.png" width="400" />
  <img src="/figs/heatmap_epsilon0p1_q.png" width="400" />
</p>

<p float="left">
  <img src="/figs/rewards_epsilon0p5_q.png" width="400" />
  <img src="/figs/heatmap_epsilon0p5_q.png" width="400" />
</p>

<p float="left">
  <img src="/figs/rewards_greedy_q.png" width="400" />
  <img src="/figs/heatmap_greedy_q.png" width="400" />
</p>

<!-- ![FIGURA KAPUTT](figs/rewards_epsilon0p1_q.png)

![FIGURA KAPUTT](figs/heatmap_epsilon0p1_q.png)

![FIGURA KAPUTT](figs/rewards_epsilon0p5_q.png)

![FIGURA KAPUTT](figs/heatmap_epsilon0p5_q.png)

![FIGURA KAPUT](figs/rewards_greedy_q.png)

![FIGURA KAPUT](figs/heatmap_greedy_q.png) -->


<!-- ![FIGURA KAPUTT](figs/rewards_greedy_q.png)

![FIGURA KAPUTT](figs/heatmap_greedy_q.png) -->

<!-- 
![FIGURA KAPUTT](figs/rewards_greedy_q.png)

![FIGURA KAPUTT](figs/heatmap_greedy_q.png) -->

### Explorando $\gamma$

- TD(0)

<p float="left">
  <img src="/figs/rewards_gamma0p5_td0.png" width="400" />
  <img src="/figs/heatmap_gamma0p5_td0.png" width="400" />
</p>

<p float="left">
  <img src="/figs/rewards_gamma0p8_td0.png" width="400" />
  <img src="/figs/heatmap_gamma0p8_td0.png" width="400" />
</p>

<p float="left">
  <img src="/figs/rewards_gamma0p9_td0.png" width="400" />
  <img src="/figs/heatmap_gamma0p9_td0.png" width="400" />
</p>


<!-- ![FIGURA KAPUTT](figs/rewards_gamma0p5_td0.png)

![FIGURA KAPUTT](figs/heatmap_gamma0p5_td0.png)

![FIGURA KAPUTT](figs/rewards_gamma0p8_td0.png)

![FIGURA KAPUTT](figs/heatmap_gamma0p8_td0.png)

![FIGURA KAPUTT](figs/rewards_gamma0p9_td0.png)

![FIGURA KAPUTT](figs/heatmap_gamma0p9_td0.png) -->

- Q-Learning

<p float="left">
  <img src="/figs/rewards_gamma0p5_q.png" width="400" />
  <img src="/figs/heatmap_gamma0p5_q.png" width="400" />
</p>

<p float="left">
  <img src="/figs/rewards_gamma0p8_q.png" width="400" />
  <img src="/figs/heatmap_gamma0p8_q.png" width="400" />
</p>

<p float="left">
  <img src="/figs/rewards_gamma0p9_q.png" width="400" />
  <img src="/figs/heatmap_gamma0p9_q.png" width="400" />
</p>

<!-- ![FIGURA KAPUTT](figs/rewards_gamma0p5_q.png)

![FIGURA KAPUTT](figs/heatmap_gamma0p5_q.png)

![FIGURA KAPUTT](figs/rewards_gamma0p8_q.png)

![FIGURA KAPUTT](figs/heatmap_gamma0p8_q.png)

![FIGURA KAPUTT](figs/rewards_gamma0p9_q.png)

![FIGURA KAPUTT](figs/heatmap_gamma0p9_q.png) -->

<!-- ### Comparando greedy e $\epsilon$-greedy

![FIGURA KAPUTT](figs/rewards_epsilon_greedy_td0.png)

![FIGURA KAPUTT](figs/heatmap_epsilon_greedy_td0.png)

![FIGURA KAPUTT](figs/heatmap_greedy_td0.png)

- Q-Learning

![FIGURA KAPUTT](figs/rewards_epsilon_greedy_q.png)

![FIGURA KAPUTT](figs/heatmap_epsilon_greedy_q.png)

![FIGURA KAPUTT](figs/rewards_greedy_q.png)

![FIGURA KAPUTT](figs/heatmap_greedy_q.png) -->

---
## Referências

[1] SUTTON, Richard S.; BARTO, Andrew G. **Reinforcement Learning: An introduction.** 2. ed. Cambridge, MA, USA: The MIT Press, 2018.

[2] CSABA SZEPESVARI. Algorithms for Reinforcement Learning. [s.l.] Morgan & Claypool Publishers, 2010.

‌