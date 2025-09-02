# Robô de Reciclagem - _Reinforcement Learning_ - Projeto 1
## Amanda Perez & Lucas Westfal

Repositório criado para implementar um algoritmo de aprendizado por reforço para um robô de reciclagem, como projeto 1 do curso de [Aprendizado por Reforço](https://github.com/fccoelho/Reinforcement-Learning-course/) da FGV EMAp. O enunciado do trabalho encontra-se disponível no arquivo `enunciado.md`. 

A seguir, neste mesmo documento, apresentamos o relatório do projeto, com a descrição do problema e os resultados observados.

> *Ambos os participantes dessa tarefa trabalharam equivalentemente no desenvolvimento do projeto, majoritariamente por meio de programação em par.*

-----

# *Relatório*

## Introdução

Este trabalho apresenta a implementação e análise de algoritmos de aprendizado por reforço para solucionar o problema do robô de reciclagem, conforme descrito por Sutton & Barto. Foram implementados dois métodos de aprendizado por diferença temporal (TD): TD(0) para avaliação de política e Q-Learning para a busca direta da política ótima. O estudo explora a influência de hiperparâmetros, como a taxa de exploração ($\epsilon$) e o *discount factor* ($\gamma$), no comportamento e na convergência do agente. Os resultados demonstram a importância da exploração (política $\epsilon$-greedy) para a descoberta da política ótima em comparação com uma abordagem puramente greedy, além de evidenciarem como o fator de desconto molda a estratégia do agente entre recompensas imediatas e ganhos a longo prazo, por meio de curvas de aprendizado (recompensa por época) e as políticas ótimas resultantes por meio de mapas de calor.

## Descrição do problema

Consideramos um robô de reciclagem, conforme apresentado no Exemplo 3.3 (página  52) de [1]. O robô que deve coletar latas para reciclagem. Seu funcionamento depende de uma bateria, que pode estar alta ou baixa, e o robô pode escolher entre andar pelo ambiente em busca de latas, ficar parado esperando que alguém o entregue alguma lata ou ir recarregar. Dessa forma, o espaço de estados $\mathcal{X}$ descreve os níveis da bateria e é dado por:

$$\mathcal{X} = \{ \textrm{alta}, \textrm{baixa} \},$$
enquanto o espaço de ações é:
$$\mathcal{A} = \{ \textrm{procurar}, \textrm{esperar}, \textrm{recarregar} \}$$

Quando a bateria está baixa, todas as ações são possíveis. Quando a bateria está alta, não é possível recarregar, apenas buscar ou esperar. Nota-se que esse é um processo de decisão de Markov finito e com poucas combinações de espaço-ação, o que possibilita o uso de métodos tabulares.

## Soluções

Usando *temporal difference learning*, o problema pode ser resolvido equivalentemente por diferentes métodos capazes de estimar a política ótima; usamos TD(0) (mais método de atualização de política) e Q-Learning.

### TD(0)

O TD(0) por si só apenas é um algoritmo de avaliação. Ele nos dá o valor de estar em um estado $s$ sob uma política específica $\pi$, $V(s)$.

Atualizamos TD(0) por:

$$ V(s) \leftarrow V(s) + \alpha \left[ R + \gamma V(s') - V(s) \right] $$

Como a atualização depende do valor do próximo estado, $V(s')$, que foi obtido por seguir a política $\pi$, o método não procura politica ótima mas sim avalia a qualidade da política atual - se fazendo necessário método adicional para evoluir as políticas. 

O código para aprender *tic-tac-toe*, adaptado para resolver esse problema, adota essa metodologia.

### Q-Learning

Em contraste, o *Q-learning* aprende diretamente o valor de tomar ação $a$ em um estado $s$, $Q(s, a)$. Com ele podemos encontrar política ótima $\pi^*$ durante jogo, dado o conhecimento acumulado do jogo.

A atualização da função $Q$ é feita pela seguinte equação:

$$ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] $$

Onde com $\max_{a'} Q(s', a')$ encontramos ação $a'$ que maximiza a função $Q$ depois de tomada ação $a$. É atualizado o valor da ação atual $(s, a)$ assumindo que a melhor ação possível será tomada no próximo estado $(s')$, independentemente da ação que a política de exploração realmente escolheu. Isso o torna um método off-policy, pois aprende sobre a política ótima (greedy) enquanto segue outra política (por exemplo, $\epsilon$-greedy)

## Implementação

O código implementado encontra-se disponível no arquivo `main.ipynb`. Acompanhando o código, estão alguns comentários para complementar e melhorar a legibilidade. 

Fora criado uma classe para simular o ambiente e duas classes para simular jogador, uma destinada ao TD(0) e outra ao Q-Learning.

Buscando seguir uma estrutura parecida com a do exemplo do jogo da velha, foram definidas as classes (com apenas as descrições abaixo):

- `Enviroment`: 

```
class Environment:
    """
    Simula o ambiente do robô de reciclagem.
    Gerencia as transições de estado e as recompensas com base nas ações do robô.
    """
    def get_available_actions(self, state):
        """ Retorna as ações possíveis para um dado estado. """
        ...

    def step(self, state, action):
        """
        Executa uma ação e retorna o próximo estado e a recompensa.
        A lógica de transição é baseada nas probabilidades alfa e beta.
        """
        ...

```

- `Agent` (caso TD(0)):
Conta com um passo a mais com relação ao Q-Learning, que atualiza a policy após avaliar a nova função de $V(s)$.

```
class AgentTD:
    """
    Agente que usa TD(0) para avaliação e melhora a política de forma explícita.
    """
    def __init__(self, discount_factor, epsilon, learning_rate=LEARNING_RATE,):
        ...
    def choose_action(self, state):
        """
        Escolhe uma ação baseada na política atual (com exploração epsilon-greedy).
        """
        ...

    def update(self, state, reward, next_state):
        """
        PASSO DE AVALIAÇÃO: Atualiza o valor do estado V(s) usando a regra do TD(0).
        V(s) <- V(s) + lr * [R + gamma * V(s') - V(s)]
        """
        ...

    def improve_policy(self):
        """
        PASSO DE MELHORA: Atualiza a política para ser greedy em relação aos valores de estado atuais.
        Para cada estado, escolhe a ação que maximiza a recompensa esperada.
        """
        ...

    def save_policy(self, filename="policy_td0.pkl"):
        ...

```


- `Agent` (caso Q-Learning):

```
class AgentQ:
    """
    O agente que aprende a política ótima usando Q-learning (um método TD).
    """
    def __init__(self, discount_factor, epsilon, learning_rate=LEARNING_RATE):
        ...

    def get_q_value(self, state, action):
        """ Acessa o valor Q para um par (estado, ação), retornando 0 se não existir. """
        ...

    def choose_action(self, state):
        """
        Escolhe uma ação usando uma política epsilon-greedy.
        - Com probabilidade (1 - epsilon), escolhe a melhor ação (explotação).
        - Com probabilidade epsilon, escolhe uma ação aleatória (exploração).
        """
        ...

    def update(self, state, action, reward, next_state):
        """
        Atualiza o valor Q para o par (estado, ação) usando a regra do Q-learning.
        Q(s, a) <- Q(s, a) + lr * [R + gamma * max_a'(Q(s', a')) - Q(s, a)]
        """
        ...

    def save_policy(self, filename="policy.pkl"):
        """ Salva o dicionário de valores Q em um arquivo. """
        ...

    def load_policy(self, filename="policy.pkl"):
        """ Carrega o dicionário de valores Q de um arquivo. """
        ...
```


..

Em sendo um jogo infinito, foram simulados 1000 iterações do jogo por 100 épocas, para gerar os rewards e heatmaps como solicitado.

Funções auxiliares (para gerar imagens e treinar os modelos) também ser encontrados em `main.ipynb`. 

## Resultados e discussão

Para explorar a influência dos hiperparâmetros no aprendizado de cada método, fixamos valores das recompensas e rodamos os jogos com diferentes valores de $\epsilon$ e $\gamma$. Também comparamos o efeito de tomar ou não políticas exploratórias durante o treinamento.



### Explorando $\epsilon$

- TD(0)

![FIGURA KAPUTT](figs/rewards_epsilon0p1_td0.png)

![FIGURA KAPUTT](figs/heatmap_epsilon0p1_td0.png)

![FIGURA KAPUTT](figs/rewards_epsilon0.5_td0.png)

![FIGURA KAPUTT](figs/heatmap_epsilon0.5_td0.png)


- Q-Learning

![FIGURA KAPUTT](figs/rewards_epsilon_greedy_q.png)

![FIGURA KAPUTT](figs/heatmap_epsilon_greedy_q.png)


![FIGURA KAPUTT](figs/rewards_greedy_q.png)

![FIGURA KAPUTT](figs/heatmap_greedy_q.png)

### Explorando $\gamma$

- TD(0)

![FIGURA KAPUTT](figs/rewards_gamma0p5_td0.png)

![FIGURA KAPUTT](figs/heatmap_gamma0p5_td0.png)

![FIGURA KAPUTT](figs/rewards_gamma0p9_td0.png)

![FIGURA KAPUTT](figs/heatmap_gamma0p9_td0.png)

- Q-Learning

![FIGURA KAPUTT](figs/rewards_gamma0p5_q.png)

![FIGURA KAPUTT](figs/heatmap_gamma0p5_q.png)


![FIGURA KAPUTT](figs/rewards_gamma0p9_q.png)

![FIGURA KAPUTT](figs/heatmap_gamma0p9_q.png)

### Comparando greedy e $\epsilon$-greedy

![FIGURA KAPUTT](figs/rewards_epsilon_greedy_td0.png)

![FIGURA KAPUTT](figs/heatmap_epsilon_greedy_td0.png)

![FIGURA KAPUTT](figs/heatmap_greedy_td0.png)

- Q-Learning

![FIGURA KAPUTT](figs/rewards_epsilon_greedy_q.png)

![FIGURA KAPUTT](figs/heatmap_epsilon_greedy_q.png)

![FIGURA KAPUTT](figs/rewards_greedy_q.png)

![FIGURA KAPUTT](figs/heatmap_greedy_q.png)

## Conclusão

[Resumir achados dos plots acima]

---
## Referências

[1] SUTTON, Richard S.; BARTO, Andrew G. **Reinforcement Learning: An introduction.** 2. ed. Cambridge, MA, USA: The MIT Press, 2018.

[2] CSABA SZEPESVARI. Algorithms for Reinforcement Learning. [s.l.] Morgan & Claypool Publishers, 2010.

‌