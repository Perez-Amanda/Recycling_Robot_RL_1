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

Quando a bateria está baixa, todas as ações são possíveis. Quando a bateria está alta, não é possível recarregar, apenas buscar ou esperar. Nota-se que esse é um processo de decisão de Markov finito e com poucas combinações de espaço-ação, o que possibilita o uso de métodos tabulares. Seguindo o solicitado pelo enunciado, [falar sobre usar TD e a escolha de usar Q-learning]


## Implementação

O código implementado encontra-se disponível no arquivo `main.ipynb`. Acompanhando o código, estão alguns comentários para complementar e melhorar a legibilidade. 

Buscando seguir uma estrutura parecida com a do exemplo do jogo da velha, foram definidas as classes:
- `Enviroment`: ...
- `Agent`: ...



## Resultados e discussão



---
## Referências

[1] SUTTON, Richard S.; BARTO, Andrew G. **Reinforcement Learning: An introduction.** 2. ed. Cambridge, MA, USA: The MIT Press, 2018.

[2]