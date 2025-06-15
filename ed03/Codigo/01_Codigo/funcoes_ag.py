import random
import numpy as np

# --- Funções do AG ---


def calcular_aptidao(cromossomo, items_info, capacidade, penalidade_peso_excedido=1000):
    """
    Calcula a aptidão de um cromossomo. Penaliza soluções que excedem a capacidade.

    Args:
        cromossomo (list): Lista binária representando o cromossomo.
        items_info (list): Lista de dicionários de itens (peso, valor).
        capacidade (int): Capacidade máxima da mochila.
        penalidade_peso_excedido (int): Fator de penalidade para excesso de peso.

    Returns:
        float: A aptidão calculada do cromossomo.
    """
    peso_total = sum(cromossomo[i] * items_info[i]["peso"]
                     for i in range(len(cromossomo)))
    valor_total = sum(cromossomo[i] * items_info[i]["valor"]
                      for i in range(len(cromossomo)))

    if peso_total > capacidade:
        return valor_total - penalidade_peso_excedido * (peso_total - capacidade)
    return valor_total


def inicializar_populacao_aleatoria(tamanho_populacao, num_itens):
    """
    Gera uma população inicial aleatória.

    Args:
        tamanho_populacao (int): Número de indivíduos na população.
        num_itens (int): Número total de itens no problema.

    Returns:
        list: Uma lista de cromossomos (listas binárias).
    """
    return [[random.randint(0, 1) for _ in range(num_itens)] for _ in range(tamanho_populacao)]


def inicializar_populacao_heuristica(tamanho_populacao, items_info, capacidade, num_itens):
    """
    Gera uma população inicial com base em heurísticas e aleatoriamente.

    Args:
        tamanho_populacao (int): Número de indivíduos na população.
        items_info (list): Lista de dicionários de itens (peso, valor).
        capacidade (int): Capacidade máxima da mochila.
        num_itens (int): Número total de itens no problema.

    Returns:
        list: Uma lista de cromossomos (listas binárias).
    """
    populacao = []
    # Adiciona alguns indivíduos gerados heuristicamente (maior razão valor/peso)
    # Ex: Metade da população por heurística
    num_heuristicos = tamanho_populacao // 2
    # Garante pelo menos um heurístico se a população não for 0
    if num_heuristicos == 0 and tamanho_populacao > 0:
        num_heuristicos = 1

    for _ in range(num_heuristicos):
        cromossomo_heuristico = [0] * num_itens
        # Calcula razão valor/peso e ordena itens
        # Tratamento para evitar divisão por zero se peso for 0 (embora raro em problemas de mochila)
        itens_ordenados = sorted(enumerate(items_info), key=lambda x: x[1]["valor"] / (
            x[1]["peso"] if x[1]["peso"] != 0 else 0.001), reverse=True)
        peso_atual = 0
        for idx, item in itens_ordenados:
            if peso_atual + item["peso"] <= capacidade:
                cromossomo_heuristico[idx] = 1
                peso_atual += item["peso"]
        populacao.append(cromossomo_heuristico)

    # Preenche o restante com indivíduos aleatórios para garantir diversidade
    while len(populacao) < tamanho_populacao:
        populacao.append([random.randint(0, 1) for _ in range(num_itens)])
    return populacao


def selecao_torneio(populacao, aptidoes, tamanho_torneio=3):
    """
    Seleciona dois pais usando o método de seleção por torneio.

    Args:
        populacao (list): Lista de cromossomos.
        aptidoes (list): Lista de aptidões correspondentes aos cromossomos.
        tamanho_torneio (int): Número de indivíduos no torneio.

    Returns:
        tuple: Um par de cromossomos selecionados como pais.
    """
    selecionados = []
    for _ in range(2):  # Selecionar 2 pais
        melhor_indice = -1
        melhor_aptidao = -float('inf')

        # Garante que tamanho_torneio não seja maior que a população disponível
        candidatos_indices = random.sample(
            range(len(populacao)), min(tamanho_torneio, len(populacao)))

        for idx in candidatos_indices:
            if aptidoes[idx] > melhor_aptidao:
                melhor_aptidao = aptidoes[idx]
                melhor_indice = idx
        selecionados.append(populacao[melhor_indice])
    return selecionados[0], selecionados[1]


def crossover_um_ponto(pai1, pai2):
    """
    Realiza o crossover de um ponto.

    Args:
        pai1 (list): Cromossomo do primeiro pai.
        pai2 (list): Cromossomo do segundo pai.

    Returns:
        tuple: Um par de cromossomos filhos.
    """
    if len(pai1) < 2:  # Crossover não faz sentido para cromossomos muito curtos
        return pai1, pai2
    ponto_corte = random.randint(1, len(pai1) - 1)
    filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
    filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]
    return filho1, filho2


def crossover_dois_pontos(pai1, pai2):
    """
    Realiza o crossover de dois pontos.

    Args:
        pai1 (list): Cromossomo do primeiro pai.
        pai2 (list): Cromossomo do segundo pai.

    Returns:
        tuple: Um par de cromossomos filhos.
    """
    if len(pai1) < 3:  # Crossover de dois pontos precisa de pelo menos 3 posições
        return pai1, pai2
    pontos_corte = sorted(random.sample(range(1, len(pai1)), 2))
    p1, p2 = pontos_corte[0], pontos_corte[1]
    filho1 = pai1[:p1] + pai2[p1:p2] + pai1[p2:]
    filho2 = pai2[:p1] + pai1[p1:p2] + pai2[p2:]
    return filho1, filho2


def crossover_uniforme(pai1, pai2, prob_troca=0.5):
    """
    Realiza o crossover uniforme.

    Args:
        pai1 (list): Cromossomo do primeiro pai.
        pai2 (list): Cromossomo do segundo pai.
        prob_troca (float): Probabilidade de um gene ser trocado entre os pais.

    Returns:
        tuple: Um par de cromossomos filhos.
    """
    filho1 = [pai1[i] if random.random() < prob_troca else pai2[i]
              for i in range(len(pai1))]
    filho2 = [pai2[i] if random.random() < prob_troca else pai1[i]
              for i in range(len(pai1))]
    return filho1, filho2


def mutacao(cromossomo, taxa_mutacao):
    """
    Realiza mutação por inversão de bit.

    Args:
        cromossomo (list): Cromossomo a ser mutado.
        taxa_mutacao (float): Probabilidade de um bit ser invertido.

    Returns:
        list: O cromossomo mutado.
    """
    mutated_cromossomo = list(
        cromossomo)  # Criar uma cópia para não modificar o original
    for i in range(len(mutated_cromossomo)):
        if random.random() < taxa_mutacao:
            mutated_cromossomo[i] = 1 - mutated_cromossomo[i]  # Inverte o bit
    return mutated_cromossomo
