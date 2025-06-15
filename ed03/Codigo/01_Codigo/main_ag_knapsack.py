import time
import numpy as np
import pandas as pd
import os  # Importar os para criação de diretório

# Importa as funções dos outros arquivos
from datasets_knapsack import load_knapsack_datasets
from funcoes_ag import (
    calcular_aptidao,
    inicializar_populacao_aleatoria,
    inicializar_populacao_heuristica,
    selecao_torneio,
    crossover_um_ponto,
    crossover_dois_pontos,
    crossover_uniforme,
    mutacao
)

# --- Algoritmo Genético Principal ---


def run_genetic_algorithm(items_info, capacidade, params):
    """
    Função principal do Algoritmo Genético para o Problema da Mochila.

    Args:
        items_info (list): Lista de dicionários, onde cada dit representa um item {'peso': x, 'valor': y}.
        capacidade (int): Capacidade máxima da mochila.
        params (dict): Dicionário de parâmetros do AG (tamanho_populacao, num_geracoes, etc.).

    Returns:
        tuple: (melhor_aptidao_encontrada_sem_penalidade, tempo_execucao_s)
    """
    tamanho_populacao = params['tamanho_populacao']
    num_geracoes = params['num_geracoes']
    taxa_mutacao = params['taxa_mutacao']
    tipo_crossover = params['tipo_crossover']
    inicializacao_tipo = params['inicializacao_tipo']
    criterio_parada = params['criterio_parada']
    geracoes_sem_melhoria_max = params.get(
        'geracoes_sem_melhoria_max', 50)  # Para critério de convergência

    num_itens = len(items_info)
    if num_itens == 0:
        return 0, 0  # Não há itens, então aptidão e tempo são 0

    start_time = time.time()

    # Inicializa a população
    if inicializacao_tipo == 'aleatoria':
        populacao = inicializar_populacao_aleatoria(
            tamanho_populacao, num_itens)
    else:  # 'heuristica'
        populacao = inicializar_populacao_heuristica(
            tamanho_populacao, items_info, capacidade, num_itens)

    # --- CORREÇÃO DE BUG: Inicializa melhor_cromossomo_global com o melhor da população inicial ---
    # Garante que melhor_cromossomo_global nunca seja None se a população não for vazia
    # Caso extremo de população vazia (tamanho_populacao = 0)
    if not populacao:
        # Cria um cromossomo vazio como fallback
        melhor_cromossomo_global = [0] * num_itens
        melhor_aptidao_global = 0  # Aptidão zero para mochila vazia
    else:
        aptidoes_iniciais = [calcular_aptidao(
            c, items_info, capacidade) for c in populacao]
        idx_melhor_inicial = np.argmax(aptidoes_iniciais)
        melhor_aptidao_global = aptidoes_iniciais[idx_melhor_inicial]
        # Copia o melhor da população inicial
        melhor_cromossomo_global = list(populacao[idx_melhor_inicial])

    geracoes_sem_melhoria_count = 0

    for geracao in range(num_geracoes):
        aptidoes = [calcular_aptidao(c, items_info, capacidade)
                    for c in populacao]

        # Verifica se há aptidões para evitar erro com np.argmax em lista vazia
        if aptidoes:
            idx_melhor_geracao = np.argmax(aptidoes)
            aptidao_melhor_geracao = aptidoes[idx_melhor_geracao]

            if aptidao_melhor_geracao > melhor_aptidao_global:
                melhor_aptidao_global = aptidao_melhor_geracao
                melhor_cromossomo_global = list(
                    populacao[idx_melhor_geracao])  # Cópia profunda
                geracoes_sem_melhoria_count = 0
            else:
                geracoes_sem_melhoria_count += 1
        # Se por algum motivo aptidoes estiver vazia (população pode ter encolhido inesperadamente)
        else:
            geracoes_sem_melhoria_count += 1

        # Critério de parada por convergência
        if criterio_parada == 'convergencia' and geracoes_sem_melhoria_count >= geracoes_sem_melhoria_max:
            break

        nova_populacao = []
        # Elitismo: Mantém o melhor indivíduo
        if melhor_cromossomo_global is not None:
            # Adiciona uma cópia do melhor
            nova_populacao.append(list(melhor_cromossomo_global))

        # Garante que sempre haja indivíduos suficientes para seleção e crossover
        # Se tamanho_populacao é 1, não há crossover/seleção como normalmente esperado.
        # Ajuste a lógica se tamanho_populacao for muito pequeno (e.g., < 2)
        while len(nova_populacao) < tamanho_populacao:
            # Garante que há pelo menos 'tamanho_torneio' indivíduos para seleção,
            # ou pelo menos 2 se tamanho_torneio for maior que len(populacao)
            if len(populacao) < 2:  # Mínimo de 2 para selecionar pais
                # Se a população for muito pequena, replique o melhor indivíduo
                if melhor_cromossomo_global is not None:
                    while len(nova_populacao) < tamanho_populacao:
                        nova_populacao.append(list(melhor_cromossomo_global))
                break  # Sai do loop se não há como gerar mais

            pai1, pai2 = selecao_torneio(populacao, aptidoes)

            if tipo_crossover == 'um_ponto':
                filho1, filho2 = crossover_um_ponto(pai1, pai2)
            elif tipo_crossover == 'dois_pontos':
                filho1, filho2 = crossover_dois_pontos(pai1, pai2)
            else:  # 'uniforme'
                filho1, filho2 = crossover_uniforme(pai1, pai2)

            filho1 = mutacao(filho1, taxa_mutacao)
            filho2 = mutacao(filho2, taxa_mutacao)

            nova_populacao.append(filho1)
            if len(nova_populacao) < tamanho_populacao:
                nova_populacao.append(filho2)

        # Se nova_populacao ainda não atingiu o tamanho_populacao, preenche com cópias dos melhores
        while len(nova_populacao) < tamanho_populacao and melhor_cromossomo_global is not None:
            nova_populacao.append(list(melhor_cromossomo_global))

        if nova_populacao:  # A população só é atualizada se não estiver vazia
            populacao = nova_populacao
        # else: # Caso extremo: nova_populacao ficou vazia. Mantém a população anterior ou loga erro.
        #     pass

    end_time = time.time()

    # Retornar a aptidão final sem penalidade para avaliação da qualidade
    # Garante que melhor_cromossomo_global não é None ao chamar calcular_aptidao
    if melhor_cromossomo_global is None:  # Fallback final, embora improvável com as correções
        final_aptidao_sem_penalidade = 0
    else:
        final_aptidao_sem_penalidade = calcular_aptidao(
            melhor_cromossomo_global, items_info, capacidade, penalidade_peso_excedido=0)

    return final_aptidao_sem_penalidade, (end_time - start_time)


# --- Configurações dos Experimentos ---
crossover_tipos = ['um_ponto', 'dois_pontos', 'uniforme']
taxas_mutacao = [0.001, 0.01, 0.05]
inicializacao_tipos = ['aleatoria', 'heuristica']
criterio_parada_tipos = ['fixa_geracoes', 'convergencia']
num_execucoes = 30  # Número de repetições para cada combinação de parâmetros
tamanho_populacao_base = 100
num_geracoes_base = 1000  # Para o critério de parada 'fixa_geracoes'
geracoes_sem_melhoria_max_base = 50  # Para o critério de parada 'convergencia'

# Carregar todos os datasets
# O caminho '../02_Dados/' assume que você está executando main_ag_knapsack.py
# de dentro da pasta '01_Codigo/'
all_datasets = load_knapsack_datasets(data_folder=os.path.join(
    os.path.dirname(__file__), '..', '02_Dados'))


# Estrutura para armazenar os resultados
results_data = []

# --- Execução dos Experimentos ---
print("Iniciando bateria de experimentos...")
total_configs = len(all_datasets) * len(crossover_tipos) * \
    len(taxas_mutacao) * len(inicializacao_tipos) * len(criterio_parada_tipos)
current_config_num = 0

for dataset_name, data in all_datasets.items():
    items = data['items']
    capacidade = data['capacidade']

    print(
        f"\n--- Iniciando experimentos para {dataset_name} (Capacidade: {capacidade}, Itens: {len(items)}) ---")

    for c_type in crossover_tipos:
        for m_rate in taxas_mutacao:
            for init_type in inicializacao_tipos:
                for stop_crit in criterio_parada_tipos:
                    current_config_num += 1
                    print(
                        f"  Progresso: {current_config_num}/{total_configs} - Config: Crossover={c_type}, Mutação={m_rate}, Init={init_type}, Parada={stop_crit}")

                    aptidoes_execucao = []
                    tempos_execucao = []

                    for i in range(num_execucoes):
                        params = {
                            'tamanho_populacao': tamanho_populacao_base,
                            'num_geracoes': num_geracoes_base,
                            'taxa_mutacao': m_rate,
                            'tipo_crossover': c_type,
                            'inicializacao_tipo': init_type,
                            'criterio_parada': stop_crit,
                            'geracoes_sem_melhoria_max': geracoes_sem_melhoria_max_base
                        }

                        melhor_aptidao, tempo = run_genetic_algorithm(
                            items, capacidade, params)

                        aptidoes_execucao.append(melhor_aptidao)
                        tempos_execucao.append(tempo)

                    # Calcular métricas para a configuração atual
                    media_aptidao = np.mean(aptidoes_execucao)
                    dp_aptidao = np.std(aptidoes_execucao)
                    melhor_aptidao_absoluta = np.max(aptidoes_execucao)
                    media_tempo = np.mean(tempos_execucao)
                    dp_tempo = np.std(tempos_execucao)

                    # Adicionar ao coletor de resultados, formatando para strings
                    results_data.append({
                        "Dataset": dataset_name,
                        "Configuracao": f"Crossover: {c_type}, Mutação: {m_rate}, Inicialização: {init_type}, Parada: {stop_crit}",
                        "Media_Aptidao": f"{media_aptidao:.2f}",
                        "DP_Aptidao": f"{dp_aptidao:.2f}",
                        "Melhor_Aptidao": f"{melhor_aptidao_absoluta:.2f}",
                        "Media_Tempo": f"{media_tempo:.4f}",
                        "DP_Tempo": f"{dp_tempo:.4f}"
                    })

# --- Imprimir os resultados e salvar em CSV ---
df_results = pd.DataFrame(results_data)
print("\n--- Resultados Finais ---")
print(df_results)

# Salvando os resultados na pasta '02_Dados'
output_folder = os.path.join(os.path.dirname(__file__), '..', '02_Dados')
os.makedirs(output_folder, exist_ok=True)  # Garante que a pasta existe
output_csv_path = os.path.join(output_folder, "resultados_ag_knapsack.csv")
df_results.to_csv(output_csv_path, index=False)

print(f"\nResultados salvos em: {output_csv_path}")

# Opcional: Salvar em Excel se tiver a biblioteca openpyxl instalada
try:
    output_excel_path = os.path.join(
        output_folder, "resultados_ag_knapsack.xlsx")
    df_results.to_excel(output_excel_path, index=False)
    print(f"Resultados também salvos em: {output_excel_path}")
except ImportError:
    print("Para salvar em .xlsx, instale 'openpyxl' (pip install openpyxl).")
except Exception as e:
    print(f"Erro ao salvar em Excel: {e}")
