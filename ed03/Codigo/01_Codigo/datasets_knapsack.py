import pandas as pd
import os


def load_knapsack_datasets(data_folder='../02_Dados/'):
    """
    Carrega os datasets do Problema da Mochila a partir de arquivos CSV.
    Define as capacidades para cada mochila.
    Inclui tratamento robusto para conversão de Peso e Valor para números.
    Este script foi ajustado para CSVs com 3 colunas (Item_ID, Peso, Valor) por linha.

    Args:
        data_folder (str): Caminho para a pasta onde os arquivos CSV estão localizados.

    Returns:
        dict: Um dicionário onde as chaves são os nomes dos datasets (ex: "Dataset 1")
              e os valores são dicionários contendo 'items' (lista de dicts com peso/valor)
              e 'capacidade'.
    """
    datasets = {}

    # Capacidades da mochila para cada dataset (definidas manualmente)
    # AJUSTE ESTES VALORES CONFORME A SUA NECESSIDADE PARA CADA DATASET
    capacidades_definidas = {
        "knapsack_1.csv": 150,
        "knapsack_2.csv": 200,
        "knapsack_3.csv": 600,
        "knapsack_4.csv": 300,
        "knapsack_5.csv": 400,
        "knapsack_6.csv": 500,
        "knapsack_7.csv": 550,
        "knapsack_8.csv": 600,
        "knapsack_9.csv": 700,
        "knapsack_10.csv": 350
    }

    # Itera sobre os arquivos CSV (knapsack_1.csv a knapsack_10.csv)
    for i in range(1, 11):
        file_name = f"knapsack_{i}.csv"
        file_path = os.path.join(data_folder, file_name)

        if not os.path.exists(file_path):
            print(
                f"Aviso: Arquivo '{file_path}' não encontrado. Pulando este dataset.")
            continue

        try:
            # --- CORREÇÃO: Lê o CSV com 3 colunas, sem cabeçalho ---
            # df = pd.read_csv(file_path, header=None, names=['item_id', 'peso', 'valor'])
            # A linha acima é o ideal, mas se o CSV tiver um cabeçalho real "Item,Peso,Valor",
            # o header=None faria a primeira linha virar dados.
            # A abordagem mais segura é tentar ler e inferir, e se der erro, tentar sem cabeçalho.

            try:
                # Tenta ler com cabeçalho primeiro (default)
                df = pd.read_csv(file_path)
                # Verifica se as colunas 'peso' e 'valor' (ou similar) existem
                df.columns = [col.strip().lower()
                              for col in df.columns]  # Padroniza
                if 'peso' not in df.columns or 'valor' not in df.columns:
                    # Se não encontrou as colunas padrão, tenta ler sem cabeçalho e nomear
                    df = pd.read_csv(file_path, header=None, names=[
                                     'item_id', 'peso', 'valor'])
                    # Verifica se realmente resultou em 3 colunas
                    if df.shape[1] != 3:
                        raise ValueError(
                            f"CSV '{file_name}' não tem o formato esperado (3 colunas com ou sem cabeçalho).")
            except Exception as e_read:
                # Se a primeira tentativa falhar (ex: cabeçalho com 1 coluna "Item,Peso,Valor")
                # ou outro erro de parsing, tenta ler sem cabeçalho e nomear explicitamente
                df = pd.read_csv(file_path, header=None, names=[
                                 'item_id', 'peso', 'valor'])
                if df.shape[1] != 3:  # Se mesmo assim não tiver 3 colunas
                    raise ValueError(
                        f"CSV '{file_name}' não tem o formato esperado (3 colunas).")

            # Agora garantimos que df tem as colunas 'item_id', 'peso', 'valor'
            # Garante que as colunas 'peso' e 'valor' são tratadas como strings antes da conversão numérica
            # Para remover separadores de milhares e garantir ponto como decimal
            df['peso'] = df['peso'].astype(str).str.replace(
                '.', '', regex=False).str.replace(',', '.', regex=False)
            df['valor'] = df['valor'].astype(str).str.replace(
                '.', '', regex=False).str.replace(',', '.', regex=False)

            # Converte para numérico. errors='coerce' transforma valores que não podem ser convertidos em NaN.
            df['peso'] = pd.to_numeric(df['peso'], errors='coerce')
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')

            # Preenche quaisquer NaNs resultantes da conversão com 0 e converte para inteiro
            df['peso'] = df['peso'].fillna(0).astype(int)
            df['valor'] = df['valor'].fillna(0).astype(int)

            items_list = df[['peso', 'valor']].to_dict(orient='records')

            datasets[f"Dataset {i}"] = {
                "items": items_list,
                "capacidade": capacidades_definidas.get(file_name, 0)
            }
        except pd.errors.EmptyDataError:
            print(
                f"Aviso: Arquivo '{file_name}' está vazio. Pulando este dataset.")
            continue
        except Exception as e:
            print(
                f"Erro inesperado ao carregar ou processar '{file_name}': {e}. Pulando este dataset.")
            # Imprime o erro real para depuração
            print(f"Detalhes do erro: {e}")
            continue

    return datasets


if __name__ == '__main__':
    # Este bloco é para testar se o carregamento dos datasets funciona
    loaded_datasets = load_knapsack_datasets()
    for name, data in loaded_datasets.items():
        print(f"\n--- {name} ---")
        print(f"Número de itens: {len(data['items'])}")
        print(f"Capacidade da mochila: {data['capacidade']}")
        if data['items']:  # Verifica se a lista de itens não está vazia
            print("Primeiros 5 itens:", data['items'][:5])
        else:
            print("Nenhum item válido carregado.")
