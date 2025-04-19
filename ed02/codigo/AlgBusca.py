import numpy as np
import time
import psutil
import os
import heapq
from collections import deque
import pandas as pd
import csv

# Classe para representar o estado do quebra-cabeça
class PuzzleState:
    def __init__(self, state, parent=None, move=None, depth=0, cost=0):
        self.state = state  # Estado atual do quebra-cabeça (matriz 3x3)
        self.parent = parent  # Estado pai (de onde veio)
        self.move = move  # Movimento que gerou este estado
        self.depth = depth  # Profundidade na árvore de busca
        self.cost = cost  # Custo para chegar a este estado
        
        # Encontra a posição do espaço vazio (0)
        self.blank_row, self.blank_col = np.where(self.state == 0)
        self.blank_row, self.blank_col = int(self.blank_row), int(self.blank_col)
        
    def __lt__(self, other):
        return self.cost < other.cost
    
    def __eq__(self, other):
        if other is None:
            return False
        return np.array_equal(self.state, other.state)
    
    def __hash__(self):
        return hash(str(self.state))
    
    def is_goal(self):
        # Estado objetivo
        goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        return np.array_equal(self.state, goal)
    
    def get_possible_moves(self):
        # Movimentos possíveis: cima, baixo, esquerda, direita
        moves = []
        
        # Cima (mover o espaço vazio para cima)
        if self.blank_row > 0:
            moves.append('cima')
        
        # Baixo (mover o espaço vazio para baixo)
        if self.blank_row < 2:
            moves.append('baixo')
        
        # Esquerda (mover o espaço vazio para a esquerda)
        if self.blank_col > 0:
            moves.append('esquerda')
        
        # Direita (mover o espaço vazio para a direita)
        if self.blank_col < 2:
            moves.append('direita')
            
        return moves
    
    def move_blank(self, direction):
        # Cria uma cópia do estado atual
        new_state = np.copy(self.state)
        
        # Realiza o movimento do espaço vazio
        if direction == 'cima':
            # Troca o espaço vazio com o número acima
            new_state[self.blank_row, self.blank_col] = new_state[self.blank_row - 1, self.blank_col]
            new_state[self.blank_row - 1, self.blank_col] = 0
        
        elif direction == 'baixo':
            # Troca o espaço vazio com o número abaixo
            new_state[self.blank_row, self.blank_col] = new_state[self.blank_row + 1, self.blank_col]
            new_state[self.blank_row + 1, self.blank_col] = 0
        
        elif direction == 'esquerda':
            # Troca o espaço vazio com o número à esquerda
            new_state[self.blank_row, self.blank_col] = new_state[self.blank_row, self.blank_col - 1]
            new_state[self.blank_row, self.blank_col - 1] = 0
        
        elif direction == 'direita':
            # Troca o espaço vazio com o número à direita
            new_state[self.blank_row, self.blank_col] = new_state[self.blank_row, self.blank_col + 1]
            new_state[self.blank_row, self.blank_col + 1] = 0
        
        return PuzzleState(new_state, self, direction, self.depth + 1)
    
    def get_neighbors(self):
        # Gera todos os estados vizinhos possíveis
        neighbors = []
        for move in self.get_possible_moves():
            neighbors.append(self.move_blank(move))
        return neighbors

# Funções heurísticas para busca informada

def manhattan_distance(state):
    """Distância de Manhattan: soma das distâncias de cada peça à sua posição final"""
    goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    distance = 0
    
    for i in range(3):
        for j in range(3):
            if state[i, j] != 0:  # Ignoramos o espaço vazio
                # Encontra onde esse número deveria estar no estado objetivo
                goal_positions = np.where(goal == state[i, j])
                goal_row, goal_col = goal_positions[0][0], goal_positions[1][0]
                
                # Calcula a distância de Manhattan
                distance += abs(i - goal_row) + abs(j - goal_col)
    
    return distance

def misplaced_tiles(state):
    """Número de peças fora do lugar"""
    goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    count = 0
    
    for i in range(3):
        for j in range(3):
            if state[i, j] != 0 and state[i, j] != goal[i, j]:
                count += 1
    
    return count

# Algoritmos de busca

def bfs(initial_state):
    """Busca em Largura (BFS)"""
    start_time = time.time()
    start_memory = process_memory()
    
    if initial_state.is_goal():
        return initial_state, 0, 0, 0, 0
    
    # Fila para BFS
    queue = deque([initial_state])
    # Conjunto para nós visitados
    visited = set()
    visited.add(hash(str(initial_state.state)))
    
    nodes_expanded = 0
    max_queue_size = 1
    
    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        
        # Remove o primeiro estado da fila
        current_state = queue.popleft()
        nodes_expanded += 1
        
        # Gera todos os estados vizinhos
        for neighbor in current_state.get_neighbors():
            neighbor_hash = hash(str(neighbor.state))
            
            if neighbor_hash not in visited:
                if neighbor.is_goal():
                    end_time = time.time()
                    end_memory = process_memory()
                    
                    return neighbor, nodes_expanded, end_time - start_time, max_queue_size, end_memory - start_memory
                
                queue.append(neighbor)
                visited.add(neighbor_hash)
    
    # Se chegou aqui, não encontrou solução
    end_time = time.time()
    end_memory = process_memory()
    return None, nodes_expanded, end_time - start_time, max_queue_size, end_memory - start_memory

def dfs(initial_state, max_depth=50):
    """Busca em Profundidade (DFS) com limite de profundidade"""
    start_time = time.time()
    start_memory = process_memory()
    
    if initial_state.is_goal():
        return initial_state, 0, 0, 0, 0
    
    # Pilha para DFS
    stack = [initial_state]
    # Conjunto para nós visitados
    visited = set()
    visited.add(hash(str(initial_state.state)))
    
    nodes_expanded = 0
    max_stack_size = 1
    
    while stack:
        max_stack_size = max(max_stack_size, len(stack))
        
        # Remove o último estado da pilha
        current_state = stack.pop()
        nodes_expanded += 1
        
        # Se atingiu o limite de profundidade, continua
        if current_state.depth >= max_depth:
            continue
        
        # Gera todos os estados vizinhos
        for neighbor in current_state.get_neighbors():
            neighbor_hash = hash(str(neighbor.state))
            
            if neighbor_hash not in visited:
                if neighbor.is_goal():
                    end_time = time.time()
                    end_memory = process_memory()
                    
                    return neighbor, nodes_expanded, end_time - start_time, max_stack_size, end_memory - start_memory
                
                stack.append(neighbor)
                visited.add(neighbor_hash)
    
    # Se chegou aqui, não encontrou solução dentro do limite de profundidade
    end_time = time.time()
    end_memory = process_memory()
    return None, nodes_expanded, end_time - start_time, max_stack_size, end_memory - start_memory

def greedy_search(initial_state, heuristic=manhattan_distance):
    """Busca Gulosa - usa apenas a heurística"""
    start_time = time.time()
    start_memory = process_memory()
    
    if initial_state.is_goal():
        return initial_state, 0, 0, 0, 0
    
    # Fila de prioridade para busca gulosa
    priority_queue = []
    # Conjunto para nós visitados
    visited = set()
    
    # Adiciona o estado inicial na fila de prioridade
    h = heuristic(initial_state.state)
    heapq.heappush(priority_queue, (h, 0, initial_state))  # (heurística, contador, estado)
    
    nodes_expanded = 0
    max_queue_size = 1
    counter = 1  # Contador para desempate
    
    while priority_queue:
        max_queue_size = max(max_queue_size, len(priority_queue))
        
        # Remove o estado com menor valor heurístico
        _, _, current_state = heapq.heappop(priority_queue)
        
        # Se já visitou este estado, continua
        current_hash = hash(str(current_state.state))
        if current_hash in visited:
            continue
        
        visited.add(current_hash)
        nodes_expanded += 1
        
        # Verifica se é o estado objetivo
        if current_state.is_goal():
            end_time = time.time()
            end_memory = process_memory()
            
            return current_state, nodes_expanded, end_time - start_time, max_queue_size, end_memory - start_memory
        
        # Gera todos os estados vizinhos
        for neighbor in current_state.get_neighbors():
            neighbor_hash = hash(str(neighbor.state))
            
            if neighbor_hash not in visited:
                h = heuristic(neighbor.state)
                heapq.heappush(priority_queue, (h, counter, neighbor))
                counter += 1
    
    # Se chegou aqui, não encontrou solução
    end_time = time.time()
    end_memory = process_memory()
    return None, nodes_expanded, end_time - start_time, max_queue_size, end_memory - start_memory

def a_star(initial_state, heuristic=manhattan_distance):
    """Algoritmo A* - usa g(n) + h(n)"""
    start_time = time.time()
    start_memory = process_memory()
    
    if initial_state.is_goal():
        return initial_state, 0, 0, 0, 0
    
    # Fila de prioridade para A*
    priority_queue = []
    # Conjunto para nós visitados
    visited = set()
    
    # Adiciona o estado inicial na fila de prioridade
    h = heuristic(initial_state.state)
    g = 0  # Custo até agora (profundidade)
    f = g + h  # f(n) = g(n) + h(n)
    
    heapq.heappush(priority_queue, (f, 0, initial_state))  # (f(n), contador, estado)
    
    nodes_expanded = 0
    max_queue_size = 1
    counter = 1  # Contador para desempate
    
    while priority_queue:
        max_queue_size = max(max_queue_size, len(priority_queue))
        
        # Remove o estado com menor f(n)
        _, _, current_state = heapq.heappop(priority_queue)
        
        # Se já visitou este estado, continua
        current_hash = hash(str(current_state.state))
        if current_hash in visited:
            continue
        
        visited.add(current_hash)
        nodes_expanded += 1
        
        # Verifica se é o estado objetivo
        if current_state.is_goal():
            end_time = time.time()
            end_memory = process_memory()
            
            return current_state, nodes_expanded, end_time - start_time, max_queue_size, end_memory - start_memory
        
        # Gera todos os estados vizinhos
        for neighbor in current_state.get_neighbors():
            neighbor_hash = hash(str(neighbor.state))
            
            if neighbor_hash not in visited:
                g = neighbor.depth  # Custo até agora (profundidade)
                h = heuristic(neighbor.state)
                f = g + h  # f(n) = g(n) + h(n)
                
                heapq.heappush(priority_queue, (f, counter, neighbor))
                counter += 1
    
    # Se chegou aqui, não encontrou solução
    end_time = time.time()
    end_memory = process_memory()
    return None, nodes_expanded, end_time - start_time, max_queue_size, end_memory - start_memory

def process_memory():
    """Retorna o uso de memória em MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # em MB

def get_solution_path(solution_state):
    """Recupera o caminho da solução a partir do estado final"""
    if solution_state is None:
        return []
    
    path = []
    current = solution_state
    
    while current is not None:
        path.append((current.state, current.move))
        current = current.parent
    
    # Inverte o caminho para obter a sequência do início ao fim
    path.reverse()
    return path

def print_solution(solution_path):
    """Imprime a solução passo a passo"""
    if not solution_path:
        print("Nenhuma solução encontrada!")
        return
    
    print(f"Solução encontrada com {len(solution_path) - 1} movimentos:")
    
    for i, (state, move) in enumerate(solution_path):
        print(f"\nPasso {i}:")
        if move:
            print(f"Movimento: {move}")
        
        for row in state:
            print(" ".join(str(cell) for cell in row))

def load_puzzles_from_csv(filename):
    """Carrega instâncias do quebra-cabeça do arquivo CSV"""
    puzzles = []
    
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Pula o cabeçalho
        
        for row in csv_reader:
            # Converte a linha em uma matriz 3x3
            puzzle = [int(cell) for cell in row]
            puzzle_matrix = np.reshape(puzzle, (3, 3))
            puzzles.append(puzzle_matrix)
    
    return puzzles

def run_algorithms(initial_state):
    """Executa todos os algoritmos de busca para um estado inicial"""
    results = {}
    
    # BFS
    print("\nExecutando Busca em Largura (BFS)...")
    solution, nodes, time_taken, max_queue, memory_used = bfs(initial_state)
    if solution:
        path = get_solution_path(solution)
        solution_length = len(path) - 1
        results["BFS"] = {
            "solution_found": True,
            "solution_length": solution_length,
            "nodes_expanded": nodes,
            "time_taken": time_taken,
            "max_queue_size": max_queue,
            "memory_used": memory_used,
            "solution": solution
        }
    else:
        results["BFS"] = {
            "solution_found": False,
            "nodes_expanded": nodes,
            "time_taken": time_taken,
            "max_queue_size": max_queue,
            "memory_used": memory_used
        }
    
    # DFS
    print("Executando Busca em Profundidade (DFS)...")
    solution, nodes, time_taken, max_queue, memory_used = dfs(initial_state)
    if solution:
        path = get_solution_path(solution)
        solution_length = len(path) - 1
        results["DFS"] = {
            "solution_found": True,
            "solution_length": solution_length,
            "nodes_expanded": nodes,
            "time_taken": time_taken,
            "max_queue_size": max_queue,
            "memory_used": memory_used,
            "solution": solution
        }
    else:
        results["DFS"] = {
            "solution_found": False,
            "nodes_expanded": nodes,
            "time_taken": time_taken,
            "max_queue_size": max_queue,
            "memory_used": memory_used
        }
    
    # Busca Gulosa com Distância de Manhattan
    print("Executando Busca Gulosa (Distância de Manhattan)...")
    solution, nodes, time_taken, max_queue, memory_used = greedy_search(initial_state, manhattan_distance)
    if solution:
        path = get_solution_path(solution)
        solution_length = len(path) - 1
        results["Greedy-Manhattan"] = {
            "solution_found": True,
            "solution_length": solution_length,
            "nodes_expanded": nodes,
            "time_taken": time_taken,
            "max_queue_size": max_queue,
            "memory_used": memory_used,
            "solution": solution
        }
    else:
        results["Greedy-Manhattan"] = {
            "solution_found": False,
            "nodes_expanded": nodes,
            "time_taken": time_taken,
            "max_queue_size": max_queue,
            "memory_used": memory_used
        }
    
    # A* com Distância de Manhattan
    print("Executando A* (Distância de Manhattan)...")
    solution, nodes, time_taken, max_queue, memory_used = a_star(initial_state, manhattan_distance)
    if solution:
        path = get_solution_path(solution)
        solution_length = len(path) - 1
        results["A*-Manhattan"] = {
            "solution_found": True,
            "solution_length": solution_length,
            "nodes_expanded": nodes,
            "time_taken": time_taken,
            "max_queue_size": max_queue,
            "memory_used": memory_used,
            "solution": solution
        }
    else:
        results["A*-Manhattan"] = {
            "solution_found": False,
            "nodes_expanded": nodes,
            "time_taken": time_taken,
            "max_queue_size": max_queue,
            "memory_used": memory_used
        }
    
    return results

def print_summary(results):
    """Imprime um resumo dos resultados de todos os algoritmos"""
    print("\n========== RESUMO DOS RESULTADOS ==========")
    
    # Cria uma tabela para os resultados
    data = []
    for algorithm, result in results.items():
        row = [algorithm]
        if result["solution_found"]:
            row.extend([
                "Sim",
                result["solution_length"],
                result["nodes_expanded"],
                f"{result['time_taken']:.6f} s",
                result["max_queue_size"],
                f"{result['memory_used']:.2f} MB"
            ])
        else:
            row.extend(["Não", "N/A", result["nodes_expanded"], f"{result['time_taken']:.6f} s", result["max_queue_size"], f"{result['memory_used']:.2f} MB"])
        data.append(row)
    
    # Cria o DataFrame
    df = pd.DataFrame(data, columns=["Algoritmo", "Solução Encontrada", "Comprimento da Solução", "Nós Expandidos", "Tempo de Execução", "Tamanho Máximo da Fila/Pilha", "Memória Utilizada"])
    
    # Imprime o DataFrame
    print(df.to_string(index=False))
    print("\n===========================================")

def main():
    # Carrega as instâncias do arquivo CSV
    try:
        puzzles = load_puzzles_from_csv("ed02-puzzle8.csv")
        print(f"Carregadas {len(puzzles)} instâncias de quebra-cabeças.")
    except Exception as e:
        print(f"Erro ao carregar o arquivo CSV: {e}")
        # Se não conseguir carregar o arquivo, usa um exemplo
        puzzles = [
            np.array([[1, 2, 3], [4, 0, 6], [7, 5, 8]]),  # Exemplo fácil
            np.array([[1, 2, 3], [0, 4, 6], [7, 5, 8]]),  # Outro exemplo
            np.array([[7, 2, 4], [5, 0, 6], [8, 3, 1]])   # Exemplo mais difícil
        ]
    
    # Executa os algoritmos para cada instância
    for i, puzzle in enumerate(puzzles[:3]):  # Limita a 3 puzzles para o exemplo
        print(f"\n\n========== INSTÂNCIA {i+1} ==========")
        print("Estado inicial:")
        for row in puzzle:
            print(" ".join(str(cell) for cell in row))
        
        initial_state = PuzzleState(puzzle)
        results = run_algorithms(initial_state)
        print_summary(results)
        
        # Imprime a solução do A* como exemplo
        if "A*-Manhattan" in results and results["A*-Manhattan"]["solution_found"]:
            print("\nSolução encontrada pelo A* (Manhattan):")
            solution_path = get_solution_path(results["A*-Manhattan"]["solution"])
            print_solution(solution_path)
        else:
            print("\nA* não encontrou solução para esta instância.")

if __name__ == "__main__":
    main()