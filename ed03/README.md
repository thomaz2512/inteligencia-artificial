# Otimização do Problema da Mochila Utilizando Algoritmos Genéticos

Este projeto é um trabalho acadêmico com o objetivo de **implementar e analisar Algoritmos Genéticos (AGs)** aplicados ao clássico **Problema da Mochila (Knapsack Problem)**. O estudo foca em comparar o impacto de diferentes configurações de operadores genéticos no desempenho do AG, avaliando tanto a **qualidade da solução** quanto o **tempo de execução**.

---

## Objetivo do Projeto

A meta principal deste trabalho é explorar como a variação de parâmetros como o tipo de **crossover**, a **taxa de mutação**, a **estratégia de inicialização da população** e o **critério de parada** afeta a eficiência e a eficácia de um Algoritmo Genético na resolução de um problema de otimização combinatória.

O trabalho foi desenvolvido como um relatório científico que detalha:
* A introdução ao problema e a justificativa para o uso de AGs.
* A metodologia de implementação do AG e as variações testadas.
* Os resultados quantitativos da comparação entre as configurações.
* A discussão sobre o impacto de cada operador.
* A conclusão com a identificação da melhor configuração e sugestões futuras.

---

## Estrutura do Projeto

O projeto está organizado nas seguintes pastas:
├── 01_Codigo/
│   ├── main_ag_knapsack.py      # Script principal para executar os experimentos do AG.
│   ├── funcoes_ag.py            # Módulo com as funções auxiliares do AG (aptidão, seleção, crossover, mutação, etc.).
│   └── datasets_knapsack.py     # Módulo para carregar os datasets do Problema da Mochila.
├── 02_Dados/
│   ├── ED03 - Instancias dos problemas.zip # Arquivo ZIP com os datasets do problema.
│   └── resultados_ag_knapsack.csv          # Saída dos resultados dos experimentos (gerado por main_ag_knapsack.py).
│   └── resultados_ag_knapsack.xlsx         # Saída dos resultados dos experimentos (opcional, gerado por main_ag_knapsack.py).
├── Relatorio_Cientifico.pdf     # O relatório científico final em formato PDF (gerado a partir do arquivo .tex).
└── README.md                    # Este arquivo.

---

## Como Executar o Projeto

Siga os passos abaixo para replicar os experimentos e gerar os resultados:

### Pré-requisitos

Certifique-se de ter o **Python 3.x** instalado. As seguintes bibliotecas Python são necessárias:
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `openpyxl` (opcional, para salvar resultados em `.xlsx`)

Você pode instalá-las usando pip:
```bash
pip install numpy pandas matplotlib seaborn openpyxl

```

### Passos de Execução
Baixe e extraia os datasets:

1. Execute o script principal:
2. Navegue até a pasta 01_Codigo/ no seu terminal.
3. Execute o script main_ag_knapsack.py:

```Bash
python main_ag_knapsack.py
```
## Requisitos

- Python 3.7+
- Tkinter (normalmente já vem instalado com Python)
## Como Executar

1. Certifique-se de que o arquivo principal está salvo;
2. Execute o script com o python:
~~~
python BatalhaNaval.py
~~~
