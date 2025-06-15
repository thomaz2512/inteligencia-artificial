import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Dados brutos fornecidos anteriormente
data = """
Dataset 1	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	13574.00	32.31	13580.00	0.4209	0.0090
Dataset 1	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	13580.00	0.00	13580.00	0.0210	0.0003
Dataset 1	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4196	0.0041
Dataset 1	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	13580.00	0.00	13580.00	0.0212	0.0012
Dataset 1	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4206	0.0051
Dataset 1	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	13580.00	0.00	13580.00	0.0213	0.0005
Dataset 1	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4236	0.0092
Dataset 1	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	13580.00	0.00	13580.00	0.0212	0.0005
Dataset 1	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4375	0.0208
Dataset 1	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	13580.00	0.00	13580.00	0.0228	0.0022
Dataset 1	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4409	0.0163
Dataset 1	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	13580.00	0.00	13580.00	0.0214	0.0004
Dataset 1	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4921	0.0222
Dataset 1	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	13580.00	0.00	13580.00	0.0244	0.0007
Dataset 1	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4820	0.0053
Dataset 1	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	13580.00	0.00	13580.00	0.0243	0.0014
Dataset 1	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4837	0.0024
Dataset 1	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	13580.00	0.00	13580.00	0.0243	0.0005
Dataset 1	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4828	0.0031
Dataset 1	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	13580.00	0.00	13580.00	0.0242	0.0007
Dataset 1	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4864	0.0025
Dataset 1	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	13580.00	0.00	13580.00	0.0248	0.0006
Dataset 1	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4868	0.0025
Dataset 1	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	13580.00	0.00	13580.00	0.0243	0.0003
Dataset 1	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4250	0.0035
Dataset 1	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	13574.00	32.31	13580.00	0.0217	0.0007
Dataset 1	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4248	0.0022
Dataset 1	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	13580.00	0.00	13580.00	0.0217	0.0011
Dataset 1	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4260	0.0022
Dataset 1	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	13580.00	0.00	13580.00	0.0222	0.0034
Dataset 1	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4266	0.0020
Dataset 1	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	13580.00	0.00	13580.00	0.0212	0.0003
Dataset 1	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4295	0.0028
Dataset 1	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	13580.00	0.00	13580.00	0.0217	0.0006
Dataset 1	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	13580.00	0.00	13580.00	0.4290	0.0039
Dataset 1	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	13580.00	0.00	13580.00	0.0216	0.0006
Dataset 2	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	17336.67	162.14	17380.00	0.4407	0.0091
Dataset 2	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	17336.67	162.14	17380.00	0.0225	0.0007
Dataset 2	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	17092.33	356.11	17380.00	0.4338	0.0024
Dataset 2	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	17109.00	334.18	17380.00	0.0221	0.0005
Dataset 2	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.4369	0.0066
Dataset 2	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	17380.00	0.00	17380.00	0.0225	0.0009
Dataset 2	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.4375	0.0036
Dataset 2	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	17124.67	338.69	17380.00	0.0258	0.0068
Dataset 2	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.4391	0.0022
Dataset 2	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	17380.00	0.00	17380.00	0.0226	0.0008
Dataset 2	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.4391	0.0032
Dataset 2	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	17380.00	0.00	17380.00	0.0250	0.0037
Dataset 2	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.5024	0.0130
Dataset 2	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	17380.00	0.00	17380.00	0.0255	0.0010
Dataset 2	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	17054.00	376.59	17380.00	0.4960	0.0053
Dataset 2	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	16913.00	361.42	17380.00	0.0252	0.0004
Dataset 2	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.5053	0.0022
Dataset 2	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	17380.00	0.00	17380.00	0.0255	0.0008
Dataset 2	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.4979	0.0028
Dataset 2	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	17146.00	359.47	17380.00	0.0277	0.0053
Dataset 2	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.5172	0.0200
Dataset 2	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	17380.00	0.00	17380.00	0.0268	0.0015
Dataset 2	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.5234	0.0167
Dataset 2	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	17380.00	0.00	17380.00	0.0289	0.0036
Dataset 2	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	17358.33	116.68	17380.00	0.4504	0.0075
Dataset 2	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	17358.33	116.68	17380.00	0.0229	0.0008
Dataset 2	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	17086.67	363.11	17380.00	0.4480	0.0034
Dataset 2	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	16967.33	389.54	17380.00	0.0229	0.0007
Dataset 2	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.4520	0.0067
Dataset 2	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	17358.33	116.68	17380.00	0.0231	0.0007
Dataset 2	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.4489	0.0026
Dataset 2	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	17108.33	360.57	17380.00	0.0253	0.0048
Dataset 2	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.4559	0.0046
Dataset 2	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	17380.00	0.00	17380.00	0.0234	0.0009
Dataset 2	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	17380.00	0.00	17380.00	0.4545	0.0065
Dataset 2	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	17380.00	0.00	17380.00	0.0248	0.0026
Dataset 3	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.4749	0.0071
Dataset 3	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	26510.00	0.00	26510.00	0.0257	0.0023
Dataset 3	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.4727	0.0030
Dataset 3	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	26510.00	0.00	26510.00	0.0238	0.0005
Dataset 3	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.4759	0.0039
Dataset 3	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	26510.00	0.00	26510.00	0.0248	0.0009
Dataset 3	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.4739	0.0027
Dataset 3	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	26510.00	0.00	26510.00	0.0237	0.0003
Dataset 3	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.4773	0.0029
Dataset 3	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	26510.00	0.00	26510.00	0.0249	0.0009
Dataset 3	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.4794	0.0061
Dataset 3	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	26510.00	0.00	26510.00	0.0245	0.0013
Dataset 3	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.5375	0.0051
Dataset 3	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	26510.00	0.00	26510.00	0.0280	0.0008
Dataset 3	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.5367	0.0036
Dataset 3	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	26510.00	0.00	26510.00	0.0271	0.0004
Dataset 3	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.5378	0.0052
Dataset 3	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	26510.00	0.00	26510.00	0.0279	0.0007
Dataset 3	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.5404	0.0045
Dataset 3	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	26510.00	0.00	26510.00	0.0270	0.0004
Dataset 3	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.5416	0.0041
Dataset 3	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	26510.00	0.00	26510.00	0.0279	0.0009
Dataset 3	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.5412	0.0032
Dataset 3	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	26510.00	0.00	26510.00	0.0273	0.0007
Dataset 3	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.5000	0.0236
Dataset 3	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	26510.00	0.00	26510.00	0.0253	0.0007
Dataset 3	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.4961	0.0167
Dataset 3	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	26510.00	0.00	26510.00	0.0249	0.0007
Dataset 3	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.4937	0.0029
Dataset 3	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	26510.00	0.00	26510.00	0.0260	0.0009
Dataset 3	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.5052	0.0206
Dataset 3	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	26510.00	0.00	26510.00	0.0298	0.0126
Dataset 3	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.5006	0.0169
Dataset 3	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	26510.00	0.00	26510.00	0.0257	0.0006
Dataset 3	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	26510.00	0.00	26510.00	0.4950	0.0069
Dataset 3	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	26510.00	0.00	26510.00	0.0249	0.0003
Dataset 4	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	25697.67	38.44	25710.00	0.5081	0.0280
Dataset 4	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	25690.00	40.00	25710.00	0.0281	0.0018
Dataset 4	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	25610.00	0.00	25610.00	0.5062	0.0157
Dataset 4	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	25613.33	17.95	25710.00	0.0253	0.0008
Dataset 4	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	25706.67	17.95	25710.00	0.5044	0.0073
Dataset 4	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	25703.33	24.94	25710.00	0.0274	0.0010
Dataset 4	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	25620.00	30.00	25710.00	0.5090	0.0164
Dataset 4	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	25616.67	24.94	25710.00	0.0255	0.0006
Dataset 4	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	25710.00	0.00	25710.00	0.5124	0.0358
Dataset 4	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	25706.67	17.95	25710.00	0.0283	0.0028
Dataset 4	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	25710.00	0.00	25710.00	0.5035	0.0018
Dataset 4	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	25670.00	48.99	25710.00	0.0331	0.0081
Dataset 4	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	25710.00	0.00	25710.00	0.5734	0.0186
Dataset 4	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	25706.67	17.95	25710.00	0.0305	0.0012
Dataset 4	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	25613.33	17.95	25710.00	0.5656	0.0075
Dataset 4	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	25620.00	30.00	25710.00	0.0286	0.0012
Dataset 4	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	25703.33	24.94	25710.00	0.5854	0.0579
Dataset 4	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	25700.00	30.00	25710.00	0.0312	0.0017
Dataset 4	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	25636.67	44.22	25710.00	0.5685	0.0063
Dataset 4	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	25616.67	24.94	25710.00	0.0288	0.0011
Dataset 4	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	25710.00	0.00	25710.00	0.5748	0.0088
Dataset 4	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	25710.00	0.00	25710.00	0.0341	0.0053
Dataset 4	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	25710.00	0.00	25710.00	0.5731	0.0068
Dataset 4	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	25670.00	48.99	25710.00	0.0357	0.0086
Dataset 4	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	25700.00	30.00	25710.00	0.5355	0.0098
Dataset 4	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	25700.00	30.00	25710.00	0.0288	0.0011
Dataset 4	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	25623.33	33.99	25710.00	0.5331	0.0113
Dataset 4	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	25613.33	17.95	25710.00	0.0267	0.0004
Dataset 4	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	25696.67	33.99	25710.00	0.5400	0.0133
Dataset 4	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	25706.67	17.95	25710.00	0.0296	0.0016
Dataset 4	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	25626.67	37.27	25710.00	0.5352	0.0072
Dataset 4	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	25616.67	24.94	25710.00	0.0266	0.0002
Dataset 4	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	25710.00	0.00	25710.00	0.5398	0.0081
Dataset 4	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	25703.33	24.94	25710.00	0.0307	0.0036
Dataset 4	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	25710.00	0.00	25710.00	0.5409	0.0072
Dataset 4	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	25673.33	48.19	25710.00	0.0350	0.0086
Dataset 5	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	29006.00	371.46	29230.00	0.5341	0.0113
Dataset 5	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	29022.00	509.77	29230.00	0.0303	0.0024
Dataset 5	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	28418.00	150.78	29230.00	0.5330	0.0074
Dataset 5	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	28390.00	0.00	28390.00	0.0268	0.0004
Dataset 5	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	29146.00	252.00	29230.00	0.5369	0.0092
Dataset 5	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	29078.33	430.02	29230.00	0.0300	0.0010
Dataset 5	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	28558.00	336.00	29230.00	0.5359	0.0093
Dataset 5	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	28390.00	0.00	28390.00	0.0268	0.0003
Dataset 5	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	29230.00	0.00	29230.00	0.5407	0.0074
Dataset 5	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	29230.00	0.00	29230.00	0.0329	0.0022
Dataset 5	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	29230.00	0.00	29230.00	0.5506	0.0248
Dataset 5	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	28894.00	411.51	29230.00	0.0349	0.0088
Dataset 5	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	29062.00	336.00	29230.00	0.5988	0.0072
Dataset 5	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	29034.00	355.28	29230.00	0.0341	0.0032
Dataset 5	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	28418.00	150.78	29230.00	0.5961	0.0061
Dataset 5	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	28390.00	0.00	28390.00	0.0300	0.0004
Dataset 5	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	29118.00	285.55	29230.00	0.6039	0.0196
Dataset 5	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	29146.00	252.00	29230.00	0.0338	0.0013
Dataset 5	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	28418.00	150.78	29230.00	0.6018	0.0104
Dataset 5	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	28390.00	0.00	28390.00	0.0303	0.0011
Dataset 5	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	29230.00	0.00	29230.00	0.6052	0.0065
Dataset 5	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	29230.00	0.00	29230.00	0.0343	0.0022
Dataset 5	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	29230.00	0.00	29230.00	0.6083	0.0117
Dataset 5	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	29006.00	371.46	29230.00	0.0406	0.0092
Dataset 5	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	29202.00	150.78	29230.00	0.5765	0.0064
Dataset 5	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	29022.00	572.10	29230.00	0.0316	0.0009
Dataset 5	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	28418.00	150.78	29230.00	0.5762	0.0083
Dataset 5	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	28418.00	150.78	29230.00	0.0289	0.0004
Dataset 5	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	29174.00	209.53	29230.00	0.5775	0.0069
Dataset 5	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	29118.00	285.55	29230.00	0.0327	0.0018
Dataset 5	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	28558.00	336.00	29230.00	0.5769	0.0082
Dataset 5	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	28390.00	0.00	28390.00	0.0292	0.0008
Dataset 5	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	29230.00	0.00	29230.00	0.5830	0.0066
Dataset 5	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	29230.00	0.00	29230.00	0.0326	0.0014
Dataset 5	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	29230.00	0.00	29230.00	0.5792	0.0046
Dataset 5	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	28894.00	411.51	29230.00	0.0368	0.0089
Dataset 6	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	34691.33	615.98	35170.00	0.5781	0.0045
Dataset 6	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	34712.33	639.80	35170.00	0.0337	0.0019
Dataset 6	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.5784	0.0044
Dataset 6	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	35170.00	0.00	35170.00	0.0296	0.0012
Dataset 6	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.5816	0.0044
Dataset 6	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	35011.00	383.26	35170.00	0.0357	0.0056
Dataset 6	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.5828	0.0068
Dataset 6	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	35170.00	0.00	35170.00	0.0292	0.0005
Dataset 6	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.5895	0.0076
Dataset 6	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	35170.00	0.00	35170.00	0.0346	0.0018
Dataset 6	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.5864	0.0045
Dataset 6	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	35170.00	0.00	35170.00	0.0297	0.0013
Dataset 6	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	35033.00	279.47	35170.00	0.6464	0.0101
Dataset 6	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	35042.00	326.34	35170.00	0.0370	0.0011
Dataset 6	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.6398	0.0032
Dataset 6	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	35170.00	0.00	35170.00	0.0323	0.0003
Dataset 6	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.6428	0.0028
Dataset 6	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	35096.00	227.39	35170.00	0.0385	0.0054
Dataset 6	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.6460	0.0088
Dataset 6	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	35170.00	0.00	35170.00	0.0325	0.0004
Dataset 6	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.6519	0.0080
Dataset 6	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	35170.00	0.00	35170.00	0.0394	0.0020
Dataset 6	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.6536	0.0099
Dataset 6	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	35170.00	0.00	35170.00	0.0329	0.0005
Dataset 6	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	35042.00	326.34	35170.00	0.6349	0.0064
Dataset 6	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	35053.00	302.82	35170.00	0.0366	0.0021
Dataset 6	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.6361	0.0080
Dataset 6	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	35170.00	0.00	35170.00	0.0318	0.0003
Dataset 6	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.6326	0.0020
Dataset 6	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	35106.00	239.47	35170.00	0.0369	0.0020
Dataset 6	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.6317	0.0023
Dataset 6	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	35170.00	0.00	35170.00	0.0319	0.0004
Dataset 6	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.6379	0.0036
Dataset 6	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	35170.00	0.00	35170.00	0.0380	0.0020
Dataset 6	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	35170.00	0.00	35170.00	0.6416	0.0097
Dataset 6	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	35170.00	0.00	35170.00	0.0323	0.0004
Dataset 7	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	42437.00	609.47	42880.00	0.6334	0.0071
Dataset 7	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	42346.33	664.33	42860.00	0.0376	0.0023
Dataset 7	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	42695.67	30.52	42860.00	0.6299	0.0062
Dataset 7	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	42695.67	30.52	42860.00	0.0322	0.0030
Dataset 7	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	42782.00	291.85	42860.00	0.6304	0.0046
Dataset 7	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	42707.00	284.20	42860.00	0.0419	0.0103
Dataset 7	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	42860.00	0.00	42860.00	0.6333	0.0143
Dataset 7	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	42752.33	81.92	42860.00	0.0367	0.0078
Dataset 7	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	42860.00	0.00	42860.00	0.6373	0.0024
Dataset 7	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	42860.00	0.00	42860.00	0.0427	0.0054
Dataset 7	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	42860.00	0.00	42860.00	0.6364	0.0019
Dataset 7	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	42860.00	0.00	42860.00	0.0391	0.0060
Dataset 7	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	42721.00	216.98	42880.00	0.6984	0.0037
Dataset 7	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	42401.00	701.72	42860.00	0.0415	0.0023
Dataset 7	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	42712.67	57.79	42860.00	0.6983	0.0036
Dataset 7	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	42695.67	30.52	42860.00	0.0355	0.0007
Dataset 7	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	42821.00	210.02	42860.00	0.7026	0.0084
Dataset 7	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	42728.67	217.80	42860.00	0.0427	0.0056
Dataset 7	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	42860.00	0.00	42860.00	0.7016	0.0055
Dataset 7	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	42752.33	81.92	42860.00	0.0413	0.0090
Dataset 7	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	42860.00	0.00	42860.00	0.7064	0.0035
Dataset 7	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	42860.00	0.00	42860.00	0.0488	0.0077
Dataset 7	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	42860.00	0.00	42860.00	0.7061	0.0032
Dataset 7	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	42860.00	0.00	42860.00	0.0437	0.0063
Dataset 7	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	42814.67	75.18	42860.00	0.7043	0.0069
Dataset 7	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	42820.33	71.90	42860.00	0.0412	0.0011
Dataset 7	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	42690.00	0.00	42690.00	0.7034	0.0038
Dataset 7	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	42690.00	0.00	42690.00	0.0353	0.0004
Dataset 7	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	42860.00	0.00	42860.00	0.7049	0.0031
Dataset 7	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	42809.33	88.35	42860.00	0.0423	0.0016
Dataset 7	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	42860.00	0.00	42860.00	0.7125	0.0133
Dataset 7	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	42724.00	68.00	42860.00	0.0375	0.0041
Dataset 7	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	42860.00	0.00	42860.00	0.7173	0.0127
Dataset 7	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	42860.00	0.00	42860.00	0.0482	0.0068
Dataset 7	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	42860.00	0.00	42860.00	0.7135	0.0031
Dataset 7	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	42860.00	0.00	42860.00	0.0405	0.0033
Dataset 8	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	49982.33	1106.13	51120.00	0.6630	0.0054
Dataset 8	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	50078.33	1068.15	51120.00	0.0414	0.0036
Dataset 8	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.6615	0.0029
Dataset 8	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	51120.00	0.00	51120.00	0.0337	0.0008
Dataset 8	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.6670	0.0037
Dataset 8	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	50854.33	540.32	51120.00	0.0452	0.0120
Dataset 8	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.6658	0.0040
Dataset 8	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	51120.00	0.00	51120.00	0.0339	0.0005
Dataset 8	Crossover: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.6754	0.0043
Dataset 8	Crossover: 0.05	 Inicialização: aleatoria	 Parada: convergencia	51120.00	0.00	51120.00	0.0463	0.0060
Dataset 8	Crossover: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.6718	0.0036
Dataset 8	Crossover: 0.05	 Inicialização: heuristica	 Parada: convergencia	51120.00	0.00	51120.00	0.0343	0.0008
Dataset 8	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	50571.67	815.26	51120.00	0.7326	0.0044
Dataset 8	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	50698.33	867.62	51120.00	0.0463	0.0068
Dataset 8	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.7316	0.0032
Dataset 8	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	51120.00	0.00	51120.00	0.0370	0.0005
Dataset 8	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.7360	0.0055
Dataset 8	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	50973.33	373.95	51120.00	0.0469	0.0039
Dataset 8	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.7405	0.0081
Dataset 8	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	51120.00	0.00	51120.00	0.0375	0.0010
Dataset 8	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.7405	0.0033
Dataset 8	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	51120.00	0.00	51120.00	0.0487	0.0043
Dataset 8	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.7449	0.0086
Dataset 8	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	51120.00	0.00	51120.00	0.0383	0.0024
Dataset 8	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	51045.00	322.44	51120.00	0.7476	0.0050
Dataset 8	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	50920.33	494.78	51120.00	0.0449	0.0033
Dataset 8	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.7459	0.0040
Dataset 8	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	51120.00	0.00	51120.00	0.0376	0.0006
Dataset 8	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.7484	0.0043
Dataset 8	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	51109.00	59.24	51120.00	0.0460	0.0034
Dataset 8	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.7435	0.0025
Dataset 8	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	51120.00	0.00	51120.00	0.0377	0.0005
Dataset 8	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.7551	0.0129
Dataset 8	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	51120.00	0.00	51120.00	0.0480	0.0025
Dataset 8	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	51120.00	0.00	51120.00	0.7551	0.0048
Dataset 8	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	51120.00	0.00	51120.00	0.0384	0.0007
Dataset 9	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	50426.33	754.44	51570.00	0.7419	0.0033
Dataset 9	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	50302.00	848.48	51570.00	0.0484	0.0070
Dataset 9	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	51570.00	0.00	51570.00	0.7447	0.0084
Dataset 9	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	51570.00	0.00	51570.00	0.0378	0.0009
Dataset 9	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	51271.33	457.94	51570.00	0.7446	0.0044
Dataset 9	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	50882.00	708.81	51570.00	0.0564	0.0122
Dataset 9	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	51570.00	0.00	51570.00	0.7452	0.0047
Dataset 9	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	51570.00	0.00	51570.00	0.0380	0.0005
Dataset 9	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	51570.00	0.00	51570.00	0.7522	0.0042
Dataset 9	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	51570.00	0.00	51570.00	0.0582	0.0090
Dataset 9	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	51570.00	0.00	51570.00	0.7574	0.0134
Dataset 9	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	51570.00	0.00	51570.00	0.0384	0.0006
Dataset 9	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	50755.00	947.36	51570.00	0.8132	0.0063
Dataset 9	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	50698.67	892.26	51570.00	0.0515	0.0047
Dataset 9	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	51570.00	0.00	51570.00	0.8145	0.0039
Dataset 9	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	51570.00	0.00	51570.00	0.0415	0.0006
Dataset 9	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	51206.67	479.19	51570.00	0.8141	0.0042
Dataset 9	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	51096.67	513.75	51570.00	0.0617	0.0169
Dataset 9	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	51570.00	0.00	51570.00	0.8169	0.0052
Dataset 9	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	51570.00	0.00	51570.00	0.0413	0.0005
Dataset 9	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	51570.00	0.00	51570.00	0.8227	0.0077
Dataset 9	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	51537.67	174.12	51570.00	0.0652	0.0121
Dataset 9	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	51570.00	0.00	51570.00	0.8220	0.0040
Dataset 9	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	51570.00	0.00	51570.00	0.0423	0.0020
Dataset 9	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	51301.33	489.03	51570.00	0.8532	0.0054
Dataset 9	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	51149.33	532.34	51570.00	0.0533	0.0028
Dataset 9	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	51570.00	0.00	51570.00	0.8525	0.0041
Dataset 9	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	51570.00	0.00	51570.00	0.0429	0.0004
Dataset 9	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	51473.00	291.00	51570.00	0.8548	0.0048
Dataset 9	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	51494.67	213.21	51570.00	0.0544	0.0036
Dataset 9	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	51570.00	0.00	51570.00	0.8563	0.0075
Dataset 9	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	51570.00	0.00	51570.00	0.0435	0.0008
Dataset 9	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	51570.00	0.00	51570.00	0.8621	0.0036
Dataset 9	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	51570.00	0.00	51570.00	0.0598	0.0061
Dataset 9	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	51570.00	0.00	51570.00	0.8609	0.0034
Dataset 9	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	51570.00	0.00	51570.00	0.0435	0.0005
Dataset 10	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	35938.67	1233.29	38210.00	0.8164	0.0039
Dataset 10	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	34775.67	1250.80	37670.00	0.0562	0.0094
Dataset 10	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	38078.67	318.78	38210.00	0.8229	0.0042
Dataset 10	Crossover: um_ponto	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	38210.00	0.00	38210.00	0.0487	0.0067
Dataset 10	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	37377.33	256.77	37670.00	0.8228	0.0041
Dataset 10	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	37265.33	501.50	38210.00	0.0716	0.0217
Dataset 10	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	37670.00	0.00	37670.00	0.8213	0.0034
Dataset 10	Crossover: um_ponto	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	37952.00	314.42	38210.00	0.0529	0.0121
Dataset 10	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	37670.00	0.00	37670.00	0.8317	0.0081
Dataset 10	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	37460.00	327.05	38210.00	0.0915	0.0225
Dataset 10	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	37670.00	0.00	37670.00	0.8271	0.0040
Dataset 10	Crossover: um_ponto	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	37825.33	253.02	38210.00	0.0598	0.0163
Dataset 10	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	35965.67	951.24	37670.00	0.8932	0.0084
Dataset 10	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	35756.33	1009.26	37170.00	0.0582	0.0064
Dataset 10	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	38095.33	230.30	38210.00	0.8932	0.0048
Dataset 10	Crossover: dois_pontos	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	38149.33	246.28	38210.00	0.0555	0.0089
Dataset 10	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	37407.00	265.65	37670.00	0.8933	0.0048
Dataset 10	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	37346.67	347.13	37670.00	0.0805	0.0281
Dataset 10	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	37670.00	0.00	37670.00	0.8922	0.0040
Dataset 10	Crossover: dois_pontos	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	37858.00	269.40	38210.00	0.0668	0.0189
Dataset 10	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	37670.00	0.00	37670.00	0.9013	0.0037
Dataset 10	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	37621.33	334.61	38210.00	0.1016	0.0300
Dataset 10	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	37670.00	0.00	37670.00	0.8983	0.0034
Dataset 10	Crossover: dois_pontos	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	37782.67	238.07	38210.00	0.0597	0.0106
Dataset 10	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: fixa_geracoes	37010.33	715.96	38210.00	0.9483	0.0047
Dataset 10	Crossover: uniforme	 Mutação: 0.001	 Inicialização: aleatoria	 Parada: convergencia	36976.67	596.58	38210.00	0.0637	0.0031
Dataset 10	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: fixa_geracoes	38059.33	250.85	38210.00	0.9535	0.0053
Dataset 10	Crossover: uniforme	 Mutação: 0.001	 Inicialização: heuristica	 Parada: convergencia	38210.00	0.00	38210.00	0.0573	0.0070
Dataset 10	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: fixa_geracoes	37512.00	242.89	37670.00	0.9623	0.0100
Dataset 10	Crossover: uniforme	 Mutação: 0.01	 Inicialização: aleatoria	 Parada: convergencia	37505.67	377.41	38210.00	0.0713	0.0092
Dataset 10	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: fixa_geracoes	37670.00	0.00	37670.00	0.9588	0.0076
Dataset 10	Crossover: uniforme	 Mutação: 0.01	 Inicialização: heuristica	 Parada: convergencia	37861.33	266.42	38210.00	0.0664	0.0160
Dataset 10	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: fixa_geracoes	37670.00	0.00	37670.00	0.9607	0.0037
Dataset 10	Crossover: uniforme	 Mutação: 0.05	 Inicialização: aleatoria	 Parada: convergencia	37424.67	298.66	38210.00	0.1035	0.0197
Dataset 10	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: fixa_geracoes	37670.00	0.00	37670.00	0.9605	0.0048
Dataset 10	Crossover: uniforme	 Mutação: 0.05	 Inicialização: heuristica	 Parada: convergencia	37800.67	249.04	38210.00	0.0661	0.0163
"""

def parse_data(raw_data):
    lines = raw_data.strip().split('\n')
    parsed_data = []

    for line in lines:
        # Define regex patterns for each field.
        # This makes parsing robust to variations in internal spacing/tabbing.
        dataset_match = re.search(r'Dataset (\d+)', line)
        crossover_match = re.search(r'Crossover: ([a-zA-Z_.]+)', line) # Catches 'um_ponto', 'dois_pontos', 'uniforme', or '0.05'
        mutacao_match = re.search(r'Mutação: ([0-9.]+)', line)
        inicializacao_match = re.search(r'Inicialização: ([a-zA-Z]+)', line)
        parada_match = re.search(r'Parada: ([a-zA-Z_]+)', line)
        
        # Numerical values are trickier if there's no clear label or consistent leading/trailing pattern.
        # However, they consistently appear after 'Parada: X' and before end of line.
        # Let's extract the known parts and then take the rest as numerical values.

        # First, extract known categorical parts
        dataset = f'Dataset {dataset_match.group(1)}' if dataset_match else 'Unknown Dataset'
        crossover = crossover_match.group(1) if crossover_match else 'Unknown Crossover'
        inicializacao = inicializacao_match.group(1) if inicializacao_match else 'Unknown Inicialização'
        parada = parada_match.group(1) if parada_match else 'Unknown Parada'
        mutacao = float(mutacao_match.group(1)) if mutacao_match else 0.0 # Default to 0.0 if not found

        # Now, extract the numerical values from the end of the line.
        # This regex looks for 5 floating-point numbers at the end of the line, potentially separated by tabs/spaces
        numerical_values_match = re.search(r'([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)$', line)
        
        if numerical_values_match:
            mean_apt = float(numerical_values_match.group(1))
            dp_apt = float(numerical_values_match.group(2))
            best_apt = float(numerical_values_match.group(3))
            mean_time = float(numerical_values_match.group(4))
            dp_time = float(numerical_values_match.group(5))
        else:
            # Fallback for numerical values if regex fails (shouldn't happen with provided data)
            # This indicates a very unexpected format.
            print(f"Warning: Could not parse numerical values from line: {line}")
            mean_apt, dp_apt, best_apt, mean_time, dp_time = 0.0, 0.0, 0.0, 0.0, 0.0
            
        parsed_data.append({
            'Dataset': dataset,
            'Crossover': crossover,
            'Mutação': mutacao,
            'Inicialização': inicializacao,
            'Parada': parada,
            'Média Aptidão': mean_apt,
            'DP Aptidão': dp_apt,
            'Melhor Aptidão': best_apt,
            'Média Tempo (s)': mean_time,
            'DP Tempo (s)': dp_time
        })
    return pd.DataFrame(parsed_data)

df = parse_data(data)

# --- Gráfico 1: Impacto do Tipo de Crossover ---
crossover_agg = df.groupby('Crossover')[['Média Aptidão', 'Média Tempo (s)']].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x='Crossover', y='Média Aptidão', data=crossover_agg, ax=axes[0], palette='viridis')
axes[0].set_title('Impacto do Crossover na Média Aptidão (Todos os Datasets)')
axes[0].set_ylabel('Média Aptidão (Valor Total)')
axes[0].set_xlabel('Tipo de Crossover')
axes[0].ticklabel_format(style='plain', axis='y') # Avoid scientific notation

sns.barplot(x='Crossover', y='Média Tempo (s)', data=crossover_agg, ax=axes[1], palette='plasma')
axes[1].set_title('Impacto do Crossover no Tempo de Execução (Todos os Datasets)')
axes[1].set_ylabel('Média Tempo (s)')
axes[1].set_xlabel('Tipo de Crossover')
axes[1].ticklabel_format(style='plain', axis='y') # Avoid scientific notation

plt.tight_layout()
plt.savefig('grafico_crossover.png', dpi=300)
plt.close()


# --- Gráfico 2: Impacto da Taxa de Mutação ---
mutacao_agg = df.groupby('Mutação')[['Média Aptidão', 'Média Tempo (s)']].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x='Mutação', y='Média Aptidão', data=mutacao_agg, ax=axes[0], palette='viridis')
axes[0].set_title('Impacto da Taxa de Mutação na Média Aptidão (Todos os Datasets)')
axes[0].set_ylabel('Média Aptidão (Valor Total)')
axes[0].set_xlabel('Taxa de Mutação')
axes[0].ticklabel_format(style='plain', axis='y')

sns.barplot(x='Mutação', y='Média Tempo (s)', data=mutacao_agg, ax=axes[1], palette='plasma')
axes[1].set_title('Impacto da Taxa de Mutação no Tempo de Execução (Todos os Datasets)')
axes[1].set_ylabel('Média Tempo (s)')
axes[1].set_xlabel('Taxa de Mutação')
axes[1].ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig('grafico_mutacao.png', dpi=300)
plt.close()

# --- Gráfico 3: Impacto da Inicialização da População ---
inicializacao_agg = df.groupby('Inicialização')[['Média Aptidão', 'Média Tempo (s)']].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x='Inicialização', y='Média Aptidão', data=inicializacao_agg, ax=axes[0], palette='viridis')
axes[0].set_title('Impacto da Inicialização na Média Aptidão (Todos os Datasets)')
axes[0].set_ylabel('Média Aptidão (Valor Total)')
axes[0].set_xlabel('Estratégia de Inicialização')
axes[0].ticklabel_format(style='plain', axis='y')

sns.barplot(x='Inicialização', y='Média Tempo (s)', data=inicializacao_agg, ax=axes[1], palette='plasma')
axes[1].set_title('Impacto da Inicialização no Tempo de Execução (Todos os Datasets)')
axes[1].set_ylabel('Média Tempo (s)')
axes[1].set_xlabel('Estratégia de Inicialização')
axes[1].ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig('grafico_inicializacao.png', dpi=300)
plt.close()

# --- Gráfico 4: Impacto do Critério de Parada ---
parada_agg = df.groupby('Parada')[['Média Aptidão', 'Média Tempo (s)']].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x='Parada', y='Média Aptidão', data=parada_agg, ax=axes[0], palette='viridis')
axes[0].set_title('Impacto do Critério de Parada na Média Aptidão (Todos os Datasets)')
axes[0].set_ylabel('Média Aptidão (Valor Total)')
axes[0].set_xlabel('Critério de Parada')
axes[0].ticklabel_format(style='plain', axis='y')

sns.barplot(x='Parada', y='Média Tempo (s)', data=parada_agg, ax=axes[1], palette='plasma')
axes[1].set_title('Impacto do Critério de Parada no Tempo de Execução (Todos os Datasets)')
axes[1].set_ylabel('Média Tempo (s)')
axes[1].set_xlabel('Critério de Parada')
axes[1].ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig('grafico_parada.png', dpi=300)
plt.close()