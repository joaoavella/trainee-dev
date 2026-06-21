# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 02:51:03 2026

@author: Cacob
"""

# Fiz o import das bases de dados e verifiquei a possibilidade de seguir o projeto.
# Aparentemente dá, mas teremos que mudar um pouco o foco, posso explicar melhor no zap

from pathlib import Path
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

try:
    pasta_base = Path(__file__).parent  # se for arquivo python
except NameError:
    pasta_base = Path.cwd() # se for arquivo jupyter

os.environ["KAGGLEHUB_CACHE"] = str(pasta_base / "dados")

dataset = "ismetsemedov/polymarket-prediction-markets"

caminho = Path(kagglehub.dataset_download(dataset))

file_path_markets = "polymarket_markets.csv"
file_path_events = "polymarket_events.csv"

df_markets = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    dataset,
    file_path_markets
)

df_events = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    dataset,
    file_path_events
)

df_markets_pequeno = df_markets.head(20)
df_events_pequeno = df_events.head(20)

df_markets_pequeno

df_markets_pequeno.to_csv('mini_polymarket.csv', index = False, encoding = 'utf-8')
