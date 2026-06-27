# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 02:51:03 2026

@author: Cacob
"""

# Fiz o import das bases de dados e verifiquei a possibilidade de seguir o projeto.
# Aparentemente dá, mas teremos que mudar um pouco o foco, posso explicar melhor no zap


# =============================
# IMPORTANDO AS BASES DE DADOS 
# =============================


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


df_markets_pequeno = df_markets.head(50).copy() # criando cópias pequenas dos datasets para que a gente possa trabalhar as operações sem gastar muito do pc, ai no final, convertemos tudo para o dataset original
df_events_pequeno = df_events.head(50).copy()

df_markets_pequeno

#df_markets_pequeno.to_csv('mini_polymarket.csv', index = False, encoding = 'utf-8')
#df_events_pequeno.to_csv('mini_events.csv', index = False, encoding = 'utf-8')


# ============================================================
# Começando análise para ver oq dá para fazer com esses dados
# ============================================================

df_columns = [col for col in df_markets_pequeno.columns] # só checando pra ver quais colunas tem no meu df, pra ver se o gpt não tá falando merda.

# o mais índice mais próximo de nos indicar o que o mercado mais acreditava é o outcomePrice, o valor dele é uma string com o primeiro valor contendo Yes, o outro sendo o NO, a formatação dessa coluna ta meio do mal, então para conseguir trabalhar de forma numéricas com essa coluna, irei trata-las para trasformar de str para float.

df_markets_pequeno["preco_yes"] = (
    df_markets_pequeno["outcomePrices"]
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.replace('"', "", regex=False)
    .str.split(",")
    .str[0]
    .astype(float)
)

# transformei de ['0.45', '0.55'] para 0.45. <- proporção de pessoas que acham que sim

df_markets_pequeno["preco_no"] = (
    df_markets_pequeno["outcomePrices"]
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.replace('"', "", regex=False)
    .str.split(",")
    .str[1]
    .astype(float)
    
) # transformei de ['0.45', '0.55'] para 0.55. <- proporção de pessoas que acham que não

eventos_prob = df_markets_pequeno.dropna(subset=["preco_yes"]).copy()

# Incerteza binária:
# quanto mais perto de 0.5, maior a incerteza
eventos_prob["incerteza_binaria"] = 1 - abs(eventos_prob["preco_yes"] - 0.5) * 2

# Zona de dúvida:
# True se a probabilidade estiver entre 25% e 75%
eventos_prob["zona_duvida"] = eventos_prob["preco_yes"].between(0.25, 0.75)

# Peso pelo volume dentro de cada evento
eventos_prob["peso_volume"] = (
    eventos_prob["volume"] /
    eventos_prob.groupby("event_title")["volume"].transform("sum")
)

# Incerteza ponderada pelo volume
eventos_prob["incerteza_ponderada"] = (eventos_prob["incerteza_binaria"] * eventos_prob["peso_volume"])

# usando loop for só para que o python não fique calculando números bizarramente pequenos para incerteza.

for incerteza in eventos_prob["incerteza_ponderada"]:
    if incerteza <= 0.0001:
        incerteza = 0

incerteza_eventos = eventos_prob.groupby("event_title").agg(
    n_opcoes=("preco_yes", "count"),
    soma_prob=("preco_yes", "sum"),
    maior_prob=("preco_yes", "max"),
    media_prob=("preco_yes", "mean"),
    desvio_prob=("preco_yes", "std"),
    opcoes_em_duvida=("zona_duvida", "sum"),
    proporcao_duvida=("zona_duvida", "mean"),
    incerteza_media=("incerteza_binaria", "mean"),
    incerteza_ponderada=("incerteza_ponderada", "sum"),
    volume_total=("volume", "sum"),
    liquidez_total=("liquidity", "sum")
)

eventos_favoritos = eventos_prob.groupby("event_title")["preco_yes"].idxmax()

favoritos = eventos_prob.loc[
    eventos_favoritos,
    ["event_title", "question", "preco_yes"]
].rename(columns={
    "question": "opcao_favorita",
    "preco_yes": "prob_favorita"
})

incerteza_eventos = incerteza_eventos.reset_index()

analise_incerteza = incerteza_eventos.merge(
    favoritos,
    on="event_title",
    how="left"
)

analise_incerteza # <- base de dados que me trás incerteza sobre cada evento do polymarket






