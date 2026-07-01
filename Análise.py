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

import numpy as np 
from pathlib import Path
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

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


df_markets_pequeno = df_markets.copy() # criando cópias pequenas dos datasets para que a gente possa trabalhar as operações sem gastar muito do pc, ai no final, convertemos tudo para o dataset original
df_events_pequeno = df_events.copy()

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

eventos_prob.loc[eventos_prob["incerteza_ponderada"] <= 0.0001,"incerteza_ponderada"] = 0

incerteza_eventos = eventos_prob.groupby("event_title").agg(
    n_opcoes=("preco_yes", "count"), # número de opções de cada aposta
    soma_prob=("preco_yes", "sum"), # soma das probabilidades que cada evento
    maior_prob=("preco_yes", "max"), # maior prob
    media_prob=("preco_yes", "mean"), # média
    desvio_prob=("preco_yes", "std"), # desvio
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


# ===========================================================
# USANDO ÍNDICE DE BRIER PARA MEDIR ASSERTIVIDADE DO MERCADO
# ===========================================================

df_brier = df_markets_pequeno.copy()

# Resultado final inferido:
# 1 = Yes aconteceu
# 0 = No aconteceu
# NaN = mercado não encerrado ou resultado não claro

df_brier["resultado_yes"] = float("nan")

resultado_yes = ((df_brier["closed"] == True) &(df_brier["preco_yes"] >= 0.99))

resultado_no = ((df_brier["closed"] == True) &(df_brier["preco_yes"] <= 0.01))

df_brier.loc[resultado_yes, "resultado_yes"] = 1
df_brier.loc[resultado_no, "resultado_yes"] = 0

# Criando também o resultado No, para não esconder essa informação
df_brier["resultado_no"] = 1 - df_brier["resultado_yes"]

df_brier["resultado_final"] = df_brier["resultado_yes"].map({
    1: "Yes",
    0: "No"
})


# Probabilidades passadas do Yes
# Atenção: isso é reconstruído a partir das colunas de mudança de preço

df_brier["p_yes_1dia"] = (df_brier["preco_yes"] - df_brier["oneDayPriceChange"])

df_brier["p_yes_1semana"] = (df_brier["preco_yes"] - df_brier["oneWeekPriceChange"])

df_brier["p_yes_1mes"] = (df_brier["preco_yes"] - df_brier["oneMonthPriceChange"])

# Remove probabilidades inválidas

for coluna in ["p_yes_1dia", "p_yes_1semana", "p_yes_1mes"]:
    df_brier.loc[~df_brier[coluna].between(0, 1), coluna] = float("nan")


# Probabilidades passadas 

df_brier["p_no_1dia"] = 1 - df_brier["p_yes_1dia"]
df_brier["p_no_1semana"] = 1 - df_brier["p_yes_1semana"]
df_brier["p_no_1mes"] = 1 - df_brier["p_yes_1mes"]


# Brier Score clássico, usando a probabilidade do Yes
# Isso já incorpora o No, porque resultado_yes = 0 quando No aconteceu

df_brier["brier_1dia"] = (df_brier["p_yes_1dia"] - df_brier["resultado_yes"]) ** 2

df_brier["brier_1semana"] = ( df_brier["p_yes_1semana"] - df_brier["resultado_yes"]) ** 2

df_brier["brier_1mes"] = (df_brier["p_yes_1mes"] - df_brier["resultado_yes"]) ** 2


# Probabilidade atribuída ao resultado que realmente aconteceu
# Se o resultado foi Yes, usa p_yes.
# Se o resultado foi No, usa p_no.

df_brier["prob_resultado_1dia"] = float("nan")
df_brier["prob_resultado_1semana"] = float("nan")
df_brier["prob_resultado_1mes"] = float("nan")

df_brier.loc[df_brier["resultado_yes"] == 1, "prob_resultado_1dia"] = df_brier["p_yes_1dia"]
df_brier.loc[df_brier["resultado_yes"] == 1, "prob_resultado_1semana"] = df_brier["p_yes_1semana"]
df_brier.loc[df_brier["resultado_yes"] == 1, "prob_resultado_1mes"] = df_brier["p_yes_1mes"]

df_brier.loc[df_brier["resultado_yes"] == 0, "prob_resultado_1dia"] = df_brier["p_no_1dia"]
df_brier.loc[df_brier["resultado_yes"] == 0, "prob_resultado_1semana"] = df_brier["p_no_1semana"]
df_brier.loc[df_brier["resultado_yes"] == 0, "prob_resultado_1mes"] = df_brier["p_no_1mes"]


# Tabela principal: Brier por subevento

analise_brier_subeventos = df_brier.dropna(subset=[
    "resultado_yes",
    "p_yes_1mes",
    "p_yes_1semana",
    "p_yes_1dia",
    "brier_1mes",
    "brier_1semana",
    "brier_1dia"
]).copy()

analise_brier_subeventos["melhora_1mes_para_1dia"] = (analise_brier_subeventos["brier_1mes"] -analise_brier_subeventos["brier_1dia"])

analise_brier_subeventos = analise_brier_subeventos[[
    "event_id",
    "event_title",
    "question",
    "resultado_final",
    "resultado_yes",
    "resultado_no",
    "preco_yes",
    "preco_no",
    "p_yes_1mes",
    "p_no_1mes",
    "p_yes_1semana",
    "p_no_1semana",
    "p_yes_1dia",
    "p_no_1dia",
    "prob_resultado_1mes",
    "prob_resultado_1semana",
    "prob_resultado_1dia",
    "brier_1mes",
    "brier_1semana",
    "brier_1dia",
    "melhora_1mes_para_1dia",
    "volume",
    "liquidity"
]]

analise_brier_subeventos

# ===========================================================
# RESUMO DO BRIER POR EVENTO
# ===========================================================

brier_por_evento = analise_brier_subeventos.groupby(
    ["event_id", "event_title"],as_index=False).agg(
    mercados=("question", "count"),
    brier_1mes=("brier_1mes", "mean"),
    brier_1semana=("brier_1semana", "mean"),
    brier_1dia=("brier_1dia", "mean"),
    prob_resultado_1mes=("prob_resultado_1mes", "mean"),
    prob_resultado_1semana=("prob_resultado_1semana", "mean"),
    prob_resultado_1dia=("prob_resultado_1dia", "mean"),
    melhora_media=("melhora_1mes_para_1dia", "mean"),
    volume_total=("volume", "sum"),
    liquidez_total=("liquidity", "sum")
)

brier_por_evento

# ===========================================================
# CORRELAÇÃO ENTRE VOLUME E PRECISÃO DO MERCADO
# ===========================================================


analise_brier_subeventos["log_volume"] = np.log1p(analise_brier_subeventos["volume"])

analise_brier_subeventos["log_liquidity"] = np.log1p(analise_brier_subeventos["liquidity"])


# Como muitos mercados apresentavam erro igual a zero, possivelmente por acerto perfeito ou por preços já próximos da resolução no momento da coleta, foi realizada uma análise complementar excluindo esses casos. Nessa amostra restrita a mercados com erro positivo, a correlação entre volume e erro preditivo tornou-se fraca e negativa, indicando que, entre os mercados que apresentaram algum erro, maior volume não esteve associado a maior erro. Pelo contrário, houve leve associação com menor erro, embora de baixa magnitude.

analise_brier_subeventos = analise_brier_subeventos[analise_brier_subeventos['brier_1mes'] > 0.001]

correlacao_subeventos = analise_brier_subeventos[[
    "log_volume",
    "log_liquidity",
    "brier_1mes",
    "brier_1semana",
    "brier_1dia",
    "prob_resultado_1mes",
    "prob_resultado_1semana",
    "prob_resultado_1dia"
]].corr(method="spearman")

correlacao_subeventos

# ===========================================================
# Criando apenas os nomes adicionais necessários para os plots
# ===========================================================

analise_brier_subeventos["id_evento"] = analise_brier_subeventos["event_id"]

analise_brier_subeventos["evento"] = analise_brier_subeventos["event_title"]

analise_brier_subeventos["subevento_pergunta"] = analise_brier_subeventos["question"]

analise_brier_subeventos["volume_submercado"] = analise_brier_subeventos["volume"]

analise_brier_subeventos["liquidez_submercado"] = analise_brier_subeventos["liquidity"]

analise_brier_subeventos["log_volume_submercado"] = np.log1p(analise_brier_subeventos["volume_submercado"])

analise_brier_subeventos["log_liquidez_submercado"] = np.log1p(analise_brier_subeventos["liquidez_submercado"])

analise_brier_subeventos["erro_brier_1mes"] = analise_brier_subeventos["brier_1mes"]

analise_brier_subeventos["erro_brier_1semana"] = analise_brier_subeventos["brier_1semana"]

analise_brier_subeventos["erro_brier_1dia"] = analise_brier_subeventos["brier_1dia"]

analise_brier_subeventos["erro_prob_resultado_1mes"] = (1 - analise_brier_subeventos["prob_resultado_1mes"])

analise_brier_subeventos["erro_prob_resultado_1semana"] = (1 - analise_brier_subeventos["prob_resultado_1semana"])

analise_brier_subeventos["erro_prob_resultado_1dia"] = (1 - analise_brier_subeventos["prob_resultado_1dia"])

# Isso mandei o chat criar, pois eu estava tendo erro pra caralho com "Invalid KeyValue"


# =========================================
# PLOT COM ERRO ABSOLUTO POR HORIZONTE
# =========================================

# Erro absoluto:
# quanto menor, melhor.
# 0 = mercado deu 100% de probabilidade ao resultado correto
# 1 = mercado deu 0% de probabilidade ao resultado correto

analise_brier_subeventos["erro_abs_1mes"] = (
    1 - analise_brier_subeventos["prob_resultado_1mes"]
)

analise_brier_subeventos["erro_abs_1semana"] = (
    1 - analise_brier_subeventos["prob_resultado_1semana"]
)

analise_brier_subeventos["erro_abs_1dia"] = (
    1 - analise_brier_subeventos["prob_resultado_1dia"]
)

# Garantindo que o log do volume existe
if "log_volume_submercado" not in analise_brier_subeventos.columns:
    analise_brier_subeventos["log_volume_submercado"] = np.log1p(
        analise_brier_subeventos["volume"]
    )

base_plot = analise_brier_subeventos.dropna(subset=[
    "log_volume_submercado",
    "erro_abs_1mes",
    "erro_abs_1semana",
    "erro_abs_1dia"
]).copy()

# Transformando para formato longo
base_plot_longo = base_plot.melt(
    id_vars=["log_volume_submercado"],
    value_vars=[
        "erro_abs_1mes",
        "erro_abs_1semana",
        "erro_abs_1dia"
    ],
    var_name="horizonte_tempo",
    value_name="erro_absoluto"
)

# Renomeando os horizontes
base_plot_longo["horizonte_tempo"] = base_plot_longo["horizonte_tempo"].replace({
    "erro_abs_1mes": "1 mês antes",
    "erro_abs_1semana": "1 semana antes",
    "erro_abs_1dia": "1 dia antes"
})

base_plot_longo = base_plot_longo[
    base_plot_longo["erro_absoluto"] >= 0
].copy()

# Selecionando até 10000 pontos de cada horizonte
base_plot_longo = (
    base_plot_longo
    .groupby("horizonte_tempo", group_keys=False)
    .apply(lambda grupo: grupo.sample(n=min(10000, len(grupo)), random_state=42))
    .reset_index(drop=True)
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=base_plot_longo,
    x="log_volume_submercado",
    y="erro_absoluto",
    hue="horizonte_tempo",
    alpha=0.35
)

for horizonte in ["1 mês antes", "1 semana antes", "1 dia antes"]:
    
    temp = base_plot_longo[base_plot_longo["horizonte_tempo"] == horizonte]
    
    sns.regplot(
        data=temp,
        x="log_volume_submercado",
        y="erro_absoluto",
        scatter=False,
        lowess=True
    )

plt.title("Volume do submercado versus erro absoluto em cada t instante")
plt.xlabel("Log do volume do submercado")
plt.ylabel("Erro absoluto")
plt.show()