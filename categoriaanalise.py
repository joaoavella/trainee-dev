# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 02:51:03 2026

@author: Cacob
"""

# =============================
# IMPORTANDO AS BASES DE DADOS 
# =============================

import numpy as np 
import pandas as pd
from pathlib import Path
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
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

# Criando as cópias completas para processar toda a base de dados
df_markets_pequeno = df_markets.copy() 
df_events_pequeno = df_events.copy()

# ============================================================
# Tratamento inicial dos dados de mercados
# ============================================================

df_markets_pequeno["preco_yes"] = (
    df_markets_pequeno["outcomePrices"]
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.replace('"', "", regex=False)
    .str.split(",")
    .str[0]
    .astype(float)
)

df_markets_pequeno["preco_no"] = (
    df_markets_pequeno["outcomePrices"]
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.replace('"', "", regex=False)
    .str.split(",")
    .str[1]
    .astype(float)
)

eventos_prob = df_markets_pequeno.dropna(subset=["preco_yes"]).copy()

# Incerteza binária
eventos_prob["incerteza_binaria"] = 1 - abs(eventos_prob["preco_yes"] - 0.5) * 2

# Zona de dúvida
eventos_prob["zona_duvida"] = eventos_prob["preco_yes"].between(0.25, 0.75)

# Peso pelo volume dentro de cada evento
eventos_prob["peso_volume"] = (
    eventos_prob["volume"] /
    eventos_prob.groupby("event_title")["volume"].transform("sum")
)

# Incerteza ponderada pelo volume
eventos_prob["incerteza_ponderada"] = (eventos_prob["incerteza_binaria"] * eventos_prob["peso_volume"])
eventos_prob.loc[eventos_prob["incerteza_ponderada"] <= 0.0001,"incerteza_ponderada"] = 0

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
).reset_index()

eventos_favoritos = eventos_prob.groupby("event_title")["preco_yes"].idxmax()

favoritos = eventos_prob.loc[
    eventos_favoritos,
    ["event_title", "question", "preco_yes"]
].rename(columns={
    "question": "opcao_favorita",
    "preco_yes": "prob_favorita"
})

analise_incerteza = incerteza_eventos.merge(favoritos, on="event_title", how="left")


# ===========================================================
# USANDO ÍNDICE DE BRIER PARA MEDIR ASSERTIVIDADE DO MERCADO
# ===========================================================

df_brier = df_markets_pequeno.copy()

df_brier["resultado_yes"] = float("nan")
resultado_yes = ((df_brier["closed"] == True) & (df_brier["preco_yes"] >= 0.99))
resultado_no = ((df_brier["closed"] == True) & (df_brier["preco_yes"] <= 0.01))

df_brier.loc[resultado_yes, "resultado_yes"] = 1
df_brier.loc[resultado_no, "resultado_yes"] = 0

df_brier["resultado_no"] = 1 - df_brier["resultado_yes"]
df_brier["resultado_final"] = df_brier["resultado_yes"].map({1: "Yes", 0: "No"})

df_brier["p_yes_1dia"] = (df_brier["preco_yes"] - df_brier["oneDayPriceChange"])
df_brier["p_yes_1semana"] = (df_brier["preco_yes"] - df_brier["oneWeekPriceChange"])
df_brier["p_yes_1mes"] = (df_brier["preco_yes"] - df_brier["oneMonthPriceChange"])

for coluna in ["p_yes_1dia", "p_yes_1semana", "p_yes_1mes"]:
    df_brier.loc[~df_brier[coluna].between(0, 1), coluna] = float("nan")

df_brier["p_no_1dia"] = 1 - df_brier["p_yes_1dia"]
df_brier["p_no_1semana"] = 1 - df_brier["p_yes_1semana"]
df_brier["p_no_1mes"] = 1 - df_brier["p_yes_1mes"]

df_brier["brier_1dia"] = (df_brier["p_yes_1dia"] - df_brier["resultado_yes"]) ** 2
df_brier["brier_1semana"] = (df_brier["p_yes_1semana"] - df_brier["resultado_yes"]) ** 2
df_brier["brier_1mes"] = (df_brier["p_yes_1mes"] - df_brier["resultado_yes"]) ** 2

df_brier["prob_resultado_1dia"] = float("nan")
df_brier["prob_resultado_1semana"] = float("nan")
df_brier["prob_resultado_1mes"] = float("nan")

df_brier.loc[df_brier["resultado_yes"] == 1, "prob_resultado_1dia"] = df_brier["p_yes_1dia"]
df_brier.loc[df_brier["resultado_yes"] == 1, "prob_resultado_1semana"] = df_brier["p_yes_1semana"]
df_brier.loc[df_brier["resultado_yes"] == 1, "prob_resultado_1mes"] = df_brier["p_yes_1mes"]

df_brier.loc[df_brier["resultado_yes"] == 0, "prob_resultado_1dia"] = df_brier["p_no_1dia"]
df_brier.loc[df_brier["resultado_yes"] == 0, "prob_resultado_1semana"] = df_brier["p_no_1semana"]
df_brier.loc[df_brier["resultado_yes"] == 0, "prob_resultado_1mes"] = df_brier["p_no_1mes"]

analise_brier_subeventos = df_brier.dropna(subset=[
    "resultado_yes", "p_yes_1mes", "p_yes_1semana", "p_yes_1dia",
    "brier_1mes", "brier_1semana", "brier_1dia"
]).copy()

analise_brier_subeventos["melhora_1mes_para_1dia"] = (analise_brier_subeventos["brier_1mes"] - analise_brier_subeventos["brier_1dia"])

analise_brier_subeventos = analise_brier_subeventos[[
    "event_id", "event_title", "question", "resultado_final", "resultado_yes",
    "resultado_no", "preco_yes", "preco_no", "p_yes_1mes", "p_no_1mes",
    "p_yes_1semana", "p_no_1semana", "p_yes_1dia", "p_no_1dia",
    "prob_resultado_1mes", "prob_resultado_1semana", "prob_resultado_1dia",
    "brier_1mes", "brier_1semana", "brier_1dia", "melhora_1mes_para_1dia",
    "volume", "liquidity"
]]

# ===========================================================
# Ajustando as variáveis de Plotagem
# ===========================================================

analise_brier_subeventos["volume_submercado"] = analise_brier_subeventos["volume"]
analise_brier_subeventos["log_volume_submercado"] = np.log1p(analise_brier_subeventos["volume_submercado"])
analise_brier_subeventos["erro_abs_1mes"] = (1 - analise_brier_subeventos["prob_resultado_1mes"])
analise_brier_subeventos["erro_abs_1semana"] = (1 - analise_brier_subeventos["prob_resultado_1semana"])
analise_brier_subeventos["erro_abs_1dia"] = (1 - analise_brier_subeventos["prob_resultado_1dia"])




# Identificando dinamicamente os nomes das colunas de texto no df_events
colunas_events = df_events_pequeno.columns
cat_col_events = 'category' if 'category' in colunas_events else ('slug' if 'slug' in colunas_events else None)
title_col_events = 'title' if 'title' in colunas_events else ('event_title' if 'event_title' in colunas_events else None)

if cat_col_events and title_col_events:
    # Cria o dicionário mapeando Título do Evento -> Categoria
    mapeamento_categorias = dict(zip(df_events_pequeno[title_col_events], df_events_pequeno[cat_col_events]))
    
    # Aplica o mapeamento diretamente na nossa coluna 'event_title'
    analise_brier_subeventos['category'] = analise_brier_subeventos['event_title'].map(mapeamento_categorias)
else:
    # Caso as colunas tenham nomes completamente diferentes, tenta mapear por índice parcial
    analise_brier_subeventos['category'] = 'Geral'

# Se mesmo com o mapeamento restarem nulos, preenchemos para não perder dados no dropna
analise_brier_subeventos['category'] = analise_brier_subeventos['category'].fillna('Outros / Não Especificado')


# =========================================
# PREPARANDO O PLOT COM ERRO ABSOLUTO
# =========================================

base_plot = analise_brier_subeventos.dropna(subset=[
    "log_volume_submercado", "erro_abs_1mes", "erro_abs_1semana", "erro_abs_1dia", "category"
]).copy()

# Transformando para formato longo preservando a Categoria mapeada
base_plot_longo = base_plot.melt(
    id_vars=["log_volume_submercado", "category"],
    value_vars=["erro_abs_1mes", "erro_abs_1semana", "erro_abs_1dia"],
    var_name="horizonte_tempo",
    value_name="erro_absoluto"
)

base_plot_longo["horizonte_tempo"] = base_plot_longo["horizonte_tempo"].replace({
    "erro_abs_1mes": "1 mês antes",
    "erro_abs_1semana": "1 semana antes",
    "erro_abs_1dia": "1 dia antes"
})

base_plot_longo = base_plot_longo[base_plot_longo["erro_absoluto"] >= 0.00001].copy()


# Plot geral

if not base_plot_longo.empty:
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=base_plot_longo,
        x="log_volume_submercado",
        y="erro_absoluto",
        hue="horizonte_tempo",
        alpha=0.4
    )

    for horizonte in ["1 mês antes", "1 semana antes", "1 dia antes"]:
        temp = base_plot_longo[base_plot_longo["horizonte_tempo"] == horizonte]
        if not temp.empty and len(temp) > 1:
            sns.regplot(
                data=temp,
                x="log_volume_submercado",
                y="erro_absoluto",
                scatter=False,
            )
            
    plt.title("Assertividade Global do Mercado — Volume vs Erro Absoluto")
    plt.xlabel("Log do volume do submercado")
    plt.ylabel("Erro absoluto")
    plt.show()


    # Plot isolados
    
    # Descobre as categorias únicas presentes na base tratada
    categorias_unicas = base_plot_longo["category"].dropna().unique()

    for cat in categorias_unicas:
        dados_cat = base_plot_longo[base_plot_longo["category"] == cat]
        
        # Filtro de relevância: ignora categorias com pouquíssimas amostras para manter a linha estável
            
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=dados_cat,
            x="log_volume_submercado",
            y="erro_absoluto",
            hue="horizonte_tempo",
            alpha=0.4
        )
        
        for horizonte in ["1 mês antes", "1 semana antes", "1 dia antes"]:
            temp = dados_cat[dados_cat["horizonte_tempo"] == horizonte]
            if not temp.empty and len(temp) > 1:
                sns.regplot(
                    data=temp,
                    x="log_volume_submercado",
                    y="erro_absoluto",
                    scatter=False,
                )
                
        plt.title(f"Categoria Isolada: {cat} — Volume vs Erro Absoluto")
        plt.xlabel("Log do volume do submercado")
        plt.ylabel("Erro absoluto")
        plt.show() # Abre o gráfico da categoria atual e espera você fechar para exibir a próxima
else:
    print("Base plotagem vazia")