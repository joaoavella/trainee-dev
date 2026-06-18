# Analisando Previsões e Desempenho: Ciência de Dados Aplicada a Mercados de Predição e Esportes

Este repositório contém o projeto desenvolvido para o Hackathon de Ciência de Dados e Inteligência Artificial. O objetivo do trabalho é aplicar conceitos de manipulação, limpeza e análise exploratória de dados, além de modelos básicos de aprendizado de máquina, para investigar cenários reais de previsibilidade e desempenho.

## 👥 Membros do Grupo
* **Caio Luca Sousa da Silva**
* **João Victor Pereira Avella**
* **Marcelo Kenji Tanaka**
* **Miguel Cunha e Costa**

---

## 📌 Temas Propostos

Para este hackathon, estruturamos duas frentes de análise possíveis. A primeira (principal) analisa a eficiência de mercados de apostas descentralizados, enquanto a segunda (alternativa) cruza o histórico de conquistas esportivas com indicadores socioeconômicos.

### Opção 1 (Principal): Uma Análise das Previsões do Polymarket
Os mercados de previsão (como o Polymarket) funcionam como termômetros financeiros para eventos do mundo real. Este projeto analisa a precisão com que as probabilidades implícitas de apostas conseguem prever os desfechos reais em categorias como política, cripto e esportes.

* **O que pretendemos responder:**
  1. **Acurácia Geral:** Qual é a taxa real de acerto das previsões do Polymarket? Eventos com alta probabilidade implícita (ex: acima de 80%) de fato se consolidam nessa mesma proporção?
  2. **Efeito Volume e Liquidez:** Existe correlação entre o volume financeiro negociado de um mercado de apostas e a precisão de suas previsões? Mercados mais líquidos são preditores estatisticamente melhores?
  3. **Análise de Categorias:** A eficiência de previsão varia significativamente entre temas diferentes (ex: o mercado é melhor em prever decisões de Cripto/Finanças do que resultados de Política e Cultura)?

### Opção 2 (Alternativa): Relação entre o Investimento de Países e seus Desempenhos no Futebol
O sucesso esportivo internacional muitas vezes é associado à estrutura financeira de um país. Esta proposta visa entender se o poder econômico de uma nação (PIB ou investimento no esporte) é o fator determinante para o sucesso de sua seleção de futebol ou se fatores como tradição e o fator casa se sobrepõem.

* **O que pretendemos responder:**
  1. **Investimento vs. Resultados:** Existe uma correlação linear forte entre indicadores de investimento/PIB de um país (que cruzaremos externamente) e seu aproveitamento histórico (taxa de vitória, saldo de gols) no futebol internacional?
  2. **Fator Casa (*Home Advantage*):** A vantagem de jogar em seu próprio país é capaz de compensar assimetrias financeiras entre seleções de diferentes portes econômicos?
  3. **Modelo Preditivo:** É possível utilizar regressão (ou modelos de classificação simples) para prever o resultado de partidas internacionais utilizando o histórico recente de desempenho associado a dados macroeconômicos do país?

---

## 📊 Bases de Dados Utilizadas

Para viabilizar as análises descritas, utilizaremos os seguintes conjuntos de dados públicos disponíveis no Kaggle:

### 1. Polymarket Prediction Markets
* **Link para download:** [Kaggle - Polymarket Prediction Markets Dataset](https://www.kaggle.com/datasets/ismetsemedov/polymarket-prediction-markets)
* **Descrição dos dados:** Esta base contém dados de mercado da plataforma Polymarket. Ela registra mais de 43.000 eventos e 100.000 mercados de previsão divididos por categorias (Política, Cripto, Cultura, Esportes, etc.). Apresenta variáveis de volume financeiro total acumulado, níveis de liquidez diária, data de criação do mercado, preços de negociação e os resultados finais apurados (resolução do mercado).
* **Utilidade:** Permite mapear a variação dos preços (probabilidades) ao longo do tempo e comparar a última probabilidade registrada antes do encerramento com o desfecho real do evento.

### 2. International Football Results (1872 - Presente)
* **Link para download:** [Kaggle - International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
* **Descrição dos dados:** Um registro histórico extenso contendo mais de 49.000 partidas internacionais de futebol masculino. A base inclui dados cruciais de cada confronto como: data, seleção mandante, seleção visitante, gols marcados por cada time, tipo de competição (copas continentais, eliminatórias, Copa do Mundo ou amistosos), cidade e país sede da partida, e se o confronto ocorreu em campo neutro.
* **Utilidade:** Permite calcular o aproveitamento histórico de cada seleção nacional. *Nota: Para viabilizar a análise do Tema 2, esses dados de desempenho esportivo serão cruzados com tabelas públicas de indicadores macroeconômicos dos países correspondentes.*
