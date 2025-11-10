# -*- coding: utf-8 -*-
"""atividade-apriori.ipynb

Atividade prática de Inteligência Artificial
TEMA: Regras de Associação com o Algoritmo Apriori
"""

import warnings
warnings.filterwarnings("ignore")

# === 1. Importação das bibliotecas =============================
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# === 2. Carregamento do dataset ================================
# Carrega o arquivo enem_ce.csv
df = pd.read_csv("enem_ce.csv", sep=";")

# Exibe as 5 primeiras linhas
print("Pré-visualização dos dados:")
print(df.head(5))

# === 3. Seleção das colunas categóricas relevantes ============
cols = [
    "genero",
    "aproveitamento",
    "dependencia_adm",
    "tipo_escola",
    "faixa_etaria",
    "estado_civil",
    "raca",
    "escolaridade_pai",
    "escolaridade_mae",
    "renda",
    "tem_computador",
    "tem_internet"
]

df = df[cols].dropna()

# Converte todas as colunas para string
for c in df.columns:
    df[c] = df[c].astype(str)

# === 4. Transformação One-Hot (necessária para o Apriori) =====
ohe = pd.get_dummies(df)

# === 5. Definição dos limiares (suporte e confiança) ==========
SUPORTE_MIN = 0.05      # item deve aparecer em pelo menos 5% das transações
CONFIANCA_MIN = 0.5     # a regra deve ser verdadeira em pelo menos 50% dos casos

# === 6. Aplicação do algoritmo Apriori ========================
itens_freq = apriori(ohe, min_support=SUPORTE_MIN, use_colnames=True)

print("\nItens frequentes:")
print(itens_freq.head())

# === 7. Geração das regras de associação ======================
regras = association_rules(itens_freq, metric="confidence", min_threshold=CONFIANCA_MIN)

# === 8. Selecionar apenas colunas principais ==================
regras = regras[["antecedents", "consequents", "support", "confidence", "lift"]]

# === 9. Ordenar e exibir as Top 10 regras =====================
regras_top10 = regras.sort_values(by="lift", ascending=False).head(10)
print("\nTop 10 Regras (ordenadas por lift):")
print(regras_top10)

# === 10. Salvar resultados em CSV =============================
regras_top10.to_csv("regras_enem_top10_min.csv", index=False, encoding="utf-8-sig")

print("\nArquivo 'regras_enem_top10_min.csv' salvo com sucesso!")
