import pandas as pd
import random
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

produtos = ['Camiseta', 'Calça Jeans', 'Jaqueta', 'Tênis', 'Boné', 'Óculos de Sol', 'Bolsa', 
            'Meias', 'Cinto', 'Vestido', 'Saia', 'Blusa', 'Shorts', 'Cachecol', 'Luvas',
            'Sandálias', 'Botas'
            ]

def gerar_transacao():
    principal = random.choice([
        ['Camiseta', 'Calça Jeans', 'Tênis', 'Boné', 'Cinto', 'Meias'],
        ['Vestido', 'Sandálias', 'Bolsa', 'Óculos de Sol'],
        ['Blusa', 'Saia', 'Tênis', 'Cachecol'],
        ['Jaqueta', 'Calça Jeans', 'Botas', 'Luvas'],
        ['Shorts', 'Camiseta', 'Tênis', 'Boné']
    ])
    
    transacao = random.sample(principal, k=random.randint(2, len(principal)))
    
    acessorios = list(set(produtos) - set(principal))
    if random.random() < 0.3:
        transacao += random.sample(acessorios, k=1)
    return transacao

random.seed(42)
num_transacoes = 50
transacoes = [gerar_transacao() for _ in range(num_transacoes)]

max_items = max(len(t) for t in transacoes)
for t in transacoes:
    while len(t) < max_items:
        t.append(None)
        

colunas = [f'Item {i+1}' for i in range(max_items)]
df_transacoes = pd.DataFrame(transacoes, columns=colunas)
csv_path = 'apriori/transacoes.csv'
df_transacoes.to_csv(csv_path, index=False)

print(f"Arquivo CSV salvo em: {csv_path}")

df = pd.read_csv(csv_path)

transacoes_processadas = df.apply(lambda row: [item for item in row if pd.notnull(item)], axis=1)

te = TransactionEncoder()
te_ary = te.fit(transacoes_processadas).transform(transacoes_processadas)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

print("\n DataFrame One-Hot Encoded:")
print(df_onehot.head())

frequent_itemsets = apriori(df_onehot, min_support=0.2, use_colnames=True)
print("\n Frequent Itemsets:")
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

if not rules.empty:
    print("\n Association Rules:")
    for idx, row in rules.iterrows():
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        print(f"📌 {antecedents} => {consequents}")
        print(f"   - Support: {row['support']*100:.2f}%")
        print(f"   - Confidence: {row['confidence']*100:.2f}%")  
        print(f"   - Lift: {row['lift']:.2f}\n")
else:
    print("\n No association rules found with the given thresholds.")