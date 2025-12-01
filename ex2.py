import argparse
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
from itertools import combinations,chain
from collections import defaultdict, Counter

st.set_page_config(page_title="Similaridade de Jaccard", layout="wide")

def jaccard(set_a, set_b):
    inter = set_a & set_b
    union = set_a | set_b
    return (len(inter) / len(union)) if union else 0.0, inter, union

def carregar_cestas(path_csv):
    df = pd.read_csv(path_csv)
    cestas = defaultdict(set)  
 
    for _, row in df.iterrows():
        cliente = row[0]
        itens = set(row[1:].dropna())  
        cestas[cliente] = itens
 
    return cestas
 
def matriz_jaccard(cestas, top_k_vizinho=3, max_itens=5):
    def recomendar_para(alvo):
        vizinhos = []
        for outro, items in cestas.items():
            if outro == alvo:
                continue
            sim, inter, union = jaccard(cestas[alvo], items)
            vizinhos.append((outro, sim))
       
        vizinhos.sort(key=lambda x: x[1], reverse=True)
        vizinhos_top = vizinhos[:top_k_vizinho]
 
        pesos = Counter()
        itens_alvo = cestas[alvo]
        for viz, sim in vizinhos_top:
            for item in cestas[viz] - itens_alvo:
                pesos[item] += sim
 
        recomendados = sorted(pesos.items(), key=lambda x: x[1], reverse=True)[:max_itens]
        return vizinhos_top, recomendados
 
    ranking = {}
    for cliente in cestas:
        ranking[cliente] = recomendar_para(cliente)
    return ranking

def plot_venn_diagram(set_a, set_b, label_a="A", label_b="B"):
    inter = set_a & set_b
    only_a = set_a - inter
    only_b = set_b - inter
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    
    circle_a = Circle((0.0,0.0), 1.0, fill=False)
    circle_b = Circle((1.0,0.0), 1.0, fill=False)
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)
    
    ax.text(-0.4, 1.1, f"{label_a}", fontsize=12, va='center', ha='left')
    ax.text(1.4, 1.1, f"{label_b}", fontsize=12, va='center', ha='right')
    
    ax.text(-0.5, 0.0, f"Somente {label_a}\n({len(only_a)})", fontsize=11, va='center', ha='center')
    ax.text(0.5, 0.0, f"Interseção\n({len(inter)})", fontsize=11, va='center', ha='center')
    ax.text(1.5, 0.0, f"Somente {label_b}\n({len(only_b)})", fontsize=11, va='center', ha='center')
    def short(items, max_itens=5):
        items = list(sorted(items))
        if len(items) > max_itens:
            items = items[:max_itens] + ["..."]
        return ", ".join(items) if items else "-"

    ax.text(-0.5, -0.75, short(only_a), fontsize=9, va='center', ha='center')
    ax.text(0.5, -0.75, short(inter), fontsize=9, va='center', ha='center')
    ax.text(1.5, -0.75, short(only_b), fontsize=9, va='center', ha='center')
    return fig
    
st.sidebar.title("Configurações")
st.sidebar.markdown("Carregue um CSV com cestas de clientes e explore a Similaridade de Jaccard.")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")
top_n = st.sidebar.slider("Top-N pares similares", min_value=1, max_value=20, value=10, step=1)
use_example = st.sidebar.checkbox("Usar CSV de exemplo", value=True if not uploaded_file else False)

if uploaded_file is not None and not use_example:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame([
        ("A", "maçã"),
        ("A", "banana"),
        ("A", "pão"),
        ("A", "leite"),
        ("A", "queijo"),
        ("B", "banana"),
        ("B", "leite"),
        ("B", "queijo"),
        ("B", "arroz"),
        ("B", "feijão"),
        ("C", "maçã"),
        ("C", "uva"),
        ("C", "iogurte"),
        ("C", "pão"),
        ("C", "queijo"),
        ("D", "arroz"),
        ("D", "feijão"),
        ("D", "oleo"),
        ("D", "macarrão"),
        ("D", "molho de tomate"),
        ("E", "banana"),
        ("E", "pão"),
        ("E", "queijo"),
        ("E", "presunto"),
        ("E", "maionese"),
        ("F", "leite"),
        ("F","café"),
        ("F", "açúcar"),
        ("F", "pão"),
        ("F", "manteiga"),
        ], columns=["Cliente", "Item"])
    



def main():
    parser = argparse.ArgumentParser(description="Demonstração da Similaridade de Jaccard")
    parser.add_argument("--csv", required=True, help="Caminho para o CSV de cestas")
    parser.add_argument("--top", type=int, default=5, help="Mostrar Top-N pares similares")
    parser.add_argument("--maxitens", type=int, default=5, help="Máximo de itens a ser exibido")
    parser.add_argument("--detalhe", nargs=2, metavar=("Client_A", "Client_B"), help="Detalhar interseção/união entre dois clientes")
    parser.add_argument("--recomendar", nargs=2, metavar=("Cliente", "K"), help="Recomendar itens para Cliente usando K vizinhos mais similares")
    args = parser.parse_args()
 
    cestas = carregar_cestas(args.csv)
    if not cestas:
        print("Nenhuma Cesta encontrada")
        return
 
    print("Clientes carregados:", ", ".join(sorted(cestas.keys())))
 
    ranking = matriz_jaccard(cestas, top_k_vizinho=3, max_itens=args.maxitens)
 
    # Detalhe interseção/união
    if args.detalhe:
        a, b = args.detalhe
        if a not in cestas or b not in cestas:
            print(f"\n[ERRO] Clientes '{a}' e/ou '{b}' não existem.")
        else:
            sim, inter, union = jaccard(cestas[a], cestas[b])
            print(f"\nDetalhe {a} x {b}")
            print(f"Jaccard: {sim:.4f}")
            print(f"Interseção ({len(inter)}): {sorted(list(inter))}")
            print(f"União ({len(union)}): {sorted(list(union))}")
 
    # Recomendações
    if args.recomendar:
        alvo, k = args.recomendar
        k = int(k)
        if alvo not in cestas:
            print(f"\n[ERRO] Cliente '{alvo}' não existe.")
        else:
            vizinhos_top, recomendados = matriz_jaccard(cestas, top_k_vizinho=k, max_itens=args.maxitens)[alvo]
            print(f"\nVizinhos mais similares de {alvo} (top {k}):")
            for viz, sim in vizinhos_top:
                print(f" - {viz}: Jaccard={sim:.4f}")
 
            print(f"\nRecomendações de itens para {alvo}:")
            if not recomendados:
                print(" Nenhuma recomendação (cliente já possui itens em comum com vizinhos).")
            else:
                for item, score in recomendados:
                    print(f" - {item} (score: {score:.4f})")
 
if __name__ == "__main__":
    main()
 