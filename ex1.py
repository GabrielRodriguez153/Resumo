import math

def distancia_euclidiana(vetor1, vetor2):
    soma = 0
    for i in range(len(vetor1)):
        soma += (vetor1[i] - vetor2[i]) ** 2
 
    return math.sqrt(soma)

usuarioA = [5, 4, 3]
usuarioB = [4, 5, 2]

print("Distância A-B:", round(distancia_euclidiana(usuarioA, usuarioB), 2))

def similaridade_cosseno(vetor1, vetor2):
        produto_interno = sum(a * b for a, b in zip(vetor1, vetor2))
        norma1 = math.sqrt(sum(a ** 2 for a in vetor1))
        norma2 = math.sqrt(sum(b ** 2 for b in vetor2))
        if norma1 == 0 or norma2 == 0:
            return 0
        return produto_interno / (norma1 * norma2)

print("Similaridade cosseno A-B:", round(similaridade_cosseno(usuarioA, usuarioB), 2))