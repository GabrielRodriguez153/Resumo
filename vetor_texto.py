from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texto1 = "Inteligência artificial é o futuro da tecnologia. Ela permite que máquinas aprendam e tomem decisões."

texto2 = "A tecnologia avança rapidamente com a inteligência artificial, que capacita máquinas a aprenderem e decidirem."

vetorizador = CountVectorizer().fit([texto1, texto2])

similaridade = cosine_similarity(
    vetorizador.transform([texto1]).toarray(),
    vetorizador.transform([texto2]).toarray()
)[0][0]

print("Similaridade (cosseno):", similaridade)

vetor_texto1 = vetorizador.transform([texto1]).toarray()
vetor_texto2 = vetorizador.transform([texto2]).toarray()

print("Vocabulário:", vetorizador.get_feature_names_out())
print("Vetor do Texto 1:", vetor_texto1)
print("Vetor do Texto 2:", vetor_texto2)