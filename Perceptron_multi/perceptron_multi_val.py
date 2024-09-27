import numpy as np
import pandas as pd 
from scipy.io import arff
import os

# Constantes globais
BIAS = -1
B = 0.5 
TAXA_APRENDIZADO = 0.1 
ERRO = 10**(-6)


# Função de ativação (sigmoide) e sua derivada
def funcao_ativacao(u): 
    return 1/(1 + np.exp(-B*u)) 

def derivada_ativacao(u):
    gu = funcao_ativacao(u) 
    return B*gu*(1 - gu) 

# Função para ler arquivo .dat e retornar os dados como uma matriz numpy
def read_file(file_path):
    def encode_class(class_name):
        if class_name == 'Iris-setosa':
            return 0
        elif class_name == 'Iris-versicolor':
            return 1
        elif class_name == 'Iris-virginica':
            return 2
        else:
            return -1  # Classe desconhecida

    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_start = lines.index('@data\n') + 1
    data_lines = lines[data_start:]
    
    data = []
    for line in data_lines:
        values = line.strip().split(', ')
        encoded_class = encode_class(values[-1])
        data_row = [-1] + [float(v) for v in values[:-1]] + [encoded_class]
        data.append(data_row)

    return np.array(data)

# Função para carregar dados de múltiplos arquivos e organizar em treino, teste e validação
def load_data_from_files(base_path, folds=10):
    datasets = {'train': [], 'test': [], 'val': []}

    for i in range(1, folds + 1):
        
        train_file = os.path.join(base_path, f'iris-10dobscv-{i}tra.dat') 
        test_file = os.path.join(base_path, f'iris-10dobscv-{i}tst.dat')
        val_file = os.path.join(base_path, f'iris-10dobscv-{i}val.dat')
       
        
        # Ler os arquivos
        train_data = read_file(train_file)
        test_data = read_file(test_file)
        val_data = read_file(val_file)
        
        # Armazenar dados
        datasets['train'].append(train_data)
        datasets['test'].append(test_data)
        datasets['val'].append(val_data)
    
    return datasets

#função que constroe e retorna uma matriz de pesos Mxy com pesos aleatórios entre 0 e 1 
def matrizes_pesos_iniciais(x, y): 
    return np.random.rand(x, y)  

# Função para cálculo da somatória
def somatoria(entradas , pesos):
    u = 0 
    for i in range(len(pesos)):  
        u += entradas[i]*pesos[i] 
    return u

# Função para arredondar os valores da saída
def arrendonda_vetor(Y): 
    X = []
    for i in range(len(Y)):
        if Y[i] >= 0.5: 
            X.append(1)
        else: 
            X.append(0) 
    return X 

# Funções para transformar e decodificar a saída
def transforma_saida(Y): 
    if Y == [0, 0]:
        return 0
    elif Y == [0, 1]:
        return 1
    elif Y == [1, 0]:
        return 2
    else:
        return 3

def decodifica_saida(x): 
    if x == 0:
        return [0, 0]
    elif x == 1:
        return [0, 1]
    elif x == 2:
        return [1, 0]
    
# Função para calcular a saída da camada oculta
def calcular_saida_oculta(entradas, matriz_camada_oculta):
    I_camada_oculta = []
    Y_camada_oculta = [-1]  # Adiciona o bias à saída da camada oculta

    for z in range(len(matriz_camada_oculta)):  
        pesos_oculta = matriz_camada_oculta[z]
        u = somatoria(entradas, pesos_oculta)
        I_camada_oculta.append(u)
        Y_camada_oculta.append(funcao_ativacao(u))

    return I_camada_oculta, Y_camada_oculta

# Função para calcular a saída da camada de saída
def calcular_saida_final(Y_camada_oculta, matriz_camada_saida):
    I_camada_saida = []
    Y_camada_saida = []

    for x in range(len(matriz_camada_saida)):
        pesos_saida = matriz_camada_saida[x]
        u = somatoria(Y_camada_oculta, pesos_saida)
        I_camada_saida.append(u)
        Y_camada_saida.append(funcao_ativacao(u))

    return I_camada_saida, Y_camada_saida

# Função para calcular gradiente da camada de saída
def calcular_gradientes_saida(vetor_saida, Y_camada_saida, I_camada_saida):
    vetor_gradiente_saida = []
    for x in range(len(Y_camada_saida)):
        grad = gradiente_oculto(vetor_saida[x], Y_camada_saida[x], I_camada_saida[x])
        vetor_gradiente_saida.append(grad)
    return vetor_gradiente_saida

# Função para atualizar os pesos da camada de saída
def atualizar_pesos_saida(matriz_camada_saida, Y_camada_oculta, vetor_gradiente_saida):
    for x in range(len(matriz_camada_saida)):
        for y in range(len(Y_camada_oculta)):
            matriz_camada_saida[x][y] += TAXA_APRENDIZADO * vetor_gradiente_saida[x] * Y_camada_oculta[y]

# Função para calcular gradientes da camada oculta
def calcular_gradientes_oculta(vetor_gradiente_saida, matriz_camada_saida, I_camada_oculta):
    vetor_gradiente_oculta = []

    for x in range(1, len(I_camada_oculta)):  # Ignora o bias
        erro_oculta = 0
        for y in range(len(vetor_gradiente_saida)):
            erro_oculta += vetor_gradiente_saida[y] * matriz_camada_saida[y][x]

        gradiente_oculta = erro_oculta * derivada_ativacao(I_camada_oculta[x - 1])
        vetor_gradiente_oculta.append(gradiente_oculta)

    return vetor_gradiente_oculta

# Função para atualizar os pesos da camada oculta
def atualizar_pesos_oculta(matriz_camada_oculta, entradas, vetor_gradiente_oculta):
    for x in range(len(vetor_gradiente_oculta)):
        for z in range(len(entradas) - 1):  # Ignora o valor de classificação nas entradas
            matriz_camada_oculta[x][z] += TAXA_APRENDIZADO * vetor_gradiente_oculta[x] * entradas[z]

# Função para calcular o gradiente na camada oculta
def gradiente_oculto(saida_esperada, saida_obtida, saida_ponderada):
    return (saida_esperada - saida_obtida)*derivada_ativacao(saida_ponderada)

# Função para validar a rede após o treinamento
def validacao(data_val, matriz_camada_oculta, matriz_camada_saida):
    acertos = 0
    for entradas in data_val:
        # Cálculo da camada oculta
        _, Y_camada_oculta = calcular_saida_oculta(entradas, matriz_camada_oculta)
        
        # Cálculo da camada de saída
        _, Y_camada_saida = calcular_saida_final(Y_camada_oculta, matriz_camada_saida)
        
        # Arredondar e transformar a saída
        vetor_arredondado = arrendonda_vetor(Y_camada_saida)
        saida_obtida = transforma_saida(vetor_arredondado)
        
        # Classe esperada
        saida_esperada = entradas[-1]  # A classe está na última posição
        
        # Verifica se a saída obtida é igual à saída esperada
        if saida_obtida == saida_esperada:
            acertos += 1

    # Calcula a acurácia (porcentagem de acertos)
    acuracia = acertos / len(data_val) * 100
    return acuracia

# Função para calcular o erro quadrático para uma amostra
def calcular_erro_quadratico(vetor_saida_esperada, vetor_saida_obtida):
    erro_quadratico = 0
    for i in range(len(vetor_saida_esperada)):
        erro_quadratico += (vetor_saida_esperada[i] - vetor_saida_obtida[i])**2
    return erro_quadratico

def treinamento(datasets):
    topologias = [
        {'oculta': 5, 'saida': 2},    # 4-5-2
        {'oculta': 10, 'saida': 2},   # 4-10-2
        {'oculta': 15, 'saida': 2},   # 4-15-2
        {'oculta': 20, 'saida': 2}    # 4-20-2
    ]

    resultados = []

    for topologia in topologias:
        eqms = []
        epocas_lista = []
        acuracias = []

        for i in range(10):  # 10-fold cross-validation
            data_treino = datasets["train"][i]
            data_val = datasets["val"][i]
            eqm_ant = 5
            eqm_atual = 0
            epoca = 0

            # Inicialização das matrizes de pesos com base na topologia atual
            matriz_camada_oculta = matrizes_pesos_iniciais(topologia['oculta'], 5)  # 5 entradas (+ bias)
            matriz_camada_saida = matrizes_pesos_iniciais(topologia['saida'], topologia['oculta'] + 1)  # +1 para o bias

            while abs(eqm_atual - eqm_ant) > ERRO:
                eqm_ant = eqm_atual
                erro_quadratico_total = 0

                for j in range(len(data_treino)):
                    entradas = data_treino[j]

                    # Cálculo da camada oculta
                    I_camada_oculta, Y_camada_oculta = calcular_saida_oculta(entradas, matriz_camada_oculta)

                    # Cálculo da camada de saída
                    I_camada_saida, Y_camada_saida = calcular_saida_final(Y_camada_oculta, matriz_camada_saida)

                    # Arredondar e transformar a saída
                    vetor_arredondado = arrendonda_vetor(Y_camada_saida)
                    saida_obtida = transforma_saida(vetor_arredondado)
                    saida_esperada = entradas[-1]
                    vetor_saida = decodifica_saida(saida_esperada)

                    # Cálculo dos gradientes e atualização dos pesos
                    vetor_gradiente_saida = calcular_gradientes_saida(vetor_saida, Y_camada_saida, I_camada_saida)
                    atualizar_pesos_saida(matriz_camada_saida, Y_camada_oculta, vetor_gradiente_saida)

                    vetor_gradiente_oculta = calcular_gradientes_oculta(vetor_gradiente_saida, matriz_camada_saida, I_camada_oculta)
                    atualizar_pesos_oculta(matriz_camada_oculta, entradas, vetor_gradiente_oculta)

                    erro_quadratico = calcular_erro_quadratico(vetor_saida, Y_camada_saida)
                    erro_quadratico_total += erro_quadratico

                eqm_atual = erro_quadratico_total / (2*len(data_treino))
                epoca += 1

            # Validação após o treinamento
            acuracia = validacao(data_val, matriz_camada_oculta, matriz_camada_saida)
            eqms.append(eqm_atual)
            epocas_lista.append(epoca)
            acuracias.append(acuracia)

            print(f"Topologia: {topologia['oculta']}-{topologia['saida']} | Fold {i + 1} | Acurácia: {acuracia:.2f}% | Épocas: {epoca}")

        # Registrar as métricas: média e desvio padrão
        media_eqm = np.mean(eqms)
        dp_eqm = np.std(eqms)
        media_epocas = np.mean(epocas_lista)
        dp_epocas = np.std(epocas_lista)
        media_acuracia = np.mean(acuracias)
        dp_acuracia = np.std(acuracias)

        resultados.append({
            'topologia': f"4-{topologia['oculta']}-{topologia['saida']}",
            'media_eqm': media_eqm,
            'dp_eqm': dp_eqm,
            'media_epocas': media_epocas,
            'dp_epocas': dp_epocas,
            'media_acuracia': media_acuracia,
            'dp_acuracia': dp_acuracia
        })

    # Exibir resultados
    print("\nResultados:")
    print("Topologia   | EQM (média ± dp)  | Épocas (média ± dp)  | Acurácia (média ± dp)")
    for resultado in resultados:
        print(f"{resultado['topologia']} | {resultado['media_eqm']:.6f} ± {resultado['dp_eqm']:.6f} | {resultado['media_epocas']:.2f} ± {resultado['dp_epocas']:.2f} | {resultado['media_acuracia']:.2f}% ± {resultado['dp_acuracia']:.2f}%")

def main(): 
    base_path = 'Perceptron_multi/iris'
    datasets = load_data_from_files(base_path, folds=10)
    treinamento(datasets)


main()
