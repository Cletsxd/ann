import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time

from IPython.display import clear_output
from sklearn.datasets import make_circles

# CLASE PARA LAS CAPAS DE UNA ANN
class NeuralLayer():

    # n_conn: número de conexiones de las neuronas
    # n_neur: número de neuronas
    # act_f: función de activación
    def __init__(self, n_conn, n_neur, act_f):

        self.act_f=act_f

        # vector b: Bias, tantos como neuronas, es decir b=n_neur
        # random values: (-1, 1)
        self.b = np.random.rand(1, n_neur) * 2 -1

        # matriz w [n_conn x n_neur]: pesos, ej: 3 conexiones => 1 neurona
        self.w = np.random.rand(n_conn, n_neur) * 2 -1

# Crear TOPOLOGÍA DE LA ANN
def create_nn(topology, act_f):
    
    # CREACIÓN DE LA ANN, VECTOR DE CAPAS
    neural_net = []

    for l, layer in enumerate(topology[: -1]):
        # Inserta las capas a la ANN
        # índice l: número de capas
        neural_net.append(NeuralLayer(topology[l], topology[l+1], act_f))

    return neural_net

# FUNCIÓN DE ENTRENAMIENTO
# X: datos entrada
# Y: datos salida esperada
def train(neural_net, X, Y, e2medio, learning_rate=0.5, train=True):
    # X, Y: matrices
    
    # vector output: salida
    output = [(None, X)]

    # Forward pass
    for l, layer in enumerate(neural_net):
        # l: recorre las capas de la neural_net
        # z: suma ponderada
        z = output[-1][1] @ neural_net[l].w + neural_net[l].b

        # a: salida capa1
        a = neural_net[l].act_f[0](z)

        output.append((z, a))

    if train:
        # Backward pass
        # Backpropagation algorithm

        deltas = []
        # len(neural_net): número de capas de la ANN
        for l in reversed(range(0, len(neural_net))):
            # output[l+1][0]: suma ponderada
            z = output[l+1][0]
            # output[l+1][1]: activación
            a = output[l+1][1]
            if l == len(neural_net) -1:
                # Calcular delta última capa con respecto al coste
                deltas.insert(0, e2medio[1](a, Y) * neural_net[l].act_f[1](a))
            else:
                # Calcular capa respecto de capa previa
                deltas.insert(0, deltas[0] @ _w.T * neural_net[l].act_f[1](a))

            _w = neural_net[l].w

            # Gradiente descendiente
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * learning_rate
            neural_net[l].w = neural_net[l].w - output[l][1].T @ deltas[0] * learning_rate

    return output[-1][1]

# Función de activación
# SIGMOIDAL
# sigmodial[0](x) // función de activación Sigmoidal
# sigmodial[1](x) // derivada de la Sigmoidal
sigmoidal = (lambda x: 1 / (1 + np.e ** (-x)),
            lambda x: x * (1 - x))

# Función de costo
# ERROR CUADRÁTICO MEDIO
# yp: salida real de la ANN
# yr: salida predicha de la entrada
# e2medio[0](yp, yr) // función
# e2medio[1](yp, yr) // derivada
e2medio = (lambda yp, yr: np.mean((yp, yr)) ** 2,
            lambda yp, yr: (yp - yr))

if __name__ == "__main__":
    # Número de registros (filas)
    n = 500
    # Características (columnas)
    p = 2

    topology = [p, 4, 8, 1]

    # hacer print a X y Y para visualizar datos
    # matriz X: entradas
    # matriz Y: salidas
    X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)

    Y = Y[:, np.newaxis]

    # vector neural_net: vector de capas
    neural_net = create_nn(topology, sigmoidal)

    loss = []

    #X = np.array([[0,0], [1, 0], [0, 1], [1, 1]])
    #Y = np.array([0,1,1,0])
    # PASAR A FUNCIÓN
    for i in range(2500):
        out = train(neural_net, X, Y, e2medio, learning_rate=0.01)
        if i % 25 == 0:
            #print(out)
            loss.append(e2medio[0](out, Y))
            res = 50

            _x0 = np.linspace(-1.5, 1.5, res)
            _x1 = np.linspace(-1.5, 1.5, res)

            _Y = np.zeros((res, res))

            for i0, x0 in enumerate(_x0):
                for i1, x1 in enumerate(_x1):
                    _Y[i0, i1] = train(neural_net, np.array([[x0,x1]]), Y, e2medio, train=False)[0][0]

            plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
            plt.axis("equal")

            plt.scatter(X[ Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
            plt.scatter(X[ Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")

            clear_output(wait=True)
            plt.show()
            plt.plot(range(len(loss)), loss)
            #print(loss)
            plt.show()
            time.sleep(0.5)