"""
item_a.py -- Ex3 -- MLP vs CNN no MNIST
=========================================
  1. MLP -- ao menos 2 camadas densas
  2. CNN -- ao menos 2 blocos Conv2D + MaxPooling2D

Para cada modelo:
  - Imprime o resumo com model.summary()
  - Plota curvas de acuracia e loss (treino vs. validacao)
  - Registra a acuracia no conjunto de teste

Ao final gera uma tabela comparativa com:
  - Numero de parametros
  - Tempo medio de treino por epoca
  - Acuracia de teste
"""

import argparse
from pathlib import Path

from mnist_utils import (
    load_mnist,
    build_mlp,
    build_cnn,
    EpochTimer,
    plot_history,
    model_summary_row,
    save_comparison_table,
)


def train_model(name, build_fn, x_tr, y_tr, x_te, y_te, epochs=10, bs=128):
    """Treina um modelo, exibe resumo, plota curvas e retorna linha da tabela.

    Args:
        name     : nome exibido nos logs e na tabela.
        build_fn : callable que constroi e compila o modelo.
        x_tr / y_tr : dados de treino.
        x_te / y_te : dados de teste.
        epochs   : numero de epocas.
        bs       : tamanho do batch.
    """
    print(f"\n{'='*50}")
    print(f"  Treinando: {name}")
    print(f"{'='*50}")

    model = build_fn()

    # Exibe arquitetura: camadas, shapes e contagem de parametros
    model.summary()

    timer = EpochTimer()
    history = model.fit(
        x_tr, y_tr,
        epochs=epochs,
        batch_size=bs,
        validation_split=0.1,   # 10% do treino usado como validacao
        callbacks=[timer],
        verbose=2,
    )

    # Acuracia no conjunto de teste (dados nunca vistos durante o treino)
    _, test_acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"[{name}] acuracia_teste = {test_acc:.4f}")

    # Salva grafico de treino vs validacao
    out_dir = Path("resultados")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"{name.lower()}_history.png"
    plot_history(history,
                 title=f"{name} - Treino vs Validacao",
                 output_path=plot_path)

    return model_summary_row(name, model, timer.epoch_times, test_acc)


def main():
    ap = argparse.ArgumentParser(description="MLP vs CNN no MNIST.")
    ap.add_argument("--epochs", type=int, default=10,
                    help="Numero de epocas (padrao: 10).")
    ap.add_argument("--batch",  type=int, default=128,
                    help="Tamanho do batch (padrao: 128).")
    args = ap.parse_args()

    # Carrega MNIST:
    #   MLP recebe vetores 1-D de 784 pixels (flatten=True).
    #   CNN recebe imagens 28x28x1 (flatten=False).
    (x_tr_f, y_tr), (x_te_f, y_te) = load_mnist(flatten=True)
    (x_tr_i, _),    (x_te_i, _)    = load_mnist(flatten=False)

    rows = []

    # ====================================================================
    # Modelo 1 -- MLP (Multi-Layer Perceptron)
    # ====================================================================
    # O MLP ve cada pixel como uma feature independente, sem explorar a
    # estrutura espacial da imagem. Isso limita sua capacidade de detectar
    # padroes locais (bordas, curvas) e o torna mais suscetivel a overfitting
    # quando o numero de parametros e grande.
    rows.append(train_model(
        "MLP", build_mlp,
        x_tr_f, y_tr, x_te_f, y_te,
        epochs=args.epochs, bs=args.batch,
    ))

    # ====================================================================
    # Modelo 2 -- CNN (Convolutional Neural Network)
    # ====================================================================
    # A CNN supera o MLP nesta tarefa porque:
    #   1. Localidade espacial: filtros Conv2D detectam padroes locais
    #      (bordas, angulos) de forma invariante a translacao -- algo que
    #      o MLP nao consegue sem tratar cada pixel independentemente.
    #   2. Eficiencia de parametros: pesos sao compartilhados em toda a
    #      imagem via convolucao, reduzindo overfitting e melhorando
    #      generalizacao.
    #   3. Hierarquia de features: MaxPooling reduz dimensionalidade e
    #      amplia o campo receptivo, permitindo capturar estruturas em
    #      multiplas escalas (tracos -> formas -> digito completo).
    rows.append(train_model(
        "CNN", build_cnn,
        x_tr_i, y_tr, x_te_i, y_te,
        epochs=args.epochs, bs=args.batch,
    ))

    # Imprime e salva tabela comparativa (parametros, tempo/epoca, acuracia)
    save_comparison_table(rows, path="resultados/comparacao_modelos.csv")

    # ====================================================================
    # OVERFITTING -- em qual curva ha indicio?
    # ====================================================================
    # O sinal classico e quando a acuracia de TREINO continua subindo mas a
    # de VALIDACAO estabiliza ou cai; ou quando a loss de VALIDACAO comeca
    # a aumentar enquanto a de TREINO diminui.
    #
    # Em experimentos tipicos (10 epocas, sem regularizacao), o MLP exibe
    # esse padrao antes da CNN. Observe resultados/mlp_history.png: a gap
    # entre as curvas de treino e validacao tende a crescer com o numero de
    # epocas, indicando que o modelo memorizou os dados de treino.
    #
    # A CNN, com menos parametros efetivos e features mais generalizaveis,
    # costuma manter treino e validacao mais proximos.


if __name__ == "__main__":
    main()
