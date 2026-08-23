import cv2
import numpy as np
import time
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    left_path = os.path.join('left.png')
    right_path = os.path.join('right.png')

    img1 = cv2.imread(left_path)
    img2 = cv2.imread(right_path)

    if img1 is None or img2 is None:
        print(f"Erro: não foi possível carregar as imagens.")
        print(f"Caminhos buscados:\n{left_path}\n{right_path}")
        print("Certifique-se de ter executado o item_a do ex1 primeiro para gerar essas imagens.")
        return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    try:
        akaze = cv2.AKAZE_create()
    except AttributeError:
        akaze = cv2.xfeatures2d.AKAZE_create() 

    detectores = {
        'SIFT': cv2.SIFT_create(),
        'ORB': cv2.ORB_create(nfeatures=1500),
        'AKAZE': akaze
    }

    cores = {
        'SIFT': (0, 255, 0),
        'ORB': (255, 0, 0),
        'AKAZE': (0, 165, 255)
    }

    try:
        surf = cv2.xfeatures2d.SURF_create()
        detectores['SURF'] = surf
        cores['SURF'] = (255, 255, 0)
    except (AttributeError, cv2.error):
        pass 

    resultados = []

    for nome, detector in detectores.items():
        t_inicio = time.perf_counter()
        
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        
        t_ms = (time.perf_counter() - t_inicio) * 1000.0

        n_kp = len(kp1) + len(kp2)
        dim_desc = desc1.shape[1] if desc1 is not None else 0

        resultados.append({
            'nome': nome,
            'n_kp1': len(kp1),
            'n_kp2': len(kp2),
            'n_kp': n_kp,
            'tempo_ms': t_ms,
            'dim_desc': dim_desc
        })

        img_kp1 = cv2.drawKeypoints(img1, kp1, None, color=cores[nome], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_kp2 = cv2.drawKeypoints(img2, kp2, None, color=cores[nome], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        combined = np.hstack((img_kp1, img_kp2))
        cv2.imshow(f'{nome} - Keypoints (Left | Right)', combined)

        print(f"\n[{nome}]")
        print(f"  Keypoints extraídos: {len(kp1)} (Left) + {len(kp2)} (Right) = {n_kp}")
        print(f"  Tempo de processamento: {t_ms:.2f} ms | Dimensão do descritor: {dim_desc}")

    # 4. Tabela Comparativa de Resultados
    print("\n" + "=" * 76)
    print(f"{'TABELA COMPARATIVA - SIFT vs ORB vs AKAZE':^76}")
    print("=" * 76)
    print(f"| {'Método':<8} | {'Keypoints':>10} | {'Tempo (ms)':>12} | {'Dim. Descritor':>15} |")
    print(f"|{'-'*10}|{'-'*12}|{'-'*14}|{'-'*17}|")

    for r in resultados:
        print(f"| {r['nome']:<8} | {r['n_kp']:>10} | {r['tempo_ms']:>12.2f} | {r['dim_desc']:>15} |")

    print("=" * 76)
    print("\nOBSERVAÇÕES:")
    print("- SIFT: Maior dimensionalidade (128, float32), mais robusto, porém mais lento.")
    print("- ORB: Descritor binário de tamanho 32, muito rápido (ideal para tempo real).")
    print("- AKAZE: Descritor binário (61), bom equilíbrio entre robustez e velocidade.")
    
    print("\nPressione qualquer tecla nas janelas de imagem para encerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
=============================================================================================
TABELA COMPARATIVA DOS DESCRITORES
=============================================================================================
| Método | Dim. Descritor | Tipo      | Velocidade   | Características Principais 
|--------|----------------|-----------|--------------|-----------------------------------------
| SIFT   | 128            | float32   | Lento        | Altamente preciso e robusto a variações 
|        |                |           |              | de escala, rotação e iluminação. Ocupa 
|        |                |           |              | mais memória. (Distância L2/Euclidiana)
|--------|----------------|-----------|--------------|-----------------------------------------
| ORB    | 32             | binário   | Muito Rápido | Alternativa super eficiente. Ideal para 
|        |                |           |              | robótica em tempo real. Menor precisão
|        |                |           |              | sob grandes variações. (Dist. Hamming)
|--------|----------------|-----------|--------------|-----------------------------------------
| AKAZE  | 61             | binário   | Médio/Rápido | Ótimo compromisso (precisão x velocidade).
|        |                |           |              | Mantém bordas melhor definidas usando um
|        |                |           |              | espaço de difusão não-linear.
=============================================================================================
"""

if __name__ == "__main__":
    main()
