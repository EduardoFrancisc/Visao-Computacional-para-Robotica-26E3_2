import cv2
import numpy as np
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    left_path = os.path.join('left.png')
    right_path = os.path.join('right.png')

    img1 = cv2.imread(left_path)
    img2 = cv2.imread(right_path)

    if img1 is None or img2 is None:
        print(f"Erro: não foi possível carregar as imagens.")
        print("Certifique-se de ter executado o item_a do ex1 primeiro para gerar left.png e right.png.")
        return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    print(f"Keypoints extraídos: {len(kp1)} (Left) | {len(kp2)} (Right)")

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_bf = bf.match(desc1, desc2)
    
    matches_bf = sorted(matches_bf, key=lambda x: x.distance)
    
    print(f"BFMatcher (com Cross-Check): {len(matches_bf)} matches encontrados.")

    img_bf = cv2.drawMatches(img1, kp1, img2, kp2, matches_bf[:50], None, 
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('BFMatcher (Cross-Check) - Top 50', img_bf)

    index_params = dict(algorithm=1, trees=5) 
    search_params = dict(checks=50)            
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches_knn = flann.knnMatch(desc1, desc2, k=2)
    
    good_matches_flann = []
    ratio_threshold = 0.75
    
    for m, n in matches_knn:
        if m.distance < ratio_threshold * n.distance:
            good_matches_flann.append(m)
            
    print(f"FLANN (+ Lowe's Ratio Test): {len(good_matches_flann)} matches fortes (de {len(matches_knn)} candidatos).")

    good_matches_flann = sorted(good_matches_flann, key=lambda x: x.distance)
    
    img_flann = cv2.drawMatches(img1, kp1, img2, kp2, good_matches_flann[:50], None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('FLANN + Lowe Ratio - Top 50', img_flann)

    if len(good_matches_flann) >= 4:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches_flann]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches_flann]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        
        inliers_count = int(np.sum(mask))
        print(f"\nHomografia (RANSAC): {inliers_count} inliers (pontos corretos) de {len(good_matches_flann)} filtrados pelo Lowe.")

        if H is not None:
            h, w = img1.shape[:2]
            
            img_alinhada = cv2.warpPerspective(img2, H, (w, h))

            cv2.imshow('Img 2 Alinhada via Homografia (RANSAC)', img_alinhada)
            
            blend = cv2.addWeighted(img1, 0.5, img_alinhada, 0.5, 0)
            cv2.imshow('Blend (Alvo + Alinhada)', blend)
    else:
        print("\nMatches insuficientes para calcular homografia (mínimo de 4).")

    print("\n" + "=" * 70)
    print(" RELEVÂNCIA DA HOMOGRAFIA E RANSAC NA ROBÓTICA / LOCALIZAÇÃO VISUAL")
    print("=" * 70)
    print("1. O teste de Lowe serve para matar ambiguidades iniciais, mas não ")
    print("   garante a integridade física geométrica da cena.")
    print("2. O RANSAC e a Homografia resolvem o problema geométrico: eles ")
    print("   identificam o modelo matemático que relaciona a câmera 1 e a 2.")
    print("3. Ao separar 'Inliers' de 'Outliers', o robô se torna robusto a ")
    print("   objetos móveis bloqueando a visão ou falsos positivos (ex: nuvens ")
    print("   no fundo).")
    print("4. É a base da Odometria Visual (descobrir para onde e quanto o robô ")
    print("   se moveu comparando os frames e montando um mapa em SLAM).")
    print("=" * 70)

    print("\nPressione qualquer tecla nas janelas de imagem para encerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
