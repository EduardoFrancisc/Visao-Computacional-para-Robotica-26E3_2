# Relatório Técnico: Pipeline de Percepção Visual para Veículos Autônomos

Este relatório descreve a estruturação e o funcionamento de um pipeline clássico de percepção visual para veículos autônomos e drones, baseando-se nas técnicas de visão computacional implementadas nos exercícios recentes. O objetivo é mapear como operações fundamentais se integram para extrair informações semânticas do ambiente e guiar decisões de navegação.

---

## 1. Diagrama Arquitetural do Pipeline

O fluxo de processamento de imagens pode ser abstraído através do seguinte diagrama em formato ASCII, que ilustra o encadeamento das etapas desde a captura do dado bruto até a extração da métrica de navegação:

```text
[ Sensor de Câmera ]
        |
        v
+---------------------------------------------------+
| 1. Captura de Frame (BGR)                         | -> Coleta de dados brutos do ambiente
+---------------------------------------------------+
        |
        v
+---------------------------------------------------+
| 2. Pré-processamento e Conversão de Cor (HSV)     | -> Isolamento de canais de cor e brilho
+---------------------------------------------------+
        |
        v
+---------------------------------------------------+
| 3. Limiarização (Máscara / Otsu / Adaptativa)     | -> Binarização e separação Fundo vs Objeto
+---------------------------------------------------+
        |
        v
+---------------------------------------------------+
| 4. Filtragem Morfológica (Erosão / Dilatação)     | -> Redução de ruídos pontuais e falhas
+---------------------------------------------------+
        |
        v
+---------------------------------------------------+
| 5. Extração de Contornos (FindContours)           | -> Análise geométrica (Área, Bounding Box)
+---------------------------------------------------+
        |
        v
+---------------------------------------------------+
| 6. Cálculo de Navegação (Centroide e Desvio)      | -> Cálculo de rota e atuadores
+---------------------------------------------------+
        |
        v
[ Sistema de Controle do Robô ]

```

---

## 2. Encadeamento das Técnicas

Em um cenário prático (como o de um drone seguindo uma marcação no solo), as técnicas isoladas formam uma cadeia de dependência estrita.

A **captura** fornece a matriz de pixels original, que é imediatamente transformada através da **conversão de cor** (como BGR para HSV). O espaço HSV é crucial porque isola a informação de luminosidade (V) da cor (H), permitindo que a **limiarização** defina uma região de interesse baseada puramente na pigmentação do objeto, ignorando sombras leves. Com a imagem binarizada (fundo preto, objeto branco), algoritmos de busca de **contornos** conseguem mapear os limites do alvo. A partir deste contorno, calcula-se a área (para estimar proximidade) e o centroide (para calcular o erro posicional lateral e comandar os motores de direção).

---

## 3. Limitações em Condições Reais

Apesar de funcional em ambientes controlados, este pipeline clássico possui falhas severas quando exposto ao mundo real:

* **Iluminação Variável:** Limiares de cor (ranges HSV) fixos ou globais falham drasticamente sob luz solar direta, reflexos especulares em superfícies metálicas ou ao entrar em túneis. A limiarização adaptativa mitiga o problema, mas não resolve distorções extremas de cor.
* **Oclusão:** Se um pedestre ou árvore passar na frente do objeto rastreado, a máscara binária é dividida ou desaparece. O algoritmo atual de busca por maior área perderá o alvo ou calculará um centroide completamente distorcido, causando desvios bruscos no robô.
* **Velocidade e Movimento:** Drones e carros operam em altas velocidades, introduzindo *motion blur* (borrão de movimento) nas imagens capturadas. Isso dilui as bordas dos objetos e mescla as cores, impedindo que a limiarização e o detector Canny identifiquem contornos e formas com precisão.

---

## 4. Evolução das Competências (TP2 em Diante)

Para solucionar os problemas inerentes a um pipeline puramente baseado em processamento clássico de imagens, as competências das próximas etapas e do **Trabalho Prático 2 (TP2)** introduzirão novos paradigmas:

* **Modelos de Deep Learning (Redes Neurais Convolucionais - CNNs):** Ao invés de depender de limiares manuais de cor (HSV) extremamente frágeis à iluminação, as CNNs (como YOLO ou SSD) aprendem a extrair características semânticas complexas dos objetos, garantindo detecção robusta independentemente da cor, iluminação ou escala.
* **Algoritmos de Rastreamento (Tracking) e Fluxo Óptico:** Para lidar com a oclusão temporal, técnicas de rastreamento baseadas em correlação e filtros preditivos (como o Filtro de Kalman) permitem que o veículo estime a posição do objeto mesmo quando ele fica parcial ou totalmente escondido por alguns frames, evitando perdas de controle repentinas.
* **Calibração e Visão Estéreo:** Em contraste com a simples contagem de área de contornos em 2D para estimar distância, abordagens mais sofisticadas introduzirão a calibração intrínseca da câmera e o uso de múltiplos sensores para mapeamento tridimensional e cálculos de profundidade precisos (fundamentais para veículos operando em alta velocidade).