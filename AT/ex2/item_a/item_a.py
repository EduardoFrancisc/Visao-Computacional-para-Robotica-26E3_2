import os
import psutil
import cv2
import csv
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from dnn_utils import list_classification_images, get_opencv_net, predict_opencv, topk, draw_top3, OUT, time_ms, ROOT, DATA_CLASS, load_labels

paths = list_classification_images()[:10]

print("--- Medindo OpenCV DNN ---")
process = psutil.Process(os.getpid())
mem_antes_ocv = process.memory_info().rss / (1024*1024)

net = get_opencv_net()

# Classificação e salvamento de imagens
for p in paths:
    img = cv2.imread(str(p))
    prob = predict_opencv(img, net)
    vis = draw_top3(img, topk(prob, 3))
    out = OUT / f'top3_{p.stem}.jpg'
    cv2.imwrite(str(out), vis)
    print('salvo:', out.name)

#Latência OpenCV DNN 
img = cv2.imread(str(list_classification_images()[0]))
media_ocv, desvio_ocv = time_ms(lambda: predict_opencv(img, net), loops=20, warmup=5)
lat_ocv = f'{media_ocv:.2f} ± {desvio_ocv:.2f} ms'

#Latência Keras
model = MobileNetV2(weights='imagenet')
rgb = cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB)
x = preprocess_input(rgb.astype('float32'))[None, ...]
media_k, desvio_k = time_ms(lambda: model.predict(x, verbose=0), loops=20, warmup=5)
lat_k = f'{media_k:.2f} ± {desvio_k:.2f} ms'

#Memória Opencv
process_ocv = psutil.Process(os.getpid())
mem_antes_ocv = process_ocv.memory_info().rss / (1024*1024)
_ = predict_opencv(img, net)
mem_depois_ocv = process_ocv.memory_info().rss / (1024*1024)
aumento_aprox_ocv = f'{mem_depois_ocv - mem_antes_ocv:.1f} MB' 

#Memória keras
process_k = psutil.Process(os.getpid())
mem_antes_k = process_k.memory_info().rss / (1024*1024)
rgb = cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB)
x = preprocess_input(rgb.astype('float32'))[None, ...]
_ = model.predict(x, verbose=0)
mem_depois_k = process_k.memory_info().rss / (1024*1024)
aumento_aprox_k = f'{mem_depois_k - mem_antes_k:.1f} MB'

#Acurácia
csv_path = ROOT / 'data' / 'labels_top1.csv'
labels = load_labels()
corretos_ocv = 0
corretos_k = 0
total = 0

if csv_path.exists():
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected = row['label_esperado'].strip().lower()
            if not expected:
                continue
            img_path = DATA_CLASS / row['arquivo']
            img_acc = cv2.imread(str(img_path))
            if img_acc is None:
                continue
            
            # OpenCV DNN
            prob_ocv = predict_opencv(img_acc, net)
            pred_idx_ocv = topk(prob_ocv, 1)[0][0]
            pred_label_ocv = labels[pred_idx_ocv].lower().replace(' ', '_')
            if expected in pred_label_ocv or pred_label_ocv in expected:
                corretos_ocv += 1
                
            # Keras
            rgb_k = cv2.cvtColor(cv2.resize(img_acc, (224,224)), cv2.COLOR_BGR2RGB)
            x_k = preprocess_input(rgb_k.astype('float32'))[None, ...]
            prob_k = model.predict(x_k, verbose=0)[0]
            pred_idx_k = prob_k.argmax()
            pred_label_k = labels[pred_idx_k].lower().replace(' ', '_')
            if expected in pred_label_k or pred_label_k in expected:
                corretos_k += 1
                
            total += 1

acc_ocv = (corretos_ocv / total) if total else 0.0
acc_k = (corretos_k / total) if total else 0.0

dados = [
    {'backend':'OpenCV DNN', 'latencia_ms':lat_ocv, 'memoria_MB':aumento_aprox_ocv, 'top1_acc':f'{acc_ocv}%'},
    {'backend':'Keras',      'latencia_ms':lat_k, 'memoria_MB':aumento_aprox_k, 'top1_acc':f'{acc_k}%'},
]

df = pd.DataFrame(dados)
print(df.to_string(index=False))
print('OpenCV DNN tende a ser preferível quando o objetivo é inferência leve,')
print('integração direta com pipeline OpenCV e menor dependência de frameworks completos.')
print('Keras é preferível para treinamento, ajuste fino, experimentação e validação de modelos.')