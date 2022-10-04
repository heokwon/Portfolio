# Portfolio
### NLP(Natural Language Processing)
* [TextMining with ML](https://github.com/heokwon/Portfolio#textmining-with-ml)(2022.04)
* [Distinguish Click-bait](https://github.com/heokwon/Portfolio#distinguish-click-bait)(2022.05)
* MUSINSA - [Recommending Items](https://github.com/heokwon/Portfolio#musinsa---recommending-items), [Predicting Star-rate Using Review](https://github.com/heokwon/Portfolio#musinsa---predicting-star-rate-using-review)(2022.06~07)
* [Survey-Analysis](https://github.com/heokwon/Portfolio#survey-analysis)(2022.09)
<br><br>
### CV(Computer Vision)
* [DietService-Objectdetection](https://github.com/heokwon/Portfolio#dietservice-objectdetection)(2022.04)
* [Car Damage Detection](https://github.com/heokwon/Portfolio#car-damage-detection)(2022.07~08)
* [Kaggel Competition : HuBMAP + HPA - Hacking The Human Body](https://github.com/heokwon/Portfolio#kaggle-competition--hubmap--hpa---hacking-the-human-body)(2022.08~09)
<br><br>
## TextMining with ML
### [Repositories](https://github.com/heokwon/Portfolio#survey-analysis)
### Introduction
* 우크라사태가 한국 경제에 미치는 영향을 파악하기 위해 웹 크롤링한 신문기사에 워드 클라우드, 토픽 모델링 기법을 활용
* 로지스틱 리그레션을 활용해 우크라사태 이전/이후 신문기사 내용의 감성 분석을 진행
<br><br>
### Data and Models
* 네이버 웹툰 장르별 회차별 독자 참여 수치 19622건
* 우크라사태 이전/이후 신문기사 제목 및 본문 2912건
* TF-IDF / LDA / Logistic Regression
<br><br>
### Envs and Requirements
* Google Colab, VScode
* BeautifulSoup, Pandas, RE, KoNLPy, Scikit-Learn, Gensim, Matplotlib, Seaborn
<br><br>
### Progress
* 웹 크롤링 및 텍스트 데이터 정제
* 데이터 분석 및 시각화
* 토큰화 및 코퍼스 사전 구축
* 불용어 사전 및 사용자 사전 구축
* TF-IDF 워드 클라우드
* LDA 토픽 모델링
* 바이그램 / 트라이그램 적용
* 응집도 / 복잡도 기준 최적 에폭 및 토픽 갯수 설정
* 로지스틱 리그레션을 활용한 감성 분석
<br><br>
### Reference
* 텍스트마이닝을 위한 한국어 불용어 목록 연구 - 길호현
* https://www.crummy.com/software/BeautifulSoup/
* https://pandas.pydata.org/
* https://konlpy.org/
* https://radimrehurek.com/gensim/
* https://scikit-learn.org/
* https://www.pythoncheatsheet.org/   

[Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## Distinguish Click-bait
### [Repositories](https://github.com/heokwon/Distinguish_Click-bait.git)
### Introduction
* 기사와 내용일 일치하지않은, 클릭을 유도하는 피시성 기사를 가리기 위해 진행한 프로젝트
* 딥러닝을 이용해 기사의 내용을 간략하게 요약하고, 헤드라인과 비교
<br><br>
### Data and Models
* 데이터 종류 : 신문기사
* 데이터 형태 : 뉴스 택스트
* 수량 : 원문데이터 30만건
* 출처 : [Ai-Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)
* model : mT5
<br><br>
### Envs and Requirements
* Google Colab, VScode
* Tensorflow, Hugging Face, Transformers, Pandas
<br><br>
### Progress
* 데이터셋 구축
* 데이터셋 정제
* json -> dictionary
* 데이터셋에서 제목, 원문, 추출요약, 생성요약 추출
* transform AutoTokenizer 사용
* Hugging Face 이용하여 모델 구축 - mT5 모델에 fine-tuning
<br><br>
### Referece
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## MUSINSA - Recommending Items
### [Repositories]()
### Introduction
* 데이터 분석을 통한 다음분기 상품 추천
<br><br>
### Data and Models
* Web Crawling을 통한 무신사 데이터
* TF-IDF, Logistic Regression
<br><br>
### Envs and Requirements
* Google Colab, VScode
* BeautifulSoup, Pandas, RE, KoNLPy, Scikit-Learn, Matplotlib, Seaborn
<br><br>
### Progress
<br><br>
### Referece
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## MUSINSA - Predicting Star-rate Using Review
### [Repositories](https://github.com/heokwon/NLP-MUSINSA/tree/main/Predicting%20Star-Rate%20Using%20Review)
### Introduction
* 신뢰성을 가진 별점 예측을 통해 musinsa 입점브랜드에 관한 실질적 평가지표를 제시
<br><br>
### Data and Models
* Web Crawling을 통한 무신사 댓글
* Web Crawling을 통한 네이버 쇼핑몰 부정댓글
* KoGPT2, KoBERT
<br><br>
### Envs and Requirements
* Google Colab, VScode
* BeautifulSoup, Pandas, Hugging Face, Transformers
<br><br>
### Progress
<br><br>
### Referece
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## Survey-Analysis
### [Repositories](https://github.com/heokwon/Survey-analysis.git)
### Introduction
* 설문조사 분석
* 데이터를 쉽게 해석하기 위한 시각화 작업
* 데이터 분석을 위한 토픽모델링 및 LDA
<br><br>
### Data and Models
* 농협 설문조사 자료 - 외주
* TF-IDF, N-gram, LDA
<br><br>
### Envs and Requirements
* Google Colab
* Pandas, Matplotlib, Seaborn, Gensim, WordCloud, Re, KoNLPy
<br><br>
### Progress
<br><br>
### Referece
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## DietService-ObjectDetection
### [Repositories](https://github.com/heokwon/DietService-ObjectDetection.git)
### Introduction
* 현재 자기관리가 하나의 트랜드처럼 자리잡고있음

* 운동만큼 식단의 중요성이 부각됨

* 핸드폰 카메라 하나만으로 식단관리가 가능한 서비스가 있으면 어떨까 하는 생각에 시작한 프로젝트

* Object Detection을 통해 음식을 인식하고 칼로리와 영양정보를 제공
<br><br>
### Data and Models
* YOLO V4

* 12,000개의 한식, 일식, 양식 이미지 데이터

* 총 100개의 class

* multiple food-item, single food-item으로 구성
<br><br>
### Envs and Requirements
* Google Colab, VScode
* Tensorflow, Glob, Shutil, Pillow, OpenCV, Darknet
<br><br>
### Progress
* 데이터셋 구축

* 데이터셋 정제   
이미지 한 폴더에 모으기 - 각각의 클래스 폴더별로 이미지가 나뉘어있음   
이미지 사이즈 정규화 - bbox 좌표 정규화

* Train & Inference
<br><br>
### Referece
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## Car Damage Detection
### [Repositories](https://github.com/heokwon/CV-CarDamageDetection.git)
### Introduction
* Semantic Segmentation을 이용한 자동차 파손부위 탐지   

* 사람이 직접 파손부위를 하나하나 검수해야 하는 부담을 덜 수 있고, 회사 입장에서도   
인적,시간적 자원 절약 측면에서 좋을 것이라 생각하여 진행하게 된 프로젝트

* 사진이나 영상 속 객체를 pixel단위로 탐지하여 object detection보다 세부적으로   
detecting이 가능한 Semantic Segmentation을 선택
<br><br>
### Data and Models
* AI-hub에 socar dataset이 올라오기 이전   

* 구글링을 통하여 segmentation annotation이 포함된 차량파손이미지 수집   

* via tool - 부족한 데이터셋 보충, 좀 더 세밀한 mask를 통해 성능개선을 기대   
차량 파손 이미지 COCOdataset을 사용, via tool을 사용하여 이미지에 polygon을 직접 달아줌으로써   
mask의 좌표를 생성 후 annotation.json 파일 생성   
2명의 팀원이서 1일동안 총 400장의 데이터셋 생성   

* 차량 파손 부위가 아닌 파손 "형태"를 detecting하는 작업으로, 차량 이외에도 스크래치나 이격과   
같은 파손형태 데이터셋도 사용   

* Augmentation - Pytorch의 albumentation을 사용하여 offline으로 데이터증식 진행   
HorizontaFlip, VerticalFlip, Blur, OpticalDistortion, Resize, RandomRotate90

* Binary 와 Multi 로 진행   
Binary Label : background - 0 , damaged - 1   
Multi Label : background - 0 , scratch - 1 , dent - 2 , spacing - 3 , broken - 4
<br><br>
### Envs and Requirements
* Semantic Segmentation에서 가장 많이 쓰이는 모델 선정   

* DeepLabV3   
reference를 git clone하여 하이퍼파라미터 변경 및 inference추가
pre-trained model에 fine-tuning

* Unet   
reference를 git clone하여 하이퍼파라미터 변경 및 inference추가 논문내용을 직접 구현하여 사용   
<br><br>
### Envs and Requirements
* Google Colab, VScode, AWS
* Pytorch, Pillow, OpenCV, Numpy, Matplotlib, via, albumentation, Weights and Biases
<br><br>
### Progress
* 데이터셋 구축 - 구글링, via프로그램사용하여 직접만들기

* 데이터셋 정제   
annotation info가 담겨있는 json파일을 이용하여 polygon2mask진행   
확장자를 jpg에서 png로 바꾸기   
binary형태의 데이터셋에서 class별로 array값을 다르게 부여햐여 multi dataset구축   
unet에서 사용하기 위해 img형식의 mask.png를 array로 바꿔 mask.npy로 변경   
split-folders를 사용하여 폴더안의 파일들을 train-set과 valid-set으로 나눔   

* 데이터셋 증식   
albumentation을 이용하여 오프라인에서 augmentation진행   
HorizontaFlip, VerticalFlip, Blur, OpticalDistortion, Resize, RandomRotate90   

* DeepLabV3 & Unet Reference 찾기

* DeepLabV3 Reference 튜닝, 최적의 hyperparameter찾기, pre-trained 모델에 fine-tuning 시키기

* Weights and Biases를 연동하여 train-log 관리

* Unet 논문 및 유투브 참고하여 직접구현 후 학습 진행

* Label별로 학습시킨 후 ensemble 시도
<br><br>
### Referece
* via tool : https://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.6.html
* DeepLabV3 : https://github.com/msminhas93/DeepLabv3FineTuning.git
* Unet : U-Net: Convolutional Networks for Biomedical Image Segmentation
* albumentation : https://albumentations.ai/
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
## Kaggle Competition : HuBMAP + HPA - Hacking The Human Body
### Repositories - [Train & Inference](https://github.com/heokwon/KaggleCompetiton-Train-and-Inference.git), [Data Handling](https://github.com/heokwon/KaggleCompetiton-DataHandling.git)
### Introduction
* Semantic Segmentation으로 HuBMAP 데이터셋을 학습하여 FTU를 찾는 대회
* 점수를 높이기 위한 Dataset Handling
* encoder를 EfficientNet과 ResNeSt를 사용하는 Unet의 Train
* 학습시킨 데이터셋의 size, encoder, model간의 ensemble과 학습한 모델을 Inference
<br><br>
### Data and Models
* HPA에서 제공한 3000x3000 size의 train image 351장
* EfficientNet (b1 - b5), ResNeSt (101, 200, 269), Unet
<br><br>
### Envs and Requirements
* Google Colab, VScode, AWS, Jupyter notebook
* Pandas, Pytorch, Fast-Ai, MMSegmentation, Pillow, OpenCV, Imageio, Matplotlib, Rasterio, Sklearn, Weights and Biases,  
<br><br>
### Progress
* mmsegmentation
* Dataset Handling
* Modeling
* Inference tuning
* Ensemble
<br><br>
### Result
* Public Score : 0.78 , Private Score : 0.76 (Final Result)
* Rank : 124 / 1245 teams (90 percentile)
<br><br>
### Referece
* https://www.kaggle.com/code/befunny/hubmap-fast-ai-starter-efficientnet
* https://www.kaggle.com/code/shuheiakahane/inference-hubmap-fast-ai-starter-efficientnet
* https://github.com/twyunting/Laplacian-Pyramids
* https://www.kaggle.com/code/nghihuynh/data-augmentation-laplacian-pyramid-blending
* https://www.kaggle.com/code/alejopaullier/how-to-create-a-coco-dataset
* https://github.com/Mr-TalhaIlyas/Mosaic-Augmentation-for-Segmentation
* https://www.kaggle.com/code/thedevastator/converting-to-256x256
* https://www.kaggle.com/code/e0xextazy/multiclass-dataset-768x768-with-stride-for-mmseg
<br><br>
#### [Back to top](https://github.com/heokwon/Portfolio/blob/main/README.md#portfolio)
***
