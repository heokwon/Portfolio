# Portfolio
### NLP(Natural Language Processing)
* [TextMining with ML](https://github.com/heokwon/Portfolio#textmining-with-ml)(2022.04)
* [Distinguish Click-bait](https://github.com/heokwon/Portfolio#distinguish-click-bait)(2022.05)
* MUSINSA - [Recommending Items](https://github.com/heokwon/Portfolio#musinsa---recommending-items), [Predicting Star-rate Using Review](https://github.com/heokwon/Portfolio#musinsa---predicting-star-rate-using-review)(2022.06~07)
* [Survey-Analysis](https://github.com/heokwon/Portfolio#survey-analysis)(2022.09)
<br><br>
### CV(Computer Vision)
* [DietService-Objectdetection](https://github.com/heokwon/DietService-ObjectDetection.git)(2022.04)
* [Car Damage Detection](https://github.com/heokwon/CV-CarDamageDetection.git)(2022.07~08)
* Kaggel Competition : HuBMAP + HPA - Hacking The Human Body-[Train & Inference](https://github.com/heokwon/KaggleCompetiton-Train-and-Inference.git),[Data Handling](https://github.com/heokwon/KaggleCompetiton-DataHandling.git)(2022.08~09)
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
<br><br>
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
<br><br>
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
