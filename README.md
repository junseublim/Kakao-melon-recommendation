# 🤖 k-최근접 이웃 알고리즘을 활용한 Melon Playlist Continuation


한양대학교 인공지능 동아리 HAI 추천팀 소속으로 카카오 아레나에서 진행한 Melon Playlist Continuation 대회를 위해 개발한 추천 시스템입니다. 

최종 스코어 0.289624(곡nDCG: 0.269720, 태그nDCG: 0.402414)로 2021 4월 29일 기준 플레이그라운드 리더보드 기준으로 1위, 실제 대회 리더보드 기준으로는 21위에 위치하였습니다.

# 개요

## 대회 개요

플레이리스트에 수록된 곡과 태그의 절반 또는 전부가 숨겨져 있을 때, 주어지지 않은 곡들과 태그를 예측하는 것을 목표로 하는 대회입니다. 만약 플레이리스트에 들어있는 곡들의 절반을 보여주고, 나머지 숨겨진 절반을 예측할 수 있는 모델을 만든다면, 플레이리스트에 들어있는 곡이 전부 주어졌을 때 이 모델이 해당 플레이리스트와 어울리는 곡들을 추천해 줄 것이라고 기대할 수 있을 것입니다.

대회에서 제공하는 데이터는 아래와 같습니다.

- 플레이리스트 메타데이터
  - 플레이리스트 제목
  - 플레이리스트에 수록된 곡
  - 플레이리스트에 달려있는 태그 목록
  - 플레이리스트 좋아요 수
  - 플레이리스트가 최종 수정된 시각
- 곡 메타데이터
  - 곡 제목
  - 앨범 제목
  - 아티스트명
  - 장르
  - 발매일
- 곡에 대한 Mel-spectrogram

## 모델 개요

이 모델은 [Efficient K-NN for Playlist Continuation (RecSys'18 Challenge)](https://eprints.sztaki.hu/9560/1/Kelen_1_30347064_ny.pdf) 논문을 기반으로 개발하였습니다. 다만 제공되는 데이터가 다르다는 점, 채점 기준이 다르다는 점을 참고하여 모델을 대회 목표에 알맞게 수정하였습니다.

이모델은 K개의 유사한 플레이리스트를 구하고 그 플레이리스트에 수록된 곡에 점수를 부여하여, 가장 점수가 높은 100개의 곡을 추천하는 모델입니다.

### 플레이리스트간의 유사도 구하기

플레이리스트간의 유사도는 **코사인 유사도**를 이용하여 구하였습니다. 코사인 유사도는 내적공간의 두 벡터간 각도의 코사인값을 이용하여 측정된 벡터간의 유사한 정도를 의미하며, -1에서 1의 값을 가집니다. 1에 가까울 수록 유사하다고 판단할 수 있습니다. 

<img Src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fneo4j.com%2Fdocs%2Fgraph-algorithms%2Fcurrent%2Flabs-algorithms%2Fcosine%2F&psig=AOvVaw0LutFK9eqaZO5uIC__gzDt&ust=1619707926771000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCPDKkq-YofACFQAAAAAdAAAAABAD" />

따라서, 두 플레이리스트간의 유사도는 다음과 같이 구할 수 있습니다.




# How To Run
res 폴더 안에 데이터들을 넣어 놓고 진행합니다.<br>
컴퓨터 성능에 따라 소요 시간이 다릅니다만 저는 약 2시간 정도 걸렸습니다.<br>
진행이 완료되면 res 폴더에 answer.json 파일이 생성됩니다.
```
$> pip install -r requirements.txt
$> python knn-recommendation.py
```
