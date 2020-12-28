# Kakao-melon-recommendation

카카오 아레나에서 진행한 Melon Playlist Continuation 대회를 위해 개발한 추천 시스템입니다. 최종 스코어 0.289624(곡nDCG: 0.269720, 태그nDCG: 0.402414)로 2020년 12월 29일 기준 플레이그라운드 리더보드 기준으로 1위, 실제 대회 리더보드 기준으로는 21위에 위치하였습니다.

KNN 최근접 이웃 알고리즘을 활용하여 가장 유사한 플레이리스트 1500개를 찾고, 각 플레이리스트에 존재하는 곡, 태그에 플레이리스트 유사도에 따른 점수를 부여하여 최종적으로 100개의 곡과 10개의 태그를 추천하는 시스템입니다. [Efficient K-NN for Playlist Continuation (RecSys'18 Challenge)](https://eprints.sztaki.hu/9560/1/Kelen_1_30347064_ny.pdf) 논문을 기반으로 개발하였습니다. 
[dingdong_demo](https://www.youtube.com/watch?v=3OvvbV-6EnE&t=188s)

# How To Run
res 폴더 안에 데이터들을 넣어 놓고 진행합니다.<br>
컴퓨터 성능에 따라 소요 시간이 다릅니다만 저는 약 2시간 정도 걸렸습니다.<br>
진행이 완료되면 res 폴더에 answer.json 파일이 생성됩니다.
```
$> pip install -r requirements.txt
$> python knn-recommendation.py
```
