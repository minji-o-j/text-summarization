#  ✨ [AI 텍스트 요약 알고리즘 개발 대회](http://aifactory.space/competition/detail/1923)

![image](https://user-images.githubusercontent.com/45448731/146986776-8ecf6abf-d3b5-4075-a39d-20319add55db.png)

---
## 🤟 팀원
|김다영|정민지|최석민|
| :---: | :---: | :---: |
| <a href="https://github.com/keemdy" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/68893924?v=4" width="50%" height="50%"> | <a href="https://github.com/minji-o-j" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/45448731?v=4" width="50%" height="50%">| <a href="https://github.com/RockMiin" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/52374789?v=4" width="50%" height="50%">|

---
## 📚 Data 
- 데이터 자세한 설명은 공개 불가
- test data 및 baseline code: [here](https://github.com/aifactory-team/AFCompetition/tree/main/1923)
---
## 🔍 Solution
### Extractive
- **Pororo** : 45.15 ~ 46.96
  - 유니코드 제거
  - 괄호 안 항목 제거 : 45.15
  
### Abstractive
- **KoBart** : 36.40
- **KoGPT2** : 39.28
- **mt5** : 30.74
- **Pororo**
  - Split 2: 30.81
  - Split 3: 29.95

### Ensemble
- ElasticSearch : 36.72
  - 원본과 가장 유사도가 높은 요약문 선택
  
---
## 🌟 순위
- 5등
![image](https://user-images.githubusercontent.com/45448731/146986972-b5708b1a-fb07-4400-b915-054a45cfb23f.png)

---
## 🔥 느낀점
- **김다영**:
- **정민지**: 결국 Pororo-extractive를 생성모델이 이길 수 없던게 아쉬웠다. 생성모델 결과를 보니 반복, 그리고 깨지는 문장들이 꽤 있었는데 이러한 점 때문에 추출 기반 성능을 이기지 못했던 것 같다. 다음에는 KoBert를 이용한 추출을 시도해보고 싶고, 요약 task가 어렵다는 것을 깨달았다.
- **최석민**: 
