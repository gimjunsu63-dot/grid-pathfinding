# 실험 데이터

## songdo_roads.json

송도국제도시(인천 연수구) 일대의 자동차 도로망입니다. 실험 RM(실제 도로지도)이 이 파일을 읽습니다.

- 출처: [OpenStreetMap](https://www.openstreetmap.org/) (Overpass API로 한 번 받아 저장)
- 라이선스: © OpenStreetMap contributors, [ODbL 1.0](https://www.openstreetmap.org/copyright)
- 범위: 위도 37.355~37.415, 경도 126.615~126.700
- 받은 방법: `python3 tools/fetch_osm.py` (받은 날짜와 지도 기준 시각은 파일 안 `downloaded_utc`, `osm_base`에 기록)
- 담긴 것: 도로(way)의 종류(`highway`), 일방통행(`oneway`), 제한 속도(`maxspeed`), 도로를 이루는 점들의 좌표. 개인정보는 없습니다.

OpenStreetMap은 계속 바뀌므로 다시 받으면 조금 다른 지도가 됩니다. 결과를 재현할 때는 저장소에 있는 이 파일을 그대로 씁니다.
