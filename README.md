# 격자 지도 최단 경로 탐색 비교 (BFS · 다익스트라 · A*)

[![test](https://github.com/gimjunsu63-dot/grid-pathfinding/actions/workflows/test.yml/badge.svg)](https://github.com/gimjunsu63-dot/grid-pathfinding/actions/workflows/test.yml)

2026-2 고급알고리즘 개인프로젝트 · 2026193209 김준수

장애물이 있는 2차원 격자 지도에서 출발 칸부터 도착 칸까지 비용이 가장 작은 경로를 찾는
세 방식(BFS, 다익스트라, A*)을 같은 조건에서 구현하고 비교합니다.

- 저장소는 [`algorithm-env`](https://github.com/lec-algorithm/algorithm-env) 템플릿으로 만들었습니다.
- Python 표준 모듈만 씁니다 (외부 라이브러리 없음).

## 실행 방법

Codespaces나 `docker compose exec lab bash`로 컨테이너에 들어간 뒤:

```sh
make run    # 작은 예제 지도 3개에서 세 알고리즘 실행
make test   # 유닛 테스트
```

컨테이너 없이도 Python 3.11 이상이면 같은 명령으로 돌아갑니다.

## 구조

```plaintext
├── src/
│   ├── grid.py      # 격자 표현, 이웃 칸, 경로 복원, 경로 검사 (세 알고리즘 공용)
│   ├── search.py    # bfs, dijkstra, astar (다익스트라와 A*는 같은 함수 best_first)
│   └── main.py      # 예제 실행
├── tests/
│   └── test_search.py
├── docs/
│   └── log.md       # 실험 노트
└── .github/workflows/test.yml   # 커밋마다 테스트 자동 실행
```

## 구현 규칙

비교가 공정하도록 세 알고리즘이 다음을 똑같이 따릅니다.

| 항목 | 규칙 |
| --- | --- |
| 입력 | `(grid, start, goal)`, 위치는 `(행, 열)` |
| 출력 | `SearchResult(path, cost, expanded, stale, max_frontier)` |
| 이동 | 상하좌우 4방향, 이웃 칸 순서 고정 (위, 아래, 왼쪽, 오른쪽) |
| 비용 | 들어가는 칸의 비용을 더함 (출발 칸 비용은 안 냄), 장애물은 0 |
| 종료 | 도착 칸을 큐/힙에서 **꺼낼 때** 종료 |
| 다익스트라 | A*에서 h = 0인 경우. 코드 하나(`best_first`)를 같이 씀 |
| A* 휴리스틱 | 맨해튼 거리. 칸 비용이 1 이상이라 실제 비용보다 크지 않으므로 최적 보장 |
| A* 동점 처리 | 기본은 f가 같으면 g가 큰 칸 먼저 (`tie_break` 인자로 바꿀 수 있음) |

측정값 정의:

| 이름 | 뜻 |
| --- | --- |
| `expanded` | 큐/힙에서 꺼내서 처리한 칸 수 (이미 확정된 칸의 낡은 항목은 제외) |
| `stale` | 꺼냈지만 낡은 항목이라 버린 수. `heapq`에 값 갱신 기능이 없어 같은 칸을 여러 번 넣기 때문 (BFS는 항상 0) |
| `max_frontier` | 탐색 중 큐/힙 크기의 최댓값 (낡은 항목 포함, 실제로 차지한 메모리 기준) |

## 테스트

`make test`가 확인하는 것:

1. 손으로 답을 아는 작은 지도 (막힌 길, 출발=도착, 출발/도착이 벽, 경로 없음, 비용 때문에 BFS가 틀리는 지도)
2. 랜덤 지도 300개에서 다익스트라·A* 비용을 **별도로 짠 참조 구현**(단순 반복 완화)과 비교
3. 랜덤 지도에서 세 알고리즘끼리 일치하는지
   - 비용이 모두 1이면 BFS 비용 = 최적 비용
   - 비용이 다르면 BFS 비용 ≥ 최적 비용
   - 경로가 없으면 세 알고리즘 모두 도달 가능한 칸을 전부 탐색
   - 동점 처리 방식이 달라도 A* 비용은 같음

## 참고 자료와 AI 사용

- 템플릿: [lec-algorithm/algorithm-env](https://github.com/lec-algorithm/algorithm-env)
- AI 사용 내역은 보고서 8절에 정리합니다.
