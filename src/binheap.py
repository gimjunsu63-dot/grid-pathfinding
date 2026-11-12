"""Binary min-heap written in plain Python, with the same interface as heapq.

    push(heap, item)   add item
    pop(heap)          remove and return the smallest item

Used only to compare against heapq (whose core is written in C), so the
difference in speed shows how much of a "cell" in Dijkstra / A* is heap work.
Items are compared with <, exactly like heapq, so both heaps pop items in
the same order and the searches expand the same cells.
"""


def push(heap, item):
    heap.append(item)
    i = len(heap) - 1
    while i > 0:                       # sift up
        parent = (i - 1) // 2
        if item < heap[parent]:
            heap[i] = heap[parent]
            i = parent
        else:
            break
    heap[i] = item


def pop(heap):
    last = heap.pop()                  # raises IndexError on empty heap, like heapq
    if not heap:
        return last
    top = heap[0]
    n = len(heap)
    i = 0
    while True:                        # sift down
        child = 2 * i + 1
        if child >= n:
            break
        right = child + 1
        if right < n and heap[right] < heap[child]:
            child = right
        if heap[child] < last:
            heap[i] = heap[child]
            i = child
        else:
            break
    heap[i] = last
    return top
