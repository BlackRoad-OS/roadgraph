"""
RoadGraph - Graph Data Structures for BlackRoad
Graph representation with algorithms and traversal.
"""

from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple
import heapq
import logging

logger = logging.getLogger(__name__)


class GraphType(str, Enum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"


@dataclass
class Node:
    id: str
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    weight: float = 1.0
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Path:
    nodes: List[str]
    edges: List[Edge]
    total_weight: float = 0.0

    def __len__(self) -> int:
        return len(self.nodes)


class Graph:
    def __init__(self, graph_type: GraphType = GraphType.UNDIRECTED):
        self.graph_type = graph_type
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Dict[str, Edge]] = defaultdict(dict)

    def add_node(self, node_id: str, data: Any = None, **metadata) -> Node:
        node = Node(id=node_id, data=data, metadata=metadata)
        self.nodes[node_id] = node
        return node

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        if node_id in self.edges:
            del self.edges[node_id]
        for source in list(self.edges.keys()):
            if node_id in self.edges[source]:
                del self.edges[source][node_id]
        return True

    def add_edge(self, source: str, target: str, weight: float = 1.0, **kwargs) -> Edge:
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
        
        edge = Edge(source=source, target=target, weight=weight, **kwargs)
        self.edges[source][target] = edge
        
        if self.graph_type == GraphType.UNDIRECTED:
            reverse = Edge(source=target, target=source, weight=weight, **kwargs)
            self.edges[target][source] = reverse
        
        return edge

    def remove_edge(self, source: str, target: str) -> bool:
        if source not in self.edges or target not in self.edges[source]:
            return False
        del self.edges[source][target]
        if self.graph_type == GraphType.UNDIRECTED:
            del self.edges[target][source]
        return True

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        return self.edges.get(source, {}).get(target)

    def neighbors(self, node_id: str) -> List[str]:
        return list(self.edges.get(node_id, {}).keys())

    def degree(self, node_id: str) -> int:
        if self.graph_type == GraphType.UNDIRECTED:
            return len(self.edges.get(node_id, {}))
        in_deg = sum(1 for s in self.edges if node_id in self.edges[s])
        out_deg = len(self.edges.get(node_id, {}))
        return in_deg + out_deg

    def bfs(self, start: str) -> Generator[str, None, None]:
        visited = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            yield node
            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)

    def dfs(self, start: str) -> Generator[str, None, None]:
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            yield node
            for neighbor in reversed(self.neighbors(node)):
                if neighbor not in visited:
                    stack.append(neighbor)

    def shortest_path(self, start: str, end: str) -> Optional[Path]:
        if start not in self.nodes or end not in self.nodes:
            return None
        
        distances = {start: 0}
        previous = {start: None}
        pq = [(0, start)]
        visited = set()
        
        while pq:
            dist, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            
            if node == end:
                break
            
            for neighbor, edge in self.edges.get(node, {}).items():
                new_dist = dist + edge.weight
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = node
                    heapq.heappush(pq, (new_dist, neighbor))
        
        if end not in previous:
            return None
        
        path_nodes = []
        current = end
        while current is not None:
            path_nodes.append(current)
            current = previous[current]
        path_nodes.reverse()
        
        edges = []
        for i in range(len(path_nodes) - 1):
            edges.append(self.edges[path_nodes[i]][path_nodes[i + 1]])
        
        return Path(nodes=path_nodes, edges=edges, total_weight=distances[end])

    def connected_components(self) -> List[Set[str]]:
        visited = set()
        components = []
        
        for node in self.nodes:
            if node not in visited:
                component = set()
                for n in self.bfs(node):
                    component.add(n)
                    visited.add(n)
                components.append(component)
        
        return components

    def is_connected(self) -> bool:
        if not self.nodes:
            return True
        start = next(iter(self.nodes))
        visited = set(self.bfs(start))
        return len(visited) == len(self.nodes)

    def has_cycle(self) -> bool:
        visited = set()
        rec_stack = set()
        
        def dfs_cycle(node: str, parent: str = None) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    if dfs_cycle(neighbor, node):
                        return True
                elif self.graph_type == GraphType.DIRECTED:
                    if neighbor in rec_stack:
                        return True
                else:
                    if neighbor != parent:
                        return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                if dfs_cycle(node):
                    return True
        return False

    def topological_sort(self) -> Optional[List[str]]:
        if self.graph_type != GraphType.DIRECTED:
            return None
        
        in_degree = {n: 0 for n in self.nodes}
        for source in self.edges:
            for target in self.edges[source]:
                in_degree[target] += 1
        
        queue = deque([n for n in in_degree if in_degree[n] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in self.neighbors(node):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.nodes):
            return None
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.graph_type.value,
            "nodes": [{"id": n.id, "data": n.data} for n in self.nodes.values()],
            "edges": [{"source": e.source, "target": e.target, "weight": e.weight} 
                     for edges in self.edges.values() for e in edges.values()]
        }

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        count = sum(len(e) for e in self.edges.values())
        if self.graph_type == GraphType.UNDIRECTED:
            count //= 2
        return count


def example_usage():
    g = Graph(GraphType.UNDIRECTED)
    
    for city in ["NYC", "LA", "Chicago", "Houston", "Phoenix", "Miami"]:
        g.add_node(city)
    
    g.add_edge("NYC", "Chicago", weight=790)
    g.add_edge("NYC", "Miami", weight=1280)
    g.add_edge("Chicago", "Houston", weight=1090)
    g.add_edge("Houston", "Phoenix", weight=1180)
    g.add_edge("Phoenix", "LA", weight=370)
    g.add_edge("LA", "Chicago", weight=2015)
    
    print(f"Nodes: {g.node_count()}, Edges: {g.edge_count()}")
    print(f"NYC neighbors: {g.neighbors('NYC')}")
    
    path = g.shortest_path("NYC", "LA")
    if path:
        print(f"Shortest NYC->LA: {' -> '.join(path.nodes)} ({path.total_weight} miles)")
    
    print(f"BFS from NYC: {list(g.bfs('NYC'))}")
    print(f"Connected: {g.is_connected()}")
    print(f"Has cycle: {g.has_cycle()}")

