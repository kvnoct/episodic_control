from typing import Any, Dict, List, Tuple
import environment
import networkx as nx
import matplotlib.pyplot as plt

class Edge:
    def __init__(self, length: float, nodes: List['Node']) -> None:
        """Initialize an Edge object.

        Args:
            length (float): length in the edge (buffer).
        """
        self.length: float = length
        # add nodes to the edge with direction (from_node, to_node)
        
        self.connected_nodes = {}

        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2:
                    self.connected_nodes[node1] = node2
                    self.connected_nodes[node2] = node1
                    

class Node:
    def __init__(self, value: Any, intersection_parameters_dic: Dict=None) -> None:
        """Initialize a Node object.

        Args:
            value (Any): Value stored in the node.
        """
        self.value: Any = value
        self.neighbors: Dict[Any, str] = {}  # Dict to hold neighboring nodes based on direction
        self.edges: Dict[str, Edge] = {}  # Dict to hold edges based on direction
        self.intersection = environment.Intersection(name=value, **intersection_parameters_dic)


    def add_neighbor(self, direction: str, node: Any, edge: Edge) -> None:
        """Add a neighboring node and edge in a particular direction.

        Args:
            direction (str): Directional indicator ('N', 'W', 'E', 'S') for the edge.
            node (Any): Neighboring node.
            edge (Edge): Edge connecting the nodes.
        """
        if direction not in self.edges.keys():
            self.neighbors[direction] = node
            self.edges[direction] = edge

    def __hash__(self) -> int:
        """Custom hash function for the Node object."""
        return hash(self.value)
    
    def __eq__(self, __value: object) -> bool:
        """Custom equality function for the Node object."""
        if isinstance(__value, Node):
            return self.value == __value.value
        return False

    def __str__(self) -> str:
        """Custom representation of the Node object."""
        neighbors_info = "\n".join([f"  {direction}: {str(neighbor)}" for direction, neighbor in self.neighbors.items()])
        return f"Node value: {self.value}\nNeighbors:\n{neighbors_info}"

class Graph:
    def __init__(self, intersection_parameter_dic: Dict, opposite_d = {'N': 'S', 'E':'W', 'S':'N', 'W':'E'}) -> None:
        """Initialize a Graph object."""
        self.opposite_d = opposite_d
        self.nodes: Dict[Any, Node] = {}
        self.input_nodes = []  # List of input nodes
        self.non_input_nodes = []
        self.graph_structure = None
        self.intersection_parameter_dic = intersection_parameter_dic

    def set_memory_based(self, is_mem_based: bool):
        for node in self.nodes.values():
            node.intersection.set_memory_based(is_mem_based=is_mem_based)
    

    def reset(self):
        # reset the interections while keeping the memory for next episodes
        for node in self.nodes.values():
            node.intersection.reset()
    
    def add_node(self, value: Any) -> None:
        """Add a node to the graph.

        Args:
            value (Any): Value to be stored in the node.
        """
       
        if value not in self.nodes:       
            if value.startswith('in'):
                self.input_nodes.append(value)
            else:
                self.non_input_nodes.append(value)
            self.nodes[value] = Node(value, intersection_parameters_dic=self.intersection_parameter_dic)
            

    def add_edge(self, from_node: Any, to_node: Any, length: float, dir: str) -> None:
        """Add an edge (buffer) between two nodes.

        Args:
            from_node (Any): Starting node of the edge.
            to_node (Any): Ending node of the edge.
            length (Float): length associated with the edge.
        """

        edge = Edge(length=length, nodes=[from_node, to_node])
        self.nodes[from_node].add_neighbor(dir, to_node, edge)
        self.nodes[to_node].add_neighbor(self.opposite_d[dir], from_node, edge)

    def add_from_dict(self, graph_structure: Dict[Any, Tuple[int, List[Tuple[Any, float, str]]]]) -> None:
        """Add nodes and edges to the graph based on the provided dictionary.

        Args:
            graph_structure (Dict): Structure defining nodes, lengths, and connections.
        """
        self.graph_structure = graph_structure
        for node, (length, connections) in graph_structure.items():
            self.add_node(node)
            for neighbor, neighbor_length, dir in connections:
                    self.add_node(neighbor)
                    self.add_edge(node, neighbor, neighbor_length, dir)

    def draw_graph(self) -> None:
        """Draw the graph using NetworkX and Matplotlib."""
        G = nx.MultiGraph()
        added_edges = set()

        for node in self.nodes.values():
            G.add_node(str(node.value))
            for direction, connected_node in node.neighbors.items():
                edge_key = tuple(sorted([str(node.value), str(connected_node)]))
                edge = node.edges[direction]
                if edge_key not in added_edges:
                    G.add_edge(str(node.value), str(connected_node), length=edge.length)
                    added_edges.add(edge_key)

        pos = nx.kamada_kawai_layout(G)
        # Adjust node positions to create a rectangular grid
        scale = 1.5  # Adjust this scale to increase or decrease edge length
        for node, (x, y) in pos.items():
            pos[node] = (scale * x, scale * y)

        labels = {str(node.value): str(node.value) for node in self.nodes.values()}
        edge_labels = {(u, v): str(data['length']) for u, v, data in G.edges(data=True)}

        nx.draw(G, pos, with_labels=True, labels=labels, node_size=800, node_color='skyblue', font_weight='bold', arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Graph Visualization")
        plt.show()


    def get_rectangular_grid_position(self):
        connected_nodes = set()
        for node, (_, neighbors) in self.graph_structure.items():
            connected_nodes.add(node)
            for neighbor, _, _ in neighbors:
                connected_nodes.add(neighbor)

        num_nodes = len(connected_nodes)
        num_cols = int(num_nodes ** 0.5) + 1

        # Get the first node in the graph_structure and assign it to (1, 1)
        first_node = next(iter(self.graph_structure.keys()))
        node_positions = {first_node: (1, 1)}

        # Assign positions based on cardinal directions (North, East, West, South) from the initial node
        directions = {'N': (0, 1), 'E': (1, 0), 'W': (-1, 0), 'S': (0, -1)}
        assigned_nodes = set([first_node])

        for node, (num_neighbors, neighbors) in self.graph_structure.items():
            if node in node_positions:
                x, y = node_positions[node]
                for neighbor, _, direction in neighbors:
                    if neighbor not in assigned_nodes:
                        dx, dy = directions[direction]
                        node_positions[neighbor] = (x + dx, y + dy)
                        assigned_nodes.add(neighbor)
        return node_positions

    
    def draw_rectangular_grid(self):
        
        node_positions = self.get_rectangular_grid_position()
        # Plotting the graph in a rectangular grid
        plt.figure(figsize=(8, 6))

        
        # Draw edges
        for node, (_, neighbors) in self.graph_structure.items():
            if node in node_positions:
                x1, y1 = node_positions[node]
                for neighbor, length, _ in neighbors:
                    if neighbor in node_positions:
                        x2, y2 = node_positions[neighbor]
                        plt.plot([x1, x2], [y1, y2], linestyle='-', color='black', zorder=1)
                        if node.startswith('in') or neighbor.startswith('in'):
                            continue
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        plt.text(mid_x+0.05, mid_y+0.06, str(length), ha='center', va='center', fontsize=10, fontweight='bold', zorder=2)

        # Draw nodes
        for node, pos in node_positions.items():
            if node.startswith('in'):
                continue
            plt.scatter(pos[0], pos[1], s=500, color='skyblue', edgecolors='black', zorder=2)
            plt.text(pos[0], pos[1], node, ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)
        
        plt.title('Graph Laid Out on a Rectangular Grid')
        plt.axis('off')  # Hide the axis
        plt.grid(visible=True)
        plt.show()


    def draw_graph_2(self) -> None:
        """Draw the graph using NetworkX and Matplotlib in a rectangular grid layout."""
        G = nx.MultiGraph()
        added_edges = set()
        pos = self.get_rectangular_grid_position()

        for node in self.nodes.values():
            G.add_node(str(node.value))

        for node in self.nodes.values():
            for direction, connected_node in node.neighbors.items():
                edge_key = tuple(sorted([str(node.value), str(connected_node)]))
                edge = node.edges[direction]
                if edge_key not in added_edges:
                    G.add_edge(str(node.value), str(connected_node), length=edge.length)
                    added_edges.add(edge_key)

        labels = {str(node.value): str(node.value) for node in self.nodes.values()}
        edge_labels = {(u, v): str(data['length']) for u, v, data in G.edges(data=True)}
        
        
        plt.figure(figsize=(10, 8))
        nx.draw_networkx(G, pos, with_labels=True, labels=labels, node_size=800, node_color='skyblue', font_weight='bold', arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Graph Visualization - Rectangular Grid Layout")
        plt.show()
