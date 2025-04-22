# Re-execute everything with corrected arrow direction: child → parent,
# so arrows go from top (leaf) to bottom (root), no y-inversion needed.

import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Base directory
base_dir = Path("/Users/hongjixu/Documents/repos/paperbench/project/paperbench/runs")
output_dir = base_dir / "images"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all pb_result.json files in the runs directory and its subdirectories
result_files = list(base_dir.glob("**/*pb_result.json"))
print(f"Found {len(result_files)} result files")

for file_path in result_files:
    print(f"Processing {file_path}")
    
    try:
        # Extract meaningful identifiers from the path
        # Get the model name from parent directory
        model_name = file_path.parent.name.split('_')[0]  # Assuming format like "rice_92f7c3ca-..."
        
        # Get the agent type from grandparent directory 
        agent_match = ""
        parent_parts = file_path.parent.parent.name.split('_')
        for part in parent_parts:
            if any(x in part.lower() for x in ["agent", "gemini", "gpt", "claude"]):
                agent_match = part
                break
                
        # Fall back to a substring if needed
        if not agent_match and len(parent_parts) > 3:
            agent_match = parent_parts[3]
            
        # Create a concise, standardized output filename
        output_filename = f"{model_name}_{agent_match}_tree.png"
        output_path = output_dir / output_filename
        
        # Load JSON data
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build reversed graph: edges go from child -> parent
        G = nx.DiGraph()

        def build_tree_reversed(task, graph, parent_id=None, node_prefix=""):
            node_id = f"{node_prefix}_{task['id']}" if "id" in task else f"{node_prefix}_root"
            score = task.get("score", None)
            graph.add_node(node_id, score=score)

            if parent_id is not None:
                graph.add_edge(node_id, parent_id)  # reversed: child → parent

            for i, sub_task in enumerate(task.get("sub_tasks", [])):
                build_tree_reversed(sub_task, graph, node_id, f"{node_prefix}_{i}")

        root_task = data["paperbench_result"]["judge_output"]["graded_task_tree"]
        build_tree_reversed(root_task, G)

        # Extract scores and map to colors
        scores = nx.get_node_attributes(G, "score")
        colors = ["#ff0000", "#ffff66", "#00cc44"]
        custom_cmap = LinearSegmentedColormap.from_list("red_to_green", colors, N=256)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        node_colors = [custom_cmap(norm(scores.get(n, 0))) for n in G.nodes]

        # Use hierarchical_layout instead of pygraphviz
        pos = nx.nx_pydot.pydot_layout(G, prog='dot')

        # If pydot is not available, fallback to a simple layout
        if not pos:
            # First try spring layout with parameters to make it look like a tree
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            
            # Get root node (should be the one with no outgoing edges in our reversed graph)
            root_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
            if root_nodes:
                root_node = root_nodes[0]
                # Compute levels (distances from root)
                levels = nx.shortest_path_length(G, target=root_node)
                
                # Adjust y-coordinate based on level (higher level = higher y-value)
                pos_copy = dict(pos)  # Make a copy of the position dictionary
                for node, level in levels.items():
                    pos_copy[node] = (pos[node][0], -level)  # Negative to have root at bottom
                pos = pos_copy  # Replace with the modified copy

        # Create figure and axis for the plot
        fig, ax = plt.subplots(figsize=(26, 16))

        # Draw final corrected tree
        nx.draw(G, pos, ax=ax, with_labels=False, node_color=node_colors, node_size=300, arrows=True)

        # Identify root node (the one with no outgoing edges in our reversed graph)
        root_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
        if root_nodes:
            root_node = root_nodes[0]
            rx, ry = pos[root_node]
            final_score = scores.get(root_node, 0)
            plt.text(rx, ry - 40, f"FINAL NODE Score = {final_score:.2f}",
                    fontsize=14, ha='center', bbox=dict(facecolor='white', edgecolor='black'))

        # Add legend and title
        gradient_legend = plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
        cbar = fig.colorbar(gradient_legend, ax=ax, orientation='vertical', fraction=0.03, pad=0.05)
        cbar.set_label('Score (0 to 1)', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # Add descriptive title
        ax.set_title(f"Score Tree: {model_name} ({agent_match}) - Final Score: {final_score:.2f}", fontsize=18, pad=20)
        ax.axis("off")

        # Save final image
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Saved visualization to {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("Completed generating all visualizations")
