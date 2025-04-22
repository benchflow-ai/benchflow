import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.gridspec as gridspec

# Base directory
base_dir = Path("/Users/hongjixu/Documents/repos/paperbench/project/paperbench/runs")
output_dir = base_dir / "images"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all pb_result.json files in the runs directory and its subdirectories
result_files = list(base_dir.glob("**/*pb_result.json"))
print(f"Found {len(result_files)} result files")

# Extract model data and scores
model_data = []

for file_path in result_files:
    try:
        # Extract meaningful identifiers from the path
        model_name = file_path.parent.name.split('_')[0]  # e.g., "rice"
        
        # Get the agent type from grandparent directory 
        agent_match = ""
        parent_dir = file_path.parent.parent
        parent_path = str(parent_dir)
        
        parent_parts = parent_dir.name.split('_')
        for part in parent_parts:
            if any(x in part.lower() for x in ["agent", "gemini", "gpt", "claude", "llama"]):
                agent_match = part
                break
                
        # Fall back to a substring if needed
        if not agent_match and len(parent_parts) > 3:
            agent_match = parent_parts[3]
        
        # Extract the full agent string for file matching
        full_agent = "-".join(parent_dir.name.split('_')[2:])
        
        # Load JSON data
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Get the final score (root task score)
        root_task = data["paperbench_result"]["judge_output"]["graded_task_tree"]
        final_score = root_task.get("score", 0)
        
        # Get model display name based on the directory name
        if "anthropic" in parent_path.lower():
            display_name = "Claude 3.7"
        elif "gpt-4.1" in parent_path.lower():
            display_name = "GPT-4.1"
        elif "gpt-4o" in parent_path.lower() or "gpt4o" in parent_path.lower():
            display_name = "GPT-4o"
        elif "llama" in parent_path.lower():
            display_name = "Llama 4"
        elif "litellm" in parent_path.lower() or "gemini" in parent_path.lower():
            display_name = "Gemini 2.5 pro"
        else:
            # 如果没有匹配到特定模型，显示默认名称
            display_name = "Unknown Model"
            print(f"Could not identify model from path: {parent_path}")
        
        # Store relevant data
        model_info = {
            'agent_type': full_agent,  # Store the full agent string for file matching
            'display_name': display_name,
            'final_score': final_score,
            'file_path': file_path,
            'model_name': model_name
        }
        
        # Extract subtask scores if available
        if "sub_tasks" in root_task:
            for i, subtask in enumerate(root_task.get("sub_tasks", [])):
                name = subtask.get("name", f"Subtask {i}")
                score = subtask.get("score", 0)
                model_info[name] = score
        
        model_data.append(model_info)
        print(f"Extracted data for {display_name}, score: {final_score:.2f}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print(f"Total models identified: {len(model_data)}")

# Check which models from README.md are missing
expected_models = ["Claude 3.7", "GPT-4o", "GPT-4.1", "Llama 4", "Gemini 2.5 pro"]
found_models = [m['display_name'] for m in model_data]
missing_models = [m for m in expected_models if m not in found_models]
if missing_models:
    print(f"WARNING: The following models from README.md are missing: {missing_models}")

# Create comparison visualizations
if model_data:
    # Sort by final score (descending)
    model_data.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Use a single consistent colormap - Blues
    cmap = plt.colormaps['Blues']
    
    # Create consistent colors for each model (darker blue = higher score)
    model_colors = [cmap(0.4 + 0.6 * score) for score in [m['final_score'] for m in model_data]]
    
    # 1. Bar chart of final scores
    plt.figure(figsize=(12, 8))
    labels = [m['display_name'] for m in model_data]
    scores = [m['final_score'] for m in model_data]
    
    bars = plt.bar(labels, scores, color=model_colors)
    plt.ylim(0, 1)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Final Score', fontsize=14)
    plt.title('Comparison of Final Scores Across Models', fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add score labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "score_comparison.png")
    plt.close()
    
    # 2. Grid of model trees - exactly 2x3 (to fit all 5 models plus empty space)
    num_models = len(model_data)
    rows = 2
    cols = 3
    
    fig = plt.figure(figsize=(22, 15))
    
    # Create a 2x3 grid 
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    
    # Add model thumbnail images from the tree visualizations
    for i, model in enumerate(model_data):
        if i >= rows * cols:  # Should include all 5 models
            break
            
        row_idx = i // cols
        col_idx = i % cols
        
        ax = fig.add_subplot(gs[row_idx, col_idx])
        
        # Get the visualization file for this model
        # Use the exact file naming pattern that visal.py generates
        viz_file = output_dir / f"{model['model_name']}_{model['agent_type']}_tree.png"
        
        if viz_file.exists():
            img = plt.imread(viz_file)
            ax.imshow(img)
            ax.set_title(f"{model['display_name']} - Score: {model['final_score']:.2f}", fontsize=16)
        else:
            print(f"No visualization found for {model['display_name']}, tried: {viz_file}")
            # Try to find by display name
            potential_files = list(output_dir.glob(f"{model['model_name']}_*{model['display_name'].lower().replace(' ', '-').replace('.', '')}*_tree.png"))
            if not potential_files:
                # Try broader search
                potential_files = list(output_dir.glob(f"{model['model_name']}_*_tree.png"))
                for viz_path in potential_files:
                    filename = str(viz_path.name).lower()
                    model_keywords = model['display_name'].lower().split()
                    if any(keyword in filename for keyword in model_keywords):
                        viz_file = viz_path
                        break
            elif potential_files:
                viz_file = potential_files[0]
                
            if viz_file.exists():
                img = plt.imread(viz_file)
                ax.imshow(img)
                ax.set_title(f"{model['display_name']} - Score: {model['final_score']:.2f}", fontsize=16)
            else:
                ax.text(0.5, 0.5, f"Visualization not available for {model['display_name']}", 
                       ha='center', va='center', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_grid.png")
    plt.close()
    
    # 3. Create a focused visual comparison with just the scores and rankings
    plt.figure(figsize=(10, 8))
    
    # Set up the table data
    table_data = []
    for rank, model in enumerate(model_data, start=1):
        table_data.append([
            f"{rank}",
            model['display_name'],
            f"{model['final_score']:.2f}"
        ])
    
    # Create a table with no axes
    ax = plt.gca()
    ax.axis('tight')
    ax.axis('off')
    
    # Set consistent blue-themed colors for rows
    cell_colors = []
    for i, model in enumerate(model_data):
        color = cmap(0.3 + 0.7 * model['final_score'])  # Scale to use middle-to-high range of colormap
        # Add slight alpha for better readability
        adjusted_color = (*color[:3], 0.3)  # RGB with alpha 0.3
        row_color = [adjusted_color] * 3  # Apply to all cells in the row
        cell_colors.append(row_color)
    
    table = ax.table(cellText=table_data,
                    colLabels=["Rank", "Model", "Score"],
                    loc='center',
                    cellLoc='center',
                    cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    
    plt.title("Model Performance Ranking", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "results.png")
    plt.close()
    
    print(f"Generated comparison visualizations in {output_dir}")
else:
    print("No model data available for comparison") 