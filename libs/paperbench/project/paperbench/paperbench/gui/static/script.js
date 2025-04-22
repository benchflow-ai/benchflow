var loading_indicator = '<marquee behavior="alternate" direction="left" scrollamount="20">Loading...</marquee>'

function saveTaskState(taskId, isCollapsed) {
    const states = JSON.parse(localStorage.getItem('taskStates') || '{}');
    states[taskId] = isCollapsed;
    localStorage.setItem('taskStates', JSON.stringify(states));
}

function loadTaskState(taskId) {
    const states = JSON.parse(localStorage.getItem('taskStates') || '{}');
    return states[taskId] || false; // default to expanded (false means not collapsed)
}

function toggleTask(taskId) {
    var taskContent = document.querySelector('#task-' + taskId + ' .task-content');
    var toggleIcon = document.querySelector('#toggle-icon-' + taskId);
    const isCollapsed = taskContent.style.display !== 'none';
    
    taskContent.style.display = isCollapsed ? 'none' : 'block';
    toggleIcon.innerHTML = isCollapsed ? '&#9654;' : '&#9660;';
    
    saveTaskState(taskId, isCollapsed);
}

function toggleAllTasks(action) {
    var tasks = document.querySelectorAll('.task');
    tasks.forEach(function (task) {
        const taskId = task.id.replace('task-', '');
        const taskContent = task.querySelector('.task-content');
        const toggleIcon = document.querySelector('#toggle-icon-' + taskId);
        
        const isCollapsed = action === 'collapse';
        taskContent.style.display = isCollapsed ? 'none' : 'block';
        toggleIcon.innerHTML = isCollapsed ? '&#9654;' : '&#9660;';
        
        saveTaskState(taskId, isCollapsed);
    });
}

function confirmationFlow(taskId) {
    var criteriaContainer = document.querySelector("#task-" + taskId + " .criteria-container");
    if (criteriaContainer.innerHTML.includes("Loading...")) {
        alert("Task is busy, please wait. (" + taskId + ")");
        return false;
    }
    if (criteriaContainer.innerHTML.trim() !== "") {
        return confirm("Are you sure?");
    }
    return true;
}

function generateRequirements(taskId) {
    var criteriaContainer = document.querySelector("#task-" + taskId + " .criteria-container");
    if (!confirmationFlow(taskId)) {
        return;
    }
    criteriaContainer.innerHTML = loading_indicator;

    console.log("Generating requirements for task", taskId);
    fetch("/generate_requirements?node_id=" + taskId)
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                console.log("Requirements written:", data);
                criteriaContainer.innerHTML = "<p>Requirements written successfully. Please wait for concurrent requests to finish.</p>";
                location.reload();
            } else {
                console.error("Error writing requirements:", data.message);
                criteriaContainer.innerHTML = `<p>Failed, please try again: ${data.message}</p>`;
            }
        })
        .catch(error => console.error("Error writing requirements:", error));
}

function deleteNode(taskId) {
    if (!confirm("Are you sure you want to delete this node?")) {
        return;
    }

    var criteriaContainer = document.querySelector("#task-" + taskId + " .criteria-container");
    criteriaContainer.innerHTML = loading_indicator;

    console.log("Deleting node", taskId);
    fetch("/delete_node?node_id=" + taskId)
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                console.log("Node deleted:", data);
                criteriaContainer.innerHTML = "<p>Node deleted successfully. Please wait for concurrent requests to finish.</p>";
                location.reload();
            } else {
                console.error("Error deleting node:", data.message);
                criteriaContainer.innerHTML = `<p>Failed, please try again: ${data.message}</p>`;
            }
        })
        .catch(error => {
            console.error("Error deleting node:", error);
            criteriaContainer.innerHTML = `<p>Failed, please try again: ${error}</p>`;
        });
}

function updateRequirements(taskId) {
    var newRequirements = document.getElementById('requirements-' + taskId).innerText.trim();
    fetch('/update_requirements', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'node_id': taskId, 'requirements': newRequirements }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Requirements updated for node', taskId);
        } else {
            console.error('Error updating requirements:', data.message);
            alert('Failed to update requirements: ' + data.message);
        }
    })
    .catch(error => console.error('Error updating requirements:', error));
}

function addSubTask(taskId) {
    var newRequirements = prompt("Enter the requirements for the new sub-task:");
    if (newRequirements === null || newRequirements.trim() === "") {
        return; // User cancelled or entered empty string
    }

    fetch('/add_sub_task', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'parent_id': taskId, 'requirements': newRequirements }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Sub-task added to node', taskId);
            location.reload();
        } else {
            console.error('Error adding sub-task:', data.message);
            alert('Failed to add sub-task: ' + data.message);
        }
    })
    .catch(error => console.error('Error adding sub-task:', error));
}

function updateWeight(taskId) {
    var newWeight = document.getElementById('weight-' + taskId).innerText.trim();
    // Ensure the weight is a positive integer
    newWeight = parseInt(newWeight);
    if (isNaN(newWeight) || newWeight <= 0) {
        alert('Weight must be a positive integer.');
        location.reload(); // Reload to reset the displayed weight
        return;
    }
    
    fetch('/update_weight', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'node_id': taskId, 'weight': newWeight }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Weight updated for node', taskId);
        } else {
            console.error('Error updating weight:', data.message);
            alert('Failed to update weight: ' + data.message);
            location.reload(); // Reload to reset the displayed weight
        }
    })
    .catch(error => {
        console.error('Error updating weight:', error);
        alert('Failed to update weight. Please try again.');
        location.reload(); // Reload to reset the displayed weight
    });
}

function initializeTaskStates() {
    var tasks = document.querySelectorAll('.task');
    tasks.forEach(function (task) {
        const taskId = task.id.replace('task-', '');
        const isCollapsed = loadTaskState(taskId);
        const taskContent = task.querySelector('.task-content');
        const toggleIcon = document.querySelector('#toggle-icon-' + taskId);
        
        if (isCollapsed) {
            taskContent.style.display = 'none';
            toggleIcon.innerHTML = '&#9654;';
        }
    });
}

function updateTaskCategory(taskId, category) {
    fetch('/update_task_category', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'node_id': taskId, 'category': category }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Task category updated for node', taskId);
        } else {
            console.error('Error updating task category:', data.message);
            alert('Failed to update task category: ' + data.message);
            location.reload(); // Reset the dropdown to its previous state
        }
    })
    .catch(error => {
        console.error('Error updating task category:', error);
        alert('Failed to update task category. Please try again.');
        location.reload(); // Reset the dropdown to its previous state
    });
}

function moveNode(taskId, direction) {
    fetch('/move_node', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'node_id': taskId, 'direction': direction }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Node moved', taskId, direction);
            location.reload();
        } else {
            console.error('Error moving node:', data.message);
            alert('Failed to move node: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error moving node:', error);
        alert('Failed to move node. Please try again.');
    });
}

function updateScore(taskId) {
    var newScore = document.getElementById('score-' + taskId).innerText.trim();
    // Convert to float and validate
    newScore = parseFloat(newScore);
    if (isNaN(newScore) || newScore < 0 || newScore > 1) {
        alert('Score must be a number between 0 and 1.');
        location.reload(); // Reload to reset the displayed score
        return;
    }
    
    fetch('/update_score', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'node_id': taskId, 'score': newScore }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Score updated for node', taskId);
            location.reload(); // Reload the page to show updated scores
        } else {
            console.error('Error updating score:', data.message);
            alert('Failed to update score: ' + data.message);
            location.reload(); // Reload to reset the displayed score
        }
    })
    .catch(error => {
        console.error('Error updating score:', error);
        alert('Failed to update score. Please try again.');
        location.reload(); // Reload to reset the displayed score
    });
}

function updateExplanation(taskId) {
    var newExplanation = document.getElementById('explanation-' + taskId).innerText.trim();
    
    fetch('/update_explanation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'node_id': taskId, 'explanation': newExplanation }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Explanation updated for node', taskId);
        } else {
            console.error('Error updating explanation:', data.message);
            alert('Failed to update explanation: ' + data.message);
            location.reload(); // Reload to reset the displayed explanation
        }
    })
    .catch(error => {
        console.error('Error updating explanation:', error);
        alert('Failed to update explanation. Please try again.');
        location.reload(); // Reload to reset the displayed explanation
    });
}

document.addEventListener('DOMContentLoaded', function() {
    initializeTaskStates();
});

function duplicateNode(taskId) {
    fetch('/duplicate_node', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'node_id': taskId }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Node duplicated:', taskId);
            location.reload();
        } else {
            console.error('Error duplicating node:', data.message);
            alert('Failed to duplicate node: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error duplicating node:', error);
        alert('Failed to duplicate node. Please try again.');
    });
}

function copyTaskId(taskId) {
    navigator.clipboard.writeText(taskId).then(() => {
        // Optional: Show a brief visual feedback
        const btn = document.querySelector(`#task-${taskId} .copy-id-btn`);
        const originalText = btn.innerHTML;
        btn.innerHTML = '✓';
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 1000);
    }).catch(err => {
        console.error('Failed to copy task ID:', err);
        alert('Failed to copy task ID to clipboard');
    });
}

function moveNodeToParent(taskId) {
    const newParentId = prompt("Enter the ID of the new parent node:");
    if (!newParentId) return; // User cancelled

    fetch('/move_node_to_parent', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            'node_id': taskId, 
            'new_parent_id': newParentId 
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Node moved to new parent:', taskId);
            location.reload();
        } else {
            console.error('Error moving node:', data.message);
            alert('Failed to move node: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error moving node:', error);
        alert('Failed to move node. Please try again.');
    });
}
