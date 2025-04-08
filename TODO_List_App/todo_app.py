import streamlit as st

# --- Initialization ---
# Use session state to store the list of tasks
# This ensures the list persists across reruns (user interactions)
if 'tasks' not in st.session_state:
    # Initialize with an empty list or some default tasks if desired
    # Each task is a dictionary with 'description' and 'completed' status
    st.session_state.tasks = []
    # Example initial tasks (optional):
    # st.session_state.tasks = [
    #     {"description": "Learn Streamlit", "completed": True},
    #     {"description": "Build a To-Do App", "completed": False},
    # ]

# --- Helper Functions ---
def add_task(new_task_description):
    """Adds a new task to the session state task list."""
    if new_task_description: # Don't add empty tasks
        st.session_state.tasks.append({"description": new_task_description, "completed": False})

def toggle_task_completion(task_index):
    """Toggles the completion status of a task at the given index."""
    st.session_state.tasks[task_index]["completed"] = not st.session_state.tasks[task_index]["completed"]

def delete_task(task_index):
    """Deletes a task from the session state task list at the given index."""
    del st.session_state.tasks[task_index]
    # No need for st.rerun() here typically, as deleting via a button click
    # which modifies session state will trigger a rerun automatically.

# --- App Layout ---
st.set_page_config(page_title="Simple To-Do App", layout="centered")

st.title("📝 My To-Do List")
st.write("Manage your tasks effectively!")

# --- Input Form for New Tasks ---
# Using st.form helps group input widgets and execute logic only on submission.
# clear_on_submit=True ensures the input field is cleared after adding a task.
with st.form("new_task_form", clear_on_submit=True):
    new_task_input = st.text_input("Enter a new task:", placeholder="What needs to be done?")
    submitted = st.form_submit_button("Add Task")
    if submitted and new_task_input:
        add_task(new_task_input)
        # Input is cleared automatically due to clear_on_submit=True

# --- Display Task List ---
st.subheader("Current Tasks")

if not st.session_state.tasks:
    st.info("You have no tasks yet. Add some above! 🎉")
else:
    # Iterate through tasks with index for unique keys and operations
    for i, task in enumerate(st.session_state.tasks):
        # Use columns for better layout: Checkbox | Task Description | Delete Button
        col1, col2, col3 = st.columns([0.1, 0.75, 0.15])

        with col1:
            # Checkbox to toggle completion status
            # The key is crucial for Streamlit to track the state of each checkbox individually
            # on_change calls the function when the checkbox state changes
            st.checkbox(
                label="", # No visible label needed here
                value=task["completed"],
                key=f"check_{i}",
                on_change=toggle_task_completion,
                args=(i,) # Pass the task index to the callback function
            )

        with col2:
            # Display task description with strikethrough if completed
            task_description = task["description"]
            if task["completed"]:
                # Use Markdown for strikethrough
                st.markdown(f"~~{task_description}~~", unsafe_allow_html=True)
            else:
                st.markdown(task_description, unsafe_allow_html=True)

        with col3:
            # Button to delete the task
            # The key is crucial here too
            # on_click calls the function when the button is clicked
            st.button(
                "🗑️", # Use an emoji for delete
                key=f"delete_{i}",
                on_click=delete_task,
                args=(i,), # Pass the task index to the callback function
                help="Delete this task" # Tooltip for the button
            )
        st.divider() # Add a visual separator between tasks

# --- Optional: Task Summary ---
st.sidebar.header("Task Summary")
total_tasks = len(st.session_state.tasks)
completed_tasks = sum(1 for task in st.session_state.tasks if task["completed"])
active_tasks = total_tasks - completed_tasks

st.sidebar.metric("Total Tasks", total_tasks)
st.sidebar.metric("Completed Tasks ✅", completed_tasks)
st.sidebar.metric("Active Tasks ⏳", active_tasks)

if total_tasks > 0:
    progress = completed_tasks / total_tasks
    st.sidebar.progress(progress)
    if progress == 1.0:
        st.sidebar.balloons()

# --- How to Run ---
# Add this as a comment or in your README
"""
How to Run this App:
1. Save the code as a Python file (e.g., `todo_app.py`).
2. Make sure you have streamlit installed (`pip install streamlit`).
3. Open your terminal or command prompt.
4. Navigate to the directory where you saved the file.
5. Run the command: `streamlit run todo_app.py`
"""