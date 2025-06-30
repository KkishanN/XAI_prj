# XAI mini project
<!-- To Do -->
<!-- Give the command to create an environment first, then comes running the requirements.txt file!! -->
## Installation Guide

To set up the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/KkishanN/XAI_prj.git
    cd XAI-project
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```

4. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the application**:
    ```bash
    # Run the models, please run the commands in the given order:
    python main.py -model sgc  # Runs the Simplified Graph Convolution model
    python main.py -model sage  # Runs the GraphSAGE model
    python main.py -show_results  # Displays the results of both models
    ```

6. **View the report**:
    You can find the **report** [here](xai_report.pdf).