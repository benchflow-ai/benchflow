FROM pb-env
# from pb-env, we inherit the env vars:
# constants: WORKSPACE_BASE, SUBMISSION_DIR, LOGS_DIR, CODE_DIR, AGENT_DIR
# overridable: CONDA_ENV_NAME, PYTHON_VERSION, REQUIREMENTS

# copy just the requirements file, so that we can cache conda separately from the agent files
COPY requirements.txt ${REQUIREMENTS}

# Install agent-specific requirements into the existing conda environment
RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${REQUIREMENTS} && \
    conda clean -afy

# put all the agent files in the expected location
COPY . ${AGENT_DIR}
