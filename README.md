# AI-HealthCare-Assistant

# Follow these steps to run:
### STEP 1:

clone the repo:
```bash
project repo: https://github.com/MarvinH02/AI-Healthcare-Assistant.git
```

### STEP2:

Create a conda environment:

```bash
conda create -n ai-healthcare-assistant python=3.10 -y
```

```bash
conda activate ai-healthcare-assistant
```

### STEP 3:
```bash
pip install -r requirements.txt
```

Don't forget to create an .env file and use the correct Names for the API keys:
PINECONE_API_KEY = "xxx"
OPENAI_API_KEY = "xxx"

Now, in VSCode go into the notebook, and initialize the enivornment.
Go into research/chatbot_pipeline.ipynb and initialize all the notebook and test the queries at the bottom.