
# SPRec: Leveraging Self-Play to Debias Preference Alignment for Large Language Model-based Recommendations

<div style="text-align: center;">
<img src="./figs/method.png" alt="introduction" style="zoom:50%;" />
</div>

This repository provides the official PyTorch implementation and reproduction for the paper titled "SPRec: Leveraging Self-Play to Debias Preference Alignment for Large Language Model-based Recommendations" 

## Installation

1. Clone this git repository and change directory to this repository:

2. A new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) is suggested. 

    ```bash
    conda create --name SPRec python=3.10 -y
    ```

3. Activate the newly created environment.

    ```bash
    conda activate SPRec
    ```

4. Install the required modules from pip.

    ```bash
    pip install -r requirements.txt
    ```


## Quick Start

Due to GitHub's file size limitations, we have uploaded the minimal sample dataset **Goodreads** in `./data/Goodreads` and `./eval/Goodreads` for reproduction purposes. Additionally, the datasets used in our experiments—**MovieLens**, **CDs and Vinyl**, and **Steam**—will be uploaded to [ link coming soon ] later. If you wish to use a different dataset, please ensure that it is processed into a similar format.

### How to Train Using SPRec Framework

1. **SFT Training**:  
   Before using the SPRec training framework, you need to run SFT to fine-tune your base model for alignment with the recommendation task. Use the following command to perform SFT training:  
   ```bash
   bash ./shell/SFT.sh 0 1 2 3 # Specify your GPUs, e.g., 0 1 2 3
2. **SPRec Training**:    
    After completing SFT training, use the following command to perform SPRec training:
    ```bash
    bash ./shell/SPRec.sh 0 1 2 3 5 # Specify your GPUs, e.g., 0 1 2 3, and the number of iterations, e.g., 5
Once the above commands are executed, the evaluation results for top-1 and top-5 recommendations will be saved as eval_top1.json and eval_top5.json in the corresponding model directory.
