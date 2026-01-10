Here is the updated, simple README code with the installation guide and the results table included.

```markdown
# Machine Learning at Scale (MovieLens 32M)

This repository contains the code for the Machine Learning at Scale practical assignment. It implements a Hierarchical Matrix Factorization model from scratch using Numba optimization on the MovieLens 32M dataset.

## 📄 Main File
**[ml_at_scale_draft.ipynb](ml_at_scale_draft.ipynb)**
> **Marker Note:** Please open this file. It contains the full pipeline

## ⚙️ How to Run

1. **Install Dependencies:**
   Ensure you have the `requirements.txt` file in your directory, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Notebook:**
   Open `ml_at_scale_draft.ipynb` in Jupyter Lab or Notebook.

3. **Execution:**
   Run the cells sequentially.
   * *Note: The code is configured to automatically download and extract the MovieLens 32M dataset if it is not found locally.*

## 📊 Results Summary

| Model | Test RMSE | Notes |
| :--- | :--- | :--- |
| Bias-Only Baseline | 0.858 | Underfitting, ignores latent preferences. |
| Standard MF (K=32) | 0.768 | Captures collaborative signals well. |
| **Hierarchical Genre MF** | **0.765** | **Best performance.** Regularizes tail items effectively. |

## 🚀 Key Features
*   **Scale:** Processes 32 million ratings using custom flattened arrays (CSR-style).
*   **Speed:** Custom Numba JIT implementation achieves ~4.7s per iteration (11x speedup over vectorization).
*   **Algorithm:** Implements Hierarchical ALS with Genre Priors to solve the Cold Start problem.
```
