# Detection of Machine-Generated Text using BERT Models

## 📌 Overview
This project focuses on detecting AI-generated text using **BERT (Bidirectional Encoder Representations from Transformers)** and complementary ensemble learning techniques. With the growing sophistication of Large Language Models (LLMs), distinguishing between human-authored and machine-generated content has become critical in fields such as academic integrity, journalism, and online communication.

The solution combines:
- **Fine-tuned BERT models** for semantic and syntactic feature extraction.
- **Ensemble classifiers** for robustness against noise and adversarial examples.

---

## 📂 Repository Structure

- ensemble-learning-technique-1.ipynb # Ensemble-based AI text detection pipeline
- bert-model-for-ai-text-detection.ipynb # BERT fine-tuning workflow
- ai-text-detection-bert-model.ipynb # Detailed BERT training and prediction
- report.pdf # Complete project documentation
- README.md # Project description (this file)


---

## 🚀 Workflow

1. **Data Cleaning & Preprocessing**
   - Removal of irrelevant symbols, punctuation, stopwords, and non-alphabetic terms.
   - Tokenization via BERT-specific preprocessing for better alignment with the model.

2. **Dataset Expansion**
   - Combined multiple public datasets.
   - Expanded training samples from **~1.3K to ~50K** for better generalization.

3. **Model Training**
   - Fine-tuned `bert-base-uncased` for sequence classification.
   - Experimented with `bert-large-cased` for deeper representations (trade-off: higher computation time).

4. **Ensemble Learning**
   - TF-IDF + Logistic Regression + SGDClassifier achieved LB scores above **0.93**.
   - Combined predictions from BERT and classical models for robustness.

5. **Prediction Phase**
   - Generated AI/human predictions for unseen text.
   - Saved outputs to CSV for evaluation.

---

## 🧠 Key Features of BERT in This Project
- Contextual word embeddings capturing subtle linguistic differences.
- Sensitivity to sentence structure and vocabulary diversity.
- Recognition of semantic coherence and sentiment variations.

---

## 📊 Experiments & Observations
- **Reducing batch size** improved offline accuracy but slightly lowered leaderboard scores.
- **Adam optimizer** generally outperformed RMSE loss; SGD offered better generalization.
- Human-written text tends to use **more rare words** and show **greater vocabulary variety**.
- AI text is often **more formal and structured** but less emotionally expressive.
- Heavy **spelling noise** in test data reduces transformer performance — TF-IDF sometimes performs better in such cases.

---

## ⚠️ Edge Cases
- Misspellings and unconventional grammar can trick detection models.
- Certain AI outputs are intentionally noisy to mimic human errors.
- Overfitting to leaderboard datasets reduces generalization.

---

## 💡 Recommendations
- Avoid over-reliance on large transformer models if runtime is limited.
- Use BERT preprocessing instead of generic NLP tokenizers.
- Introduce artificial noise into training data to improve robustness.
- Blend transformer-based and classical models for better results.

---

## 📈 Results Summary
- Transformers underperform with heavy spelling noise.
- TF-IDF remains competitive for stylistic classification.
- Ensemble methods work best when base models achieve **>50% accuracy**.

---

## 📚 References
- [A Survey on LLM-Generated Text Detection](https://aclanthology.org/2025.cl-1.8.pdf)  
- [Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness](https://arxiv.org/pdf/2409.16914)  
- [LLM4DV: Using Large Language Models for Hardware Test Stimuli Generation](https://arxiv.org/pdf/2310.04535)  
- [Implementing BERT and Fine-Tuned RoBERTa to Detect AI-Generated News](https://arxiv.org/pdf/2306.07401)  

---

## 👤 Author
**Adarsh Yadav**  
4th Year, Chemical Engineering, IIT Roorkee  
GitHub: [codebreaker0001](https://github.com/codebreaker0001) 

