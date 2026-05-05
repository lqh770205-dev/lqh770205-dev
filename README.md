# Qiheng Li

ML Engineer | Multimodal AI, Anomaly Detection, Computer Vision  

Graduate student at NYU Tandon (ECE), focused on building end-to-end machine learning systems with strong emphasis on reproducibility, evaluation, and real-world performance.

---

## 🚀 Featured Projects

### 🔹 Multimodal VQA with LoRA (Kaggle Pipeline)
Built an end-to-end multimodal QA system (image + question + options → answer) using a frozen SmolVLM backbone with parameter-efficient fine-tuning (LoRA/DoRA).  
Designed a reproducible training/inference pipeline with checkpointing and deterministic scoring; achieved ~0.83 public leaderboard score.  

**Tech:** PyTorch, Transformers, LoRA/DoRA, Multimodal Learning  
🔗 https://github.com/lqh770205-dev/text_to_svg

---

### 🔹 High-Frequency Trading Anomaly Detection
Developed a Transformer-based sequence modeling pipeline on limit order book data to detect abnormal trading behavior.  
Combined latent representations with One-Class SVM (Nyström approximation); achieved ~0.947 leaderboard score, demonstrating strong temporal modeling capability.  

**Tech:** PyTorch, Scikit-learn, Time Series, Transformer  
🔗 https://github.com/lqh770205-dev/Anomaly-Detection-in-Limit-Order-Book-Data-Transformer-One-Class-SVM

---

### 🔹 ConvNeXt Ablation & Object Detection (FCOS vs RetinaNet)
Implemented end-to-end object detection pipelines and conducted targeted ablations on ConvNeXt to analyze architectural impact.  
Compared anchor-free vs anchor-based detection, highlighting trade-offs in precision and generalization.  

**Tech:** PyTorch, Torchvision, CNN, Object Detection  
🔗 https://github.com/lqh770205-dev/convnext-ablation-object-detection

---

### 🔹 Text-to-SVG Retrieval System (Optional)
Built a retrieval-based system mapping text prompts to SVG graphics using TF-IDF similarity and reranking strategies.  
Focused on efficient inference under structural constraints (SVG validity, length limits).  

**Tech:** Python, NLP, Information Retrieval  
🔗 https://github.com/lqh770205-dev/text_to_svg
