# Ecosort-waste-management

# Intelligent Waste Classification and Recycling Assistant

## 1. Project Overview

EcoSort is an integrated waste management assistant that uses machine learning and retrieval-based natural language processing to help users identify waste and access relevant recycling guidance.

The system accepts either a text description or an image of waste. It predicts the waste category and retrieves corresponding recycling instructions from a collection of policy documents.

The project combines image classification, text classification, and retrieval-augmented generation (RAG) concepts to demonstrate how AI can support waste sorting and recycling decisions.

## 2. Project Objectives

- Classify waste using images and text descriptions.
- Apply transfer learning to image classification using MobileNetV2.
- Use TF-IDF and Logistic Regression to classify text descriptions.
- Retrieve category-specific recycling instructions from policy documents.
- Integrate the models and retrieval component into a single assistant.
- Evaluate model performance using appropriate classification metrics.

## 3. Waste Categories

EcoSort supports nine waste categories:

- Cardboard
- Food Organics
- Glass
- Metal
- Miscellaneous Trash
- Paper
- Plastic
- Textile Trash
- Vegetation

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas and NumPy | Data processing and manipulation |
| Matplotlib and Seaborn | Data visualization, where used |
| Scikit-learn | Text classification, TF-IDF, data splitting, and evaluation |
| TensorFlow and Keras | Image preprocessing and deep learning |
| MobileNetV2 | Pretrained image feature extraction and transfer learning |
| Hugging Face Transformers | Experimentation with FLAN-T5-small for text generation |
| Jupyter Notebook | Development, experimentation, and documentation |

## 5. Datasets

### Text dataset

The text classification dataset, `waste_descriptions.csv`, contains 5,000 waste descriptions and associated category labels, along with disposal-related fields.

The descriptions are cleaned and transformed into numerical features using TF-IDF before classification.

### Image dataset

The image classification component uses the RealWaste dataset, containing 4,752 images across nine waste categories.

Images are resized to 224 × 224 pixels and normalized before being passed to the model.

### Recycling policy documents

The retrieval component uses `waste_policy_documents.json` and relevant disposal instructions to build a searchable collection of recycling guidance.

Policy retrieval uses TF-IDF vectorization and cosine similarity to identify relevant text.

## 6. System Architecture

EcoSort follows this workflow:

1. **Input:** The user supplies a waste image or text description.
2. **Classification:** The appropriate model predicts the waste category.
3. **Policy retrieval:** The predicted category is used to find relevant recycling documents.
4. **Response:** The assistant returns the predicted category, confidence score, recycling instructions, and supporting policy metadata.

### Models

**Image classification:** MobileNetV2 pretrained on ImageNet is used for transfer learning. Its original classification head is replaced with a nine-class output layer. The model is first trained with the pretrained base frozen and then fine-tuned.

**Text classification:** A TF-IDF vectorizer with unigrams and bigrams transforms the descriptions into numerical features. Logistic Regression predicts the waste category.

**Policy retrieval:** TF-IDF document vectors and cosine similarity identify relevant recycling guidance. FLAN-T5-small was explored for response generation, but its outputs were not consistently useful, so a rule-based fallback was used in the tested workflow.

## 7. Model Evaluation

### Image classification

| Metric | Result |
|---|---:|
| Test images | 480 |
| Test accuracy | 83.33% |
| Macro F1-score | 0.83 |
| Weighted F1-score | 0.83 |

Vegetation performed best, with an F1-score of 0.97. Miscellaneous Trash was more challenging, with an F1-score of 0.69. Common errors included confusion between Glass and Plastic and between Cardboard and Miscellaneous Trash.

### Text classification

| Metric | Result |
|---|---:|
| Test descriptions | 1,003 |
| Test accuracy | 99.30% |
| Macro F1-score | 0.99 |
| Weighted F1-score | 0.99 |

The text classifier performed strongly on the held-out dataset. However, the descriptions may contain synthetic or repeated wording patterns, which can make classification easier than real-world use. Evaluation on diverse, independently collected descriptions is needed to establish generalization.

### Fine-tuning observations

Fine-tuning MobileNetV2 increased training accuracy but did not improve the best validation accuracy. This suggests that further training alone may not improve generalization and that additional data or regularization may be beneficial.

## 8. Installation and Setup

### Prerequisites

- Python
- Jupyter Notebook
- The project datasets
- The required Python packages

### Install dependencies

Install the libraries used in the project:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow transformers
```

Depending on your environment, TensorFlow and Transformers may require compatible Python and package versions.

### Run the project

1. Clone or download the project repository.
2. Place the required datasets in the locations expected by the notebook.
3. Install the dependencies.
4. Open the EcoSort Jupyter Notebook.
5. Run the notebook cells sequentially, starting with imports and data loading.
6. Test the assistant with a text description and an image.

Make sure the dataset paths in the notebook match your local directory structure.

## 9. Example Usage

### Text input

```python
result = waste_management_assistant(
    "an empty plastic water bottle",
    input_type="text"
)

print("Category:", result.get("waste_category"))
print("Confidence:", result.get("confidence"))
print("Instructions:", result.get("recycling_instructions"))
```

### Image input

```python
result = waste_management_assistant(
    "RealWaste/Cardboard/Cardboard_1.jpg",
    input_type="image"
)

print("Category:", result.get("waste_category"))
print("Confidence:", result.get("confidence"))
print("Instructions:", result.get("recycling_instructions"))
```

The image path is an example; update it to match the location of your dataset.

## 10. Limitations

- Image classification is not perfectly accurate and may confuse visually similar materials.
- The text dataset may not represent the variety of descriptions encountered in real-world settings.
- Model confidence scores are not guarantees that predictions are correct.
- Retrieval quality depends on the completeness, accuracy, and organization of the policy documents.
- Recycling requirements differ by location, so the retrieved instructions should not replace local authority guidance.
- The text-generation experiment did not consistently produce satisfactory answers, and the tested workflow relied on rule-based fallback behavior.

## 11. Future Improvements

- Collect more diverse, real-world waste descriptions and images.
- Improve performance on visually similar categories.
- Add more reliable category-specific policy matching and source citations.
- Improve the response-generation component and evaluate factual consistency.
- Build a web or mobile interface for easier access.
- Expand policy coverage and incorporate location-specific recycling rules.
- Test the complete system with users and independently collected waste samples.

## 12. Conclusion

EcoSort demonstrates how computer vision, text classification, and document retrieval can be combined into an integrated waste management assistant. The image model achieved 83.33% test accuracy, while the text classifier achieved 99.30% on its held-out dataset.

These results are promising within the evaluated datasets, but practical deployment requires further validation, reliable policy retrieval, and testing on real-world inputs. The project provides a foundation for developing more accessible and informative waste-sorting support tools.


