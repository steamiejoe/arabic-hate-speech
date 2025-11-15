# DATASET CARD — Arabic Hate Speech Unified Dataset

This dataset is a unified, harmonized collection of Arabic hate speech and offensive language samples collected from multiple publicly available academic sources.

## 📚 Sources Included

1. **OSACT Shared Task 2022**  
2. **MLMA (Abusive Arabic Language)**  
3. **SoHateful Dataset**  
4. **Additional Twitter IDs Dataset (sbalsefri)**
   - To be hydrated using Twarc2 and merged later.

---

## 📊 Dataset Statistics (Before Processing)

(Generated in `0_data_snapshot.ipynb`)

- Total samples: ~**X,XXX**
- Hate class: **Y%**
- Non-hate class: **Z%**
- Average text length: **N characters**

---

## ⚙️ Preprocessing Performed

- URL removal  
- Mention and hashtag normalization  
- Arabic letter normalization (`alef`, `ta marbouta`, `alef maksura`)  
- Emoji retention  
- Repeated character normalization  
- Removal of non-Arabic characters  
- Length trimming  

---

## 🧩 Planned Augmentations

- Back-translation  
- Word dropout  
- Synonym replacement via Arabic WordNet  
- Character-level noise  

---

## ⚠️ Usage Notes

Do **not** redistribute hydrated tweets directly (Twitter ToS).  
Share only Tweet IDs and labels.

This dataset is intended strictly for research and evaluation in Arabic NLP.

