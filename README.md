# Yaqeen-AI: Arabic Hate Speech Classification 

## Overview
This project presents **Yaqeen-AI**, a robust system for detecting and classifying Arabic hate speech. It addresses the scarcity of fine-grained Arabic datasets by harmonizing six major corpora (OSACT, MLMA, SoHateful, ArMI, Brothers, Egyptian-5-Way).

## Key Features
- **Data Harmonization:** Unified ~40k samples from diverse sources into a standard schema.
- **Advanced Modeling:** Compares a **Multi-Task Learning (MTL)** MARBERTv2 architecture against a **Cascade** approach and classic ML baselines.
- **Granular Classification:** Detects specific hate types:
  - **GH:** Gender-Based Hate
  - **OH:** Origin/Race-Based Hate
  - **RH:** Religious Hate
- **Deployment:** Includes an interactive Streamlit dashboard for real-time inference.

## Project Structure
- `data/`: Stores original datasets and processed dataframes.
- `models/`: Stores model files upon training.
- `notebooks/`: Step-by-step processing, training, and evaluation.
- `app/`: Streamlit web application.

## Sources & Citations
The data used in this project is derived from the following open-source works. If you use this repository, please cite the original authors:

1.  **OSACT5:** 
```text 
Mubarak, H., Al-Khalifa, H., & Al-Thubaity, A. (2022). Overview of OSACT5 shared task on Arabic offensive language and hate speech detection. *Proceedings of the 5th Workshop on Open-Source Arabic Corpora and Processing Tools*, 162-166.
```
2.  **SoHateful:** 
```text 
Zaghouani, W., Mubarak, H., & Biswas, M.R. (2024). So hateful! Building a multi-label hate speech annotated Arabic dataset. *Proceedings of LREC-COLING 2024*, 15044-15055.
```
3.  **MLMA:** 
```text 
Ousidhoum, N., Lin, Z., Zhang, H., Song, Y., & Yeung, D.Y. (2019). Multilingual and multi-aspect hate speech analysis. *arXiv preprint arXiv:1908.11049*.
```
4.  **ArMI:** 
```text 
Mulki, H., & Ghanem, B. (2021). ArMI at FIRE 2021: Overview of the First Shared Task on Arabic Misogyny Identification. *FIRE (Working Notes)*, 820-830.
```
5.  **Brothers:** 
```text 
Albadi, N., Kurdi, M., & Mishra, S. (2018). Are they our brothers? Analysis and detection of religious hate speech in the Arabic Twittersphere. *2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)*, 69-76.
```
  * *Accessed via the Hate Speech Superset:* Tonneau, M. et al. (2024)
6.  **Egyptian 5-Way:** 
```text 
Ahmed, I., Abbas, M., Hatem, R., Ihab, A., & Fahkr, M.W. (2022). Fine-tuning Arabic pre-trained transformer models for Egyptian-Arabic dialect offensive language and hate speech detection. *2022 20th International Conference on Language Engineering (ESOLEC)*, 170-174.
```
7.  **Superset Curator:** 
```text 
Tonneau, M., Liu, D., Fraiberger, S., Schroeder, R., Hale, S.A., & Röttger, P. (2024). From languages to geographies: Towards evaluating cultural bias in hate speech datasets. *arXiv preprint arXiv:2404.17874*.
```

## BibTeX
```bibtex
@inproceedings{mubarak2022overview,
  title={Overview of OSACT5 shared task on Arabic offensive language and hate speech detection},
  author={Mubarak, Hamdy and Al-Khalifa, Hend and Al-Thubaity, AbdulMohsen},
  booktitle={Proceedinsg of the 5th Workshop on Open-Source Arabic Corpora and Processing Tools with Shared Tasks on Qur'an QA and Fine-Grained Hate Speech Detection},
  pages={162--166},
  year={2022}
}

@inproceedings{zaghouani2024so,
  title={So hateful! Building a multi-label hate speech annotated Arabic dataset},
  author={Zaghouani, Wajdi and Mubarak, Hamdy and Biswas, Md Rafiul},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={15044--15055},
  year={2024}
}

@article{ousidhoum2019multilingual,
  title={Multilingual and multi-aspect hate speech analysis},
  author={Ousidhoum, Nedjma and Lin, Zizheng and Zhang, Hongming and Song, Yangqiu and Yeung, Dit-Yan},
  journal={arXiv preprint arXiv:1908.11049},
  year={2019}
}

@article{mulki2021armi,
  title={ArMI at FIRE 2021: Overview of the First Shared Task on Arabic Misogyny Identification.},
  author={Mulki, Hala and Ghanem, Bilal},
  journal={FIRE (Working Notes)},
  pages={820--830},
  year={2021}
}

@inproceedings{albadi2018they,
  title={Are they our brothers? analysis and detection of religious hate speech in the arabic twittersphere},
  author={Albadi, Nuha and Kurdi, Maram and Mishra, Shivakant},
  booktitle={2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)},
  pages={69--76},
  year={2018},
  organization={IEEE}
}

@inproceedings{ahmed2022fine,
  title={Fine-tuning arabic pre-trained transformer models for egyptian-arabic dialect offensive language and hate speech detection and classification},
  author={Ahmed, Ibrahim and Abbas, Mostafa and Hatem, Rany and Ihab, Andrew and Fahkr, Mohamed Waleed},
  booktitle={2022 20th International Conference on Language Engineering (ESOLEC)},
  volume={20},
  pages={170--174},
  year={2022},
  organization={IEEE}
}

@inproceedings{tonneau-etal-2024-languages,
    title = "From Languages to Geographies: Towards Evaluating Cultural Bias in Hate Speech Datasets",
    author = {Tonneau, Manuel and Liu, Diyi and Fraiberger, Samuel and Schroeder, Ralph and Hale, Scott and R{\"o}ttger, Paul},
    editor = {Chung, Yi-Ling and Talat, Zeerak and Nozza, Debora and Plaza-del-Arco, Flor Miriam and R{\"o}ttger, Paul and Mostafazadeh Davani, Aida and Calabrese, Agostina},
    booktitle = "Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "[https://aclanthology.org/2024.woah-1.23](https://aclanthology.org/2024.woah-1.23)",
    pages = "283--311"
}
```
## Installation
```bash
pip install -r requirements.txt
streamlit run app/app.py
