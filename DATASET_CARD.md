#### **`DATASET_CARD.md`**

```markdown
# Harmonized Arabic Hate Speech Dataset

## Dataset Summary
This dataset is a harmonization of six open-source Arabic hate speech corpora, re-mapped to a unified taxonomy (Normal, Gender Hate, Origin Hate, Religious Hate).

## Sources & Citations
1. **OSACT5:** Mubarak et al. (2022)
2. **SoHateful:** Zaghouani et al. (2024)
3. **MLMA:** Ousidhoum et al. (2019)
4. **ArMI:** Mulki & Ghanem (2021)
5. **Brothers:** Albadi et al. (2018)
6. **Egyptian 5-Way:** Ahmed et al. (2022)

## Statistics
- **Total Samples:** ~40k
- **Language:** Modern Standard Arabic (MSA) & Dialectal Arabic.
- **Label Schema:**
  - `0`: Not Hate
  - `1`: Hate (Subtypes: OH, GH, RH)

## License
The harmonized dataset is provided for research purposes. Users must adhere to the licenses of the original constituent datasets.