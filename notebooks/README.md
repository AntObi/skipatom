# Notebooks

This directory contains the Jupyter notebooks used to generate the figures in the paper _Ionic species representations for materials informatics_.  The use the following libraries which are all installable via `pip`:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `monty`
- `smact`
- `elementembeddings`[^1]
- `pymatviz`


The notebooks are organized as follows:

- `Process_propert_dataset.ipynb`: This notebook contains the code used to process the property dataset. It includes the code used to visualise Figures 2 and 3 (the distribution of the target values and the component distribution of the compositions in the oxi-MPv2022.10.28 dataset, respectively).
- `SkipSpecies_vector_analysis.ipynb`: This notebook contains the code used to analyse the induced 200-dimensonal SkipSpecies vectors. It includes the code used to visualise Figure 1 and Figure 4 (the periodic table heatmap of the oxidation states present in the oxi-MPv2022.10.28 dataset and the dimension-reduced visualisations of the SkipSpecies vector, respectively).
- ` SkipSpecies_induced_visualise_pool_dimensions.ipynb`: This notebook contains the code used to analysis the results of property prediction tasks using the induced SkipSpecies vectors as inputs to the ElemNet model. It includes the code used to visualise Figure 5 (the performance of the ElemNet model on the oxi-MPv2022.10.28 dataset using the induced SkipSpecies vectors as inputs). This particular notebook analyses the effect of dimension size and the pooling operation applied to the induced SkipSpecies vectors on the performance of the ElemNet model on the tasks.
- `SkipSpecies_heatmap_comparison.ipynb`: This notebook contains the code used to compare the difference in performance of the sum-pooled induced SkipAtom, SkipSpecies and induced SkipSpecies vectors on the property prediction tasks. It includes the code used to visualise Figure 6 (a heatmap depicting the effect of the chosen representation and dimension size on the performance of the ElemNet model on the oxi-MPv2022.10.28 dataset using the induced SkipAtom, SkipSpecies and SkipSpecies vectors as inputs).
- `Visualise_validation_errors_on_property_prediction.ipynb`: This notebook contains the code used to visualise the validation curves for the property prediction tasks. It includes the code used to visualise Figure 7 (the validation curves for the property prediction tasks on the oxi-MPv2022.10.28 dataset using the sum-pooled, 200-dimensional representations).

[^1]: `elementembeddings` is a library that provides tools for working with element and species embeddings. There will be some dependencies conflicts with the main `skipspecies` library, upon installation of the package, though the analysis can be replicated without it, if necessary.