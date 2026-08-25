## 18/08/2026
1. Created a file named testing.ipynb and put it in the scratch folder to explore the data using jupyter notebook
Used the libraries pandas, numpy, matplotlib, seaborn, pydicom, monai, timm, albumentations, kaggle

2. Created a eda.py file in the src folder to perform eda on the data
Ran the eda.py script to explore the data. It generates plots and saves them in the plots folder and also saves a text report in the results folder.

3. Created time_sheets.md file to keep track of the time spent on the project.

4. Created a execution_plan.csv and execution_plan.xlsx file in the scratch folder to keep track of the execution plan.

5. Created a generate_execution_plan.py file in the scratch folder to generate the execution plan.
Used the generate_execution_plan.py script to generate the execution plan. The file is saved in the scratch folder.

6. Created a pseudo_label_generator.py file in the scratch folder to generate pseudo-labels for the training data.
Used the pseudo_label_generator.py script to generate pseudo-labels for the training data. The file is saved in the scratch folder.

## 19/08/2026
1. Created a data_visualization.py file in the scratch folder to visualize the data.
Used the data_visualization.py script to visualize the data. It generates plots and saves them in the plots folder.

2. Created a src/dataset.py file to create a custom dataset for the RSNA knee abnormality detection task.
Used the load_dicom_slice and load_and_resize_series functions to load and preprocess the DICOM slices and stack them into a 3D volume.
Used the RSNAKneeDataset class to create a custom dataset for the RSNA knee abnormality detection task.
Ran the dataset.py script to test the dataset.

## 23/08/2026
1. Created a src/model.py file to create a custom model for the RSNA knee abnormality detection task.
Used the SliceEncoder class to create a custom model for the RSNA knee abnormality detection task.
Ran the model.py script to test the model.

2. Created a SequenceAggregator class to aggregate the sequence of slice features into a single study-level feature vector.
Used the SequenceAggregator class to aggregate the sequence of slice features into a single study-level feature vector.
Ran the model.py script to test the model.

3. Created a RSNAKneeModel class to create a custom model for the RSNA knee abnormality detection task.
Used the RSNAKneeModel class to create a custom model for the RSNA knee abnormality detection task.
Ran the model.py script to test the model.