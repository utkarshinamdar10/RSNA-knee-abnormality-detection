import pandas as pd
import os

# Define the tasks for the execution plan
plan_data = [
    {
        "Phase": "Phase 1: EDA & Setup",
        "Task ID": "1.1",
        "Task Name": "Environment Verification",
        "Description": "Verify GPU availability and install dependencies (pydicom, monai, timm, albumentations, pandas, openpyxl).",
        "Priority": "High",
        "Estimated Effort": "2 Hours",
        "Deliverables": "Functional virtual environment with all required scientific libraries."
    },
    {
        "Phase": "Phase 1: EDA & Setup",
        "Task ID": "1.2",
        "Task Name": "CSV Data Profiling",
        "Description": "Analyze train.csv and train_series.csv to map unique study-to-series UIDs, check missing labels, and profile report languages.",
        "Priority": "High",
        "Estimated Effort": "3 Hours",
        "Deliverables": "Detailed EDA notebook with language distribution and label frequency stats."
    },
    {
        "Phase": "Phase 1: EDA & Setup",
        "Task ID": "1.3",
        "Task Name": "DICOM Image Inspection",
        "Description": "Load sample DICOM images, check metadata (pixel spacing, orientation, manufacturer), and analyze slice dimensions/counts.",
        "Priority": "Medium",
        "Estimated Effort": "4 Hours",
        "Deliverables": "Visualizations of slices along Sagittal, Coronal, and Axial planes."
    },
    {
        "Phase": "Phase 2: Report Parsing",
        "Task ID": "2.1",
        "Task Name": "Multilingual Lexicon Creation",
        "Description": "Gather clinical terms for the 12 target abnormalities across English, Spanish, Dutch, German, Turkish, Greek, and French.",
        "Priority": "High",
        "Estimated Effort": "4 Hours",
        "Deliverables": "A dictionary mapping target classes to multilingual keywords."
    },
    {
        "Phase": "Phase 2: Report Parsing",
        "Task ID": "2.2",
        "Task Name": "Negation Check Parser",
        "Description": "Build regex/NLP patterns to handle negation structures (e.g., 'no tear', 'efüzyon izlenmedi') across different languages.",
        "Priority": "High",
        "Estimated Effort": "6 Hours",
        "Deliverables": "Python NLP utility class that parses text and outputs class detection flags."
    },
    {
        "Phase": "Phase 2: Report Parsing",
        "Task ID": "2.3",
        "Task Name": "Pseudo-Label Generation",
        "Description": "Apply parser to 4,349 unlabeled reports. Validate parser accuracy on the 58 ground-truth studies to compute precision/recall.",
        "Priority": "High",
        "Estimated Effort": "6 Hours",
        "Deliverables": "A generated CSV containing pseudo-labels / soft targets for all training studies."
    },
    {
        "Phase": "Phase 3: Image Processing",
        "Task ID": "3.1",
        "Task Name": "DICOM Reader & Windowing",
        "Description": "Implement DICOM pixel loading, rescale intercept/slope, and apply windowing/clipping to standardize contrast.",
        "Priority": "High",
        "Estimated Effort": "6 Hours",
        "Deliverables": "Preprocessed raw array loader supporting min-max scaling."
    },
    {
        "Phase": "Phase 3: Image Processing",
        "Task ID": "3.2",
        "Task Name": "3D Resizing & Interpolation",
        "Description": "Standardize slice thickness and resize/interpolate MRI series to a uniform size (e.g., 32 x 256 x 256).",
        "Priority": "High",
        "Estimated Effort": "6 Hours",
        "Deliverables": "Function to resample volumes to consistent dimensions."
    },
    {
        "Phase": "Phase 3: Image Processing",
        "Task ID": "3.3",
        "Task Name": "Plane Grouping & Pipeline",
        "Description": "Group input scans by Anatomical Plane (Axial, Coronal, Sagittal) and write a PyTorch Dataset class.",
        "Priority": "High",
        "Estimated Effort": "8 Hours",
        "Deliverables": "PyTorch Dataset loader returning processed multi-view tensors."
    },
    {
        "Phase": "Phase 4: Model Architecture",
        "Task ID": "4.1",
        "Task Name": "2.5D Slice Encoder",
        "Description": "Create a feature extraction encoder using a pretrained 2D CNN (e.g., EfficientNet-B4 or ConvNeXt) from the timm library.",
        "Priority": "High",
        "Estimated Effort": "6 Hours",
        "Deliverables": "Feature extractor neural net modules."
    },
    {
        "Phase": "Phase 4: Model Architecture",
        "Task ID": "4.2",
        "Task Name": "Sequence Aggregator",
        "Description": "Implement aggregation blocks (Global Pooling, GRU, or a Transformer Encoder) to merge slice-level features.",
        "Priority": "High",
        "Estimated Effort": "8 Hours",
        "Deliverables": "Aggregated 3D spatial classification model."
    },
    {
        "Phase": "Phase 4: Model Architecture",
        "Task ID": "4.3",
        "Task Name": "View Fusion Design",
        "Description": "Combine predictions from Axial, Coronal, and Sagittal branches into a unified 12-class prediction head.",
        "Priority": "Medium",
        "Estimated Effort": "6 Hours",
        "Deliverables": "Multi-view classification network model."
    },
    {
        "Phase": "Phase 5: CV & Loss Strategy",
        "Task ID": "5.1",
        "Task Name": "Validation Split Setup",
        "Description": "Set aside a hold-out test split of 20-30 ground-truth studies. Use GroupKFold (grouped by patient/study) on remaining data.",
        "Priority": "High",
        "Estimated Effort": "4 Hours",
        "Deliverables": "Cross-validation index list mapped in train.csv."
    },
    {
        "Phase": "Phase 5: CV & Loss Strategy",
        "Task ID": "5.2",
        "Task Name": "Soft-Label Loss Function",
        "Description": "Implement a BCE loss function optimized for soft/pseudo labels, with positive class weight scaling.",
        "Priority": "High",
        "Estimated Effort": "4 Hours",
        "Deliverables": "Custom PyTorch Loss module handles soft probability targets."
    },
    {
        "Phase": "Phase 6: Model Training",
        "Task ID": "6.1",
        "Task Name": "3D Augmentation",
        "Description": "Implement 3D random rotations, flips, elastic transforms, and intensity shifts using albumentations/MONAI.",
        "Priority": "High",
        "Estimated Effort": "6 Hours",
        "Deliverables": "Robust train data augmentation pipeline."
    },
    {
        "Phase": "Phase 6: Model Training",
        "Task ID": "6.2",
        "Task Name": "Baseline View Training",
        "Description": "Train individual models on Sagittal and Coronal views. Track validation macro-AUC on the 58 ground-truth labels.",
        "Priority": "High",
        "Estimated Effort": "12 Hours",
        "Deliverables": "Weights of baseline Sagittal/Coronal model checkpoints."
    },
    {
        "Phase": "Phase 6: Model Training",
        "Task ID": "6.3",
        "Task Name": "Hyperparameter Tuning",
        "Description": "Tune learning rate, weight decay, and dropout. Evaluate performance improvement using validation ROC curves.",
        "Priority": "Medium",
        "Estimated Effort": "10 Hours",
        "Deliverables": "Optimized model checkpoints with improved macro-AUC scores."
    },
    {
        "Phase": "Phase 7: Inference & Submit",
        "Task ID": "7.1",
        "Task Name": "View Ensembling",
        "Description": "Aggregate prediction logits from Axial, Coronal, and Sagittal models using a weighted average or meta-classifier.",
        "Priority": "High",
        "Estimated Effort": "6 Hours",
        "Deliverables": "Ensembled predictions maximizing validation AUC."
    },
    {
        "Phase": "Phase 7: Inference & Submit",
        "Task ID": "7.2",
        "Task Name": "Submission Notebook",
        "Description": "Develop an offline-compliant Kaggle submission script. Include package wheel uploads and quick run checks.",
        "Priority": "High",
        "Estimated Effort": "6 Hours",
        "Deliverables": "Functional Kaggle-ready submission file conforming to sample_submission.csv format."
    }
]

df = pd.DataFrame(plan_data)

# Export to CSV
csv_path = "scratch/execution_plan.csv"
df.to_csv(csv_path, index=False)
print(f"Exported plain text execution plan to: {csv_path}")

# Export to Styled Excel
xlsx_path = "scratch/execution_plan.xlsx"
try:
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Execution Plan")
        
        # Style sheet
        workbook = writer.book
        worksheet = writer.sheets["Execution Plan"]
        
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        
        # Define styles
        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid") # Dark Navy Blue
        
        cell_font = Font(name="Segoe UI", size=10)
        phase_fill = PatternFill(start_color="E9EDF4", end_color="E9EDF4", fill_type="solid") # Light Blue Accent
        
        thin_side = Side(border_style="thin", color="D9D9D9")
        cell_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
        
        # Format Headers
        for col_num in range(1, len(df.columns) + 1):
            cell = worksheet.cell(row=1, column=col_num)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = cell_border
        
        # Format Data Rows
        current_phase = ""
        use_zebra = False
        
        for row_idx in range(2, len(df) + 2):
            phase_val = worksheet.cell(row=row_idx, column=1).value
            
            # Switch background fill color on new phase to group them visually
            if phase_val != current_phase:
                current_phase = phase_val
                use_zebra = not use_zebra
            
            row_fill = phase_fill if use_zebra else PatternFill(fill_type=None)
            
            for col_idx in range(1, len(df.columns) + 1):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                cell.font = cell_font
                cell.border = cell_border
                
                # Apply fill
                if row_fill.fill_type:
                    cell.fill = row_fill
                
                # Alignments
                if col_idx in [1, 2, 5, 6]:  # Phase, Task ID, Priority, Effort
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
                    
                # Priority highlights
                if col_idx == 5: # Priority column
                    priority_val = cell.value
                    if priority_val == "High":
                        cell.font = Font(name="Segoe UI", size=10, bold=True, color="C00000") # Dark Red
                    elif priority_val == "Medium":
                        cell.font = Font(name="Segoe UI", size=10, bold=True, color="E36C09") # Orange
                        
        # Auto-fit column widths
        for col in worksheet.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                val = str(cell.value or '')
                lines = val.split('\n')
                for line in lines:
                    if len(line) > max_len:
                        max_len = len(line)
            # Give column extra width for layout breathing room
            worksheet.column_dimensions[col_letter].width = min(max(max_len + 4, 10), 50)
            
    print(f"Successfully generated styled Excel execution plan at: {xlsx_path}")
except Exception as e:
    print(f"Excel styling failed (missing openpyxl?): {e}")
    print("Plain text Excel sheet generated without styling.")
    df.to_excel(xlsx_path, index=False)
