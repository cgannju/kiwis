# Automatic Kidney Immunofluorescence Whole Slide Image Interpretation with a Vision Foundation Model

## Abstract
Chronic kidney disease represents a progressive and irreversible pathological condition, which imposes a profound burden on global healthcare systems and patient well-being. Current diagnostic workflows relying on manual interpretation of renal biopsy pathology using light and immunofluorescence microscopy, which is labor-intensive, time-consuming, and subjective. While deep learning has revolutionized automated LM whole slide image analysis, immunofluorescence interpretation remains an understudied frontier in computational pathology. Here we establish a large multi-center renal immunofluorescence whole slide image dataset, and develop an annotation-efficient framework integrating a human-in-the-loop training strategy with a large vision foundation model. Extensive experiments show promising results of our method in detecting structures and attributes for renal structures, providing objective comparisons of WSIs through different staining antibodies. The framework also has the potential for broader clinical applications beyond nephropathology to significantly reduce the annotation workload.

## Usage
### Preliminary
```bash
conda create -n kiwis python=3.11.0
pip install -r requirements.txt
```

### First Step: Renal Structures Segmentatioseg
```bash
python seg/scripts/inference.py --save_dir "/xxx/KIWIS/seg/inference_outs" --saved_model_path "/xxx/seg KIWIS/seg/checkpoints/checkpoint.pth.tar" --test_data_dir_path "/xxx/KIWIS/data" --cuda 0
```

### Second Step: Renal Structure's attributes Interpretation
```bash
python cls/scripts/inference.py --type_ckpt_path "/xxx/KIWIS/cls/checkpoints/type.ckpt" --appearance_ckpt_path "/xxx/KIWIS/cls/checkpoints/appearance.ckpt" --distribution_ckpt_path "/xxx/KIWIS/cls/checkpoints/distribution.ckpt" --fluorescence_ckpt_path "/xxx/KIWIS/cls/checkpoints/fluorescence.ckpt" --location_ckpt_path "/xxx/KIWIS/cls/checkpoints/location.ckpt" --posneg_ckpt_path "/xxx/KIWIS/cls/checkpoints/posneg.ckpt" --t_ckpt_path "/xxx/KIWIS/cls/checkpoints/t.ckpt" --image_dir_path "/xxx/KIWIS/data" --mask_dir_path "/xxx/KIWIS/seg/inference_outs" --save_dir_path "/xxx/KIWIS/cls/results" --cuda 0
```

## Acknowledgments
This project builds upon the following open-source projects. We gratefully acknowledge the authors for their work.
- AutoSAM(https://github.com/xhu248/AutoSAM)
- MedSAM(https://github.com/bowang-lab/MedSAM)
- SAM(https://github.com/facebookresearch/segment-anything)
