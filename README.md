
# Image2PlantArchitecture

## Overview
A brief summary of the project, its goals, and what problem it solves. (Replace this text with your own summary.)

## Paper Citation
If you use this code, please cite our paper:

```
@article{your2025paper,
  title={Your Paper Title},
  author={Author1, Author2, ...},
  journal={Journal Name},
  year={2025},
  doi={DOI or arXiv link}
}
```

## Repository Structure

```
├── CowpeaSimulator/      # C++ plant simulation code
├── models/              # Model definitions and scripts
├── src/                 # Python source code
├── log/                 # Training and experiment logs
├── environment.yml      # Conda environment file
├── run_experiments.sh   # Script to run experiments
└── README.md            # This file
```

## Installation

1. Clone the repository:
	```bash
	git clone https://github.com/GEMINI-Breeding/Image2PlantArchitecture.git
	cd Image2PlantArchitecture
	```
2. Create the conda environment:
	```bash
	conda env create -f environment.yml -p .env # or replace it with environment_cuda.yml if CUDA available
	conda activate .env 
	```
3. Build the C++ simulator:
	```bash
	cd CowpeaSimulator
	mkdir -p build && cd build
	cmake -DCMAKE_BUILD_TYPE=Release .. && make
	```

## Usage

Provide example commands or scripts to run the main experiments, train models, or use the simulator. For example:

```bash
# Train a model
python src/train.py
```

To test the model, see src/test.ipynb
```python
from models.model import PlantArchitectureModel
checkpoint_path = "heesup/dinov2-small_448_Sideview_gpt2-medium"
model = PlantArchitectureModel.from_pretrained(checkpoint_path,
                                    torch_dtype=torch.float16,).to(device)


# Set the model to evaluation mode
model.eval()

# Extract image size (looking for a number followed by underscore and Sideview or another word)
if "224" in checkpoint_path:
    image_size = 224
elif "448" in checkpoint_path:
    image_size = 448
else:
    image_size = 224

max_length = 4096 * 2
# Extract side_view (True if "Sideview" appears in the path, False otherwise)
side_view = "Sideview" in checkpoint_path

# Try to load the image processor from the encoder's config name
encoder_name = model.encoder.config._name_or_path
image_processor = AutoImageProcessor.from_pretrained(encoder_name)
image_processor.crop_size['width'] = image_size
image_processor.crop_size['height'] = image_size
image_processor.size['shortest_edge'] = image_size


dataset_path = "/home/lion397/datasets/GEMINI/plant_architecture/20250311_Sideview_40Days"
test_dataset = PlantDataset(root_dir=dataset_path,
            process_leaf=True, image_size=image_size,
            side_view=side_view,
            image_processor=image_processor,
            mode='test',
            preload=False, add_sos_token=False)

############## Generate
with torch.no_grad():
    plant_info = torch.tensor(plant_info, dtype=torch.long).unsqueeze(0).to(model.device)  # Ensure plant_info is a tens
    with torch.cuda.amp.autocast():
        result = model.generate(image,
                                decoder_start_token_id=SOS_TOKEN,
                                decoder_input_ids=plant_info,
                                eos_token_id=EOS_TOKEN,
                                pad_token_id=PAD_TOKEN,
                                # do_sample=True,
                                # num_beams=5,
                                # early_stopping=True,  # Stop when EOS is generated
                                output_attentions=False,  # Don't compute attentions
                                max_length=max_length,
                                output_hidden_states=False,  # Don't compute hidden states
                                repetition_penalty=1.1,  # Avoid repetitive sequences
                                use_cache=True,
                                )
        result = result.squeeze().cpu().numpy()[6:]

plant_vec = token2vec(result)
plant_xml = vec2xml(plant_vec)
plant_xml_file_name = f"temp/plant_{idx}_est.xml"
plant_xml = recursive_to_linked(plant_xml)
plant_xml_str = pretty_print_xml(plant_xml)
with open(plant_xml_file_name, "w") as f:
    f.write(plant_xml_str)

re_render_xml(os.path.abspath(temp_folder), os.path.abspath(plant_xml_file_name))

if side_view:
    img, _ = load_sideview_images(temp_folder, plant_xml_file_name.replace("xml","jpeg"), image_size, True)
else:
    img = cv2.imread(plant_xml_file_name.replace("xml","jpeg"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    leaf_area, plant_width, plant_height, leaf_img, _ = process_leaf_image(gt_img, sqaure_crop=True, thr=0.0)
    img = cv2.resize(leaf_img, (image_size, image_size))

image_vis = image[0].permute(1, 2, 0).cpu()
image_vis = cv2.normalize(np.array(image_vis), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

```


## License

Specify your license here (e.g., MIT, Apache 2.0, etc.).

## Contact

For questions or collaborations, contact: [your.email@domain.com]
