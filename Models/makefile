make download_llms:
	source myenv/bin/activate
	pip install -U huggingface_hub pyyaml mlx-lm mlx-vlm
	python3 Codes/Download_Models.py.     > Codes/Download_Models.txt
	python3 Codes/Model_RAM_Usage.py      > Codes/Model_RAM_Usage.txt
	python3 Codes/Query_Models_Toy.py     > Codes/Query_Models_Toy.txt
	python3 Codes/Temperature_Analysis.py > Codes/Temperature_Analysis.txt