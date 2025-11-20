install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:	
	black *.py 

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	cml comment create report.md || true

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
    git pull origin update
    git switch -C update || git switch update
    pip install --upgrade huggingface_hub
    python -m huggingface_hub login --token $(HF)

push-hub: 
	huggingface-cli upload ./App --repo-id=ikram-abhih-2021/Breast-Cancer-Classification --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload ./Model --repo-id=ikram-abhih-2021/Breast-Cancer-Classification --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload ./Results --repo-id=ikram-abhih-2021/Breast-Cancer-Classification --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
