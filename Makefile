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
	# Valeurs par défaut si secrets absents
	USER_NAME?=ikramabhih
	USER_EMAIL?=i.abhih9312@uca.ac.ma
	git config --global user.name "$(USER_NAME)"
	git config --global user.email "$(USER_EMAIL)"
	# Commit seulement s'il y a des changements
	if [ -n "$$(git status --porcelain)" ]; then \
		git add . ; \
		git commit -m "Update with new results"; \
		git push --force origin HEAD:update; \
	else \
		echo "No changes to commit"; \
	fi

hf-login: 
	pip install -U "huggingface_hub[cli]"
	# Crée ou switch vers la branche update
	git switch -C update || git switch update
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub: 
	huggingface-cli upload ./App --repo-id=ikram-abhih-2021/Breast-Cancer-Classification --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload ./Model --repo-id=ikram-abhih-2021/Breast-Cancer-Classification --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload ./Results --repo-id=ikram-abhih-2021/Breast-Cancer-Classification --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
