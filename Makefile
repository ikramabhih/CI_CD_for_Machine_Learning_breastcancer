install:
<TAB>pip install --upgrade pip &&\
<TAB>pip install -r requirements.txt

format:
<TAB>black *.py

train:
<TAB>python train.py

eval:
<TAB>echo "## Model Metrics" > report.md
<TAB>cat ./Results/metrics.txt >> report.md
<TAB>echo '\n## Confusion Matrix Plot' >> report.md
<TAB>echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
<TAB>cml comment create report.md || true

update-branch:
<TAB>git config --global user.name $(USER_NAME)
<TAB>git config --global user.email $(USER_EMAIL)
<TAB>git commit -am "Update with new results"
<TAB>git push --force origin HEAD:update

hf-login:
<TAB>git pull origin update
<TAB>git switch -C update || git switch update
<TAB>pip install --upgrade huggingface_hub
<TAB>python -m huggingface_hub login --token $(HF)

push-hub:
<TAB>python -m huggingface_hub upload ./App --repo-id ikram-abhih-2021/Breast-Cancer-Classification --repo-type space --commit-message "Sync App files"
<TAB>python -m huggingface_hub upload ./Model --repo-id ikram-abhih-2021/Breast-Cancer-Classification --repo-type space --commit-message "Sync Model"
<TAB>python -m huggingface_hub upload ./Results --repo-id ikram-abhih-2021/Breast-Cancer-Classification --repo-type space --commit-message "Sync Results"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
