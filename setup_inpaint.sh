git clone https://github.com/geekyutao/Inpaint-Anything.git
pip install -r requirements.txt
pip install -e Inpaint-Anything/segment_anything
pip install -r Inpaint-Anything/lama/requirements.txt
pip install pytorch-lightning==1.8.4
cd Inpaint-Anything/pretrained_models
gdown 1-nH9yyQR-Hj5xZnH-pCQUKwDaIj0WOow --folder
gdown 1TKW5YAIGIqHgOeuoXuBxDCnI2wPRqxfP
