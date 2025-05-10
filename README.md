# UVQA
[ICLR'25] Official code for "Can Video LLMs Refuse to Answer? Alignment for Answerability in Video Large Language Models"

### Dataset Preparation
* **Relation & Object related categories** (MOMA-LRG)
You can refer following GitHub link to get the video for the dataset [MOMA-LRG](https://github.com/StanfordVL/moma). \
You can download the annotations with their script or directly from [Google Drive](https://drive.google.com/file/d/1stizUmyHY6aNxxbxUPD5DvoibBvUrKZW/view?usp=sharing):


* **Attributed related categories**  (DiDeMo)
Now all the way that we can use is not applicable. I'm looking for the way to download the YFCC100M videos.

* Answerable dataset (Video-ChatGPT)
You can refer following GitHub link to get the video for the dataset [Video-ChatGPT/data](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main/data). \
You can download the annotations with provided link directly from [Link](https://mbzuaiac-my.sharepoint.com/personal/hanoona_bangalath_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhanoona%5Fbangalath%5Fmbzuai%5Fac%5Fae%2FDocuments%2FVideo%2DChatGPT%2FData%5FCode%5FModel%5FRelease%2FVideoInstruct%5FDataset%2Ejson&parent=%2Fpersonal%2Fhanoona%5Fbangalath%5Fmbzuai%5Fac%5Fae%2FDocuments%2FVideo%2DChatGPT%2FData%5FCode%5FModel%5FRelease&ga=1)

* **Frame Extraction**
For the fast process, we extract the frame from the video in advance. 
After extraction, you can add each frame folder in `training_code/data_utils/data_constant.py`.

