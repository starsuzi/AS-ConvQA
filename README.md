# Realistic Conversational Question Answering with Answer Selection based on Calibrated Confidence and Uncertainty Measurement

Official Code Repository for the paper "Realistic Conversational Question Answering with Answer Selection based on Calibrated Confidence and Uncertainty Measurement" (EACL 2023): https://arxiv.org/abs/2302.05137

## Abstract
Conversational Question Answering (ConvQA) models aim at answering a question with its relevant paragraph and previous question-answer pairs that occurred during conversation multiple times. To apply such models to a real-world scenario, some existing work uses predicted answers, instead of unavailable ground-truth answers, as the conversation history for inference. However, since these models usually predict wrong answers, using all the predictions without filtering significantly hampers the model performance. To address this problem, we propose to filter out inaccurate answers in the conversation history based on their estimated confidences and uncertainties from the ConvQA model, without making any architectural changes. Moreover, to make the confidence and uncertainty values more reliable, we propose to further calibrate them, thereby smoothing the model predictions. We validate our models, Answer Selection-based realistic Conversation Question Answering, on two standard ConvQA datasets, and the results show that our models significantly outperform relevant baselines.

## Installation
```bash
$ conda create -n asconvqa python=3.8
$ conda activate asconvqa
$ conda install tqdm
$ conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch
$ pip install transformers==3.3.1
$ conda install tensorboardX
$ pip install matplotlib
```