import json
import os
import pandas as pd
import numpy as np

import torch
import transformers
from torch.utils.data import DataLoader, TensorDataset
from transformers import BartModel, AutoConfig, BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
from datasets import load_metric
import random

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AdamW

# from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
import wandb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_dataset(tokenizer, df):    
    origin= tokenizer(list(df['original']),
                        padding= 'max_length',
                        truncation= True,
                        stride= 128,
                        max_length= 512,
                        return_tensors='pt')
    label= tokenizer(list(df['summary']),
                        padding= 'max_length',
                        truncation= True,
                        stride= 128,
                        max_length= 512,
                        return_tensors='pt')

    return TensorDataset(origin['input_ids'],
                         origin['attention_mask'],
                         origin['token_type_ids'],
                         label['input_ids'],
                         label['attention_mask'],
                         label['token_type_ids'])

BATCH_SIZE= 16
EPOCHS= 10
LR= 3e-6

seed_everything(42)


org_train_df= pd.read_json('./data/train_summary.json')
test_df= pd.read_json('./data/test_summary.json')

train_df, val_df= train_test_split(org_train_df, test_size= 0.2, random_state= 42)

# kobart_tokenizer = get_kobart_tokenizer()
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
config = AutoConfig.from_pretrained('gogamza/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization', config=config)# 일단 우리는 tokenize 해준 것들을 dataset으로 만들어 준 다음에.. dataloader로 넣어주는 것을 한다..

print("cls token id", tokenizer.cls_token_id)
print("pad token id" , tokenizer.pad_token_id)
trainset= get_dataset(tokenizer, train_df)
valset= get_dataset(tokenizer, val_df)
# print(trainset)
print(len(trainset), len(valset))

def accuracy_function(real, pred):
    accuracies = torch.eq(real, pred)
    mask = torch.logical_not(torch.eq(real, 3))
    accuracies = torch.logical_and(mask, accuracies)
    accuracies = torch.tensor(accuracies, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.float32)
    
    return torch.sum(accuracies)/torch.sum(mask)

train_loader=DataLoader(trainset,batch_size= BATCH_SIZE, shuffle= True)
val_loader= DataLoader(valset, batch_size= 32, shuffle= True)

torch.cuda.empty_cache()

device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=LR* 0.1)
metric = load_metric('rouge')

run= wandb.init(project= 'bakbak', entity= 'quarter100', name= f'KB')

for epoch in range(EPOCHS):
    model.train()
    model.to(device)

    train_loss, val_loss= 0, 0
    train_acc, val_acc= 0, 0
    pbar= tqdm(enumerate(train_loader), total= len(train_loader))
    for step, batch in pbar:
        # print(batch[0].shape, batch[1].shape, batch[3].shape, batch[4].shape)
        optimizer.zero_grad()
        batch= tuple(t.to(device) for t in batch)
        
        outputs= model(input_ids= batch[0].contiguous(), attention_mask= batch[1].contiguous(),
                    decoder_input_ids= batch[3][:, :-1].contiguous(), labels= batch[3][:, 1:].contiguous())
        loss= outputs.loss
        logits= outputs.logits
        
        preds= logits.argmax(dim= -1)
        # print(f'logits : {logits.shape}')
        # print(f'preds: {preds.shape}')

        acc= accuracy_function(batch[3][:, 1:], preds)
        train_loss += loss.item() / BATCH_SIZE
        train_acc += acc.item()
        # print(acc)

        # print(f'lr : {optimizer.param_groups[0]["lr"]}')
        loss.backward()
        optimizer.step()
        scheduler.step()

        # generated_ids= model.generate(input_ids=batch[0].contiguous(),
        #         num_beams=10, max_length=100, repetition_penalty=2.0, 
        #         length_penalty=1.2, no_repeat_ngram_size=2, early_stopping=True)

        # decoded_preds= tokenizer.batch_decode(generated_ids, skip_special_tokens= True, clean_up_tokenization_spaces= True)
        # decoded_labels= tokenizer.batch_decode(batch[3][:, :].contiguous(), skip_special_tokens= True, clean_up_tokenization_spaces= True)
        # print('---------preds--------------')
        # print(decoded_preds[0])
        # print('---------labels--------------')
        # print(decoded_labels[0])
        

        # if step%BATCH_SIZE==0 :
        #     wandb.log({'lr': optimizer.param_groups[0]["lr"]})


    train_loss_= train_loss/ len(train_loader)
    train_acc_= train_acc/ len(train_loader)
    print(f'epoch: {epoch}, train loss: {train_loss_}, train_acc: {train_acc_}')   
    print(optimizer.param_groups[0]["lr"])

    # PATH= './model_epoch_8.pt'
    # model= torch.load(PATH)
    # model.to(device)
    
    model.eval()
    with torch.no_grad():
        val_pbar= tqdm(enumerate(val_loader), total= len(val_loader))
        for val_step, val_batch in val_pbar:
            
            # 어차피 1번째 token은 special token이므로..
            label= val_batch[3][:, 1:].contiguous()
            val_batch= tuple(t.to(device) for t in val_batch)


            val_outputs= model(input_ids= val_batch[0].contiguous(), attention_mask= val_batch[1].contiguous(),
                    decoder_input_ids= val_batch[3][:, :-1].contiguous(), labels= val_batch[3][:, 1:].contiguous())
            v_loss= val_outputs.loss
            val_loss += v_loss.item() / 32

            # batch * generate ids.. 이런 느낌의 형태일 것임..!
            generated_ids= model.generate(input_ids=val_batch[0].contiguous(),
                num_beams=10, max_length=120, repetition_penalty=2.0, 
                length_penalty=0.8, no_repeat_ngram_size=2, early_stopping=True)

            decoded_preds= tokenizer.batch_decode(generated_ids, skip_special_tokens= True, clean_up_tokenization_spaces= True)
            decoded_labels= tokenizer.batch_decode(label, skip_special_tokens= True, clean_up_tokenization_spaces= True)
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            decoded_preds= tokenizer.batch_decode(generated_ids, skip_special_tokens= True, clean_up_tokenization_spaces= True)
            decoded_labels= tokenizer.batch_decode(val_batch[3][:, :].contiguous(), skip_special_tokens= True, clean_up_tokenization_spaces= True)
            print('---------preds--------------')
            print(decoded_preds[0])
            print('---------labels--------------')
            print(decoded_labels[0])

        result = metric.compute(use_stemmer=True)
        result = {key: value.mid.fmeasure for key, value in result.items()}
        result = {k: round(v, 3) for k, v in result.items()}

        r1, r2, rl = result['rouge1'], result['rouge2'], result['rougeL']
        best = np.mean([r1, r2, rl])
    val_loss_= val_loss/ len(val_loader)
    print(f'Epoch {epoch}, Rouge 1: {result["rouge1"]}, Rouge 2: {result["rouge2"]}, Rouge L: {result["rougeL"]}')
    print(optimizer.param_groups[0]["lr"])
    print(f'val_loss : {val_loss_}')
    wandb.log({'train_acc': train_acc_, 'train_loss': train_loss_, 'val_loss': val_loss_,
    'val rouge1': result["rouge1"], 'val rouge2': result["rouge2"], 'val rougeL': result["rougeL"]})

    torch.save(model, f'./model_epoch_{epoch}.pt')

    





