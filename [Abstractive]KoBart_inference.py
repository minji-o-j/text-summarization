{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gsCJbbiqT0Yl"
      },
      "source": [
        "### 샘플 제출 파일을 다운로드 받습니다."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WHboA_AxyEVm",
        "outputId": "8b62190b-2b65-46bc-aa66-d0b2262536bb"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "--2021-11-29 13:50:41--  https://raw.githubusercontent.com/aifactory-team/AFCompetition/main/1923/1923_test_summary.json\n",
            "Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.111.133, 185.199.109.133, 185.199.110.133, ...\n",
            "Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.111.133|:443... connected.\n",
            "HTTP request sent, awaiting response... 200 OK\n",
            "Length: 11459557 (11M) [text/plain]\n",
            "Saving to: ‘1923_test_summary.json.3’\n",
            "\n",
            "1923_test_summary.j 100%[===================>]  10.93M  --.-KB/s    in 0.05s   \n",
            "\n",
            "2021-11-29 13:50:41 (219 MB/s) - ‘1923_test_summary.json.3’ saved [11459557/11459557]\n",
            "\n"
          ]
        }
      ],
      "source": [
        "!wget https://raw.githubusercontent.com/aifactory-team/AFCompetition/main/1923/1923_test_summary.json"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5Gf5jdORdnnR"
      },
      "source": [
        "### 데이터를 열어봅니다.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sVKIcAInknp2",
        "outputId": "1c76a478-d425-48f5-9865-0a2e62a42c2b"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "There are 4640 paragrphs in the test set.\n",
            "\n",
            "The first paragraph in the test set: \n",
            "{\n",
            "\t\"original\": \"공주시 무령왕릉에서 발견된 청동거울로 청동신수경, 의자손수대경, 수대경 3점이다. 청동신수경은 ‘방격규구문경’이라는 중국 후한의 거울을 모방하여 만든 것이다. 거울 내부에는 반나체 인물상과 글이 새겨져 있는데 이는 한나라의 거울에서 흔히 볼 수 있는 것이다. 의자손수대경은 중국 한대의 수대경을 본떠 만든 방제경이다. 거울 중앙의 꼭지를 중심으로 9개의 돌기가 있고, 안에는 크고 작은 원과 7개의 돌기가 솟아있다. 내부 주위의 테두리에는 명문이 새겨져 있으나 선명하지 못하여 알아볼 수 없다. 수대경 역시 한나라 때 동물 문양을 새겨 넣은 수대경을 본떠서 만들어진 방제경이다. 그러나 한나라 거울에 비해 선이 굵고 무늬가 정교하지 못하다.\",\n",
            "\t\"summary\": \"\",\n",
            "\t\"Meta\": {\n",
            "\t\t\"passage_id\": \"REPORT-cultural_assets-00164-01180\",\n",
            "\t\t\"doc_name\": \"무령왕릉 청동거울 일괄 (武寧王陵 銅鏡 一括)\",\n",
            "\t\t\"category\": \"cul_ass\",\n",
            "\t\t\"author\": null,\n",
            "\t\t\"publisher\": null,\n",
            "\t\t\"publisher_year\": null,\n",
            "\t\t\"doc_origin\": \"문화재청\"\n",
            "\t}\n",
            "}\n"
          ]
        }
      ],
      "source": [
        "import json\n",
        "\n",
        "with open('1923_test_summary.json', encoding='UTF-8') as file:\n",
        "    test_dataset = json.load(file)\n",
        "\n",
        "print(\"There are {} paragrphs in the test set.\\n\".format(len(test_dataset)))\n",
        "\n",
        "print(\"The first paragraph in the test set: \")\n",
        "print(json.dumps(test_dataset[0], indent='\\t', ensure_ascii=False))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ulzW0MuWfJyP"
      },
      "source": [
        "### 참가자가 개발한 알고리즘으로 추론한 요약을 업데이트 합니다.\n",
        "\n",
        "아래는 랜덤하게 3 개의 문장을 뽑아 요약하는 예시입니다."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "id": "QnF0Xf_Idj5Y"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "  1%|          | 54/4640 [00:41<55:52,  1.37it/s]"
          ]
        }
      ],
      "source": [
        "from random import choices\n",
        "from transformers import BartModel, AutoConfig, BartForConditionalGeneration\n",
        "from transformers import PreTrainedTokenizerFast\n",
        "from torch.utils.data import DataLoader, TensorDataset\n",
        "import torch\n",
        "from tqdm import tqdm\n",
        "\n",
        "tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')\n",
        "config = AutoConfig.from_pretrained('gogamza/kobart-summarization')\n",
        "device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\n",
        "model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization', config=config)# 일단 우리는 tokenize 해준 것들을 dataset으로 만들어 준 다음에.. dataloader로 넣어주는 것을 한다..\n",
        "\n",
        "# PATH= './model_epoch_8.pt'\n",
        "# model = torch.load(PATH)\n",
        "model.to(device)\n",
        "\n",
        "my_summaries = []\n",
        "for paragraph in tqdm(test_dataset):\n",
        "    original = paragraph['original']\n",
        "    # print('----------original-----------')\n",
        "    # print(original)\n",
        "    origin= tokenizer(original,\n",
        "                        padding= 'max_length',\n",
        "                        truncation= True,\n",
        "                        stride= 128,\n",
        "                        max_length= 512,\n",
        "                        return_tensors='pt')\n",
        "    origin.to(device)\n",
        "    generated_ids= model.generate(input_ids=origin['input_ids'],\n",
        "                num_beams=10, max_length=120, repetition_penalty=2.0, \n",
        "                length_penalty=1.5, no_repeat_ngram_size=2, early_stopping=True)\n",
        "    decoded_sentence= tokenizer.decode(generated_ids.squeeze(), skip_special_tokens= True, clean_up_tokenization_spaces= True)\n",
        "    # print('----------generate-----------')\n",
        "    # print(decoded_sentence)\n",
        "    # print()\n",
        "\n",
        "    # pick 3 sentences randomly and update 'summary'\n",
        "    paragraph['summary'] = decoded_sentence\n",
        "    my_summaries.append(paragraph)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7QoM7wQPfv58"
      },
      "source": [
        "### 추론한 요약문을 확인합니다."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3Egk30y_fyEC",
        "outputId": "12db7499-c06e-4a01-e6df-b24b9e7106c4"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The first paragraph in the test set: \n",
            "{\n",
            "\t\"original\": \"공주시 무령왕릉에서 발견된 청동거울로 청동신수경, 의자손수대경, 수대경 3점이다. 청동신수경은 ‘방격규구문경’이라는 중국 후한의 거울을 모방하여 만든 것이다. 거울 내부에는 반나체 인물상과 글이 새겨져 있는데 이는 한나라의 거울에서 흔히 볼 수 있는 것이다. 의자손수대경은 중국 한대의 수대경을 본떠 만든 방제경이다. 거울 중앙의 꼭지를 중심으로 9개의 돌기가 있고, 안에는 크고 작은 원과 7개의 돌기가 솟아있다. 내부 주위의 테두리에는 명문이 새겨져 있으나 선명하지 못하여 알아볼 수 없다. 수대경 역시 한나라 때 동물 문양을 새겨 넣은 수대경을 본떠서 만들어진 방제경이다. 그러나 한나라 거울에 비해 선이 굵고 무늬가 정교하지 못하다.\",\n",
            "\t\"summary\": \"주시 무령왕릉에서 발견된 청동신수경은 ‘방격규구문경’이라는 중국 후한의 거울을 모방하여 만든 것이다.  그러나 한나라 거울에 비해 선이 굵고 무늬가 정교하지 못하다.  수자손수대경은 중국 한대의 수대경을 본떠서 만들어진 방제경이다.  거울 중앙의 꼭지를 중심으로 9개의 돌기가 있고, 안에는 크고 작은 원과 7개의돌기가 솟아있다.  내부 주위의 테두리에는 명문이 새겨져 있으나 선명하지 못하고 못하여 알아볼 수 없다. \\n\\n㗡\",\n",
            "\t\"Meta\": {\n",
            "\t\t\"passage_id\": \"REPORT-cultural_assets-00164-01180\",\n",
            "\t\t\"doc_name\": \"무령왕릉 청동거울 일괄 (武寧王陵 銅鏡 一括)\",\n",
            "\t\t\"category\": \"cul_ass\",\n",
            "\t\t\"author\": null,\n",
            "\t\t\"publisher\": null,\n",
            "\t\t\"publisher_year\": null,\n",
            "\t\t\"doc_origin\": \"문화재청\"\n",
            "\t}\n",
            "}\n"
          ]
        }
      ],
      "source": [
        "print(\"The first paragraph in the test set: \")\n",
        "print(json.dumps(my_summaries[0], indent='\\t', ensure_ascii=False))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aFKnavZyfb6N"
      },
      "source": [
        "### 추론한 요약문을 저장합니다.\n",
        "\n",
        "json.dump 메소드로 딕셔너리를 저장할 때에 한글이 깨지지 않게 \n",
        "주의하세요."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PklKBTtcfePA"
      },
      "outputs": [],
      "source": [
        "with open('1923_my_summary_org.json', 'w', encoding=\"UTF-8\") as file:\n",
        "    json.dump(my_summaries, file, indent='\\t', ensure_ascii=False)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "R3XQIj0N5VQW"
      },
      "source": [
        "왼쪽 사이드바에 파일 아이콘을 선택한 후, \"1923_my_summary.json\" 파일을 선택하여, 파일 이름 옆 메뉴에서 다운로드를 클릭 후 파일을 다운로드 받습니다.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZUYWCv6MU-UP"
      },
      "source": [
        "다운로드 받은 1923_my_summary.json 파일을 아래 태스크에 제출합니다. http://aifactory.space/competition/detail/1923"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "name": "1897_baseline_submit.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
