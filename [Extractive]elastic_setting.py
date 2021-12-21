import os
from subprocess import Popen, PIPE, STDOUT
from elasticsearch import Elasticsearch
import json
import pandas as pd
from tqdm import tqdm
import time
import re


# 172.17.0.2
es_server = Popen(
    ["./elasticsearch-7.15.1/bin/elasticsearch"],
    stdout=PIPE,
    stderr=STDOUT,
    preexec_fn=lambda: os.setuid(1),
)

INDEX_NAME = "document"


INDEX_SETTINGS = {
    "settings": {
        "index": {
            "analysis": {
                "analyzer": {
                    "korean": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "filter": ["shingle"],
                    }
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "korean",
                "search_analyzer": "korean",
            },
            "title": {
                "type": "text",
                "analyzer": "korean",
                "search_analyzer": "korean",
            },
        }
    },
}
try:
    es.transport.close()
except:
    pass
es = Elasticsearch()
if es.indices.exists(INDEX_NAME, request_timeout=10):
    es.indices.delete(index=INDEX_NAME, request_timeout=10)

print(es.info())
