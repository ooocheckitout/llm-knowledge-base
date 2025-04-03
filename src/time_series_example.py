from typing import List

import torch
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, date, timedelta

import chromadb
import numpy as np

client = chromadb.PersistentClient("chroma_test")

if "events" in client.list_collections():
    client.delete_collection("events")

collection = client.create_collection("events", metadata={
    "hnsw:space": "l2",
})


def normalize_array(arr):
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return norm_arr


@dataclass(frozen=True)
class Event:
    tenant_id: uuid.UUID
    event_id: uuid.UUID
    event_op_code: int
    event_user_id: int
    event_folder_id: int
    event_time: datetime

    def to_embedding(self):
        return [
            int(self.event_time.timestamp()),
            int(self.event_op_code),
            int(self.event_user_id),
            int(self.event_folder_id)
        ]


def generate_event(
        *, tenant_id: uuid = None, event_date: date = None, operation_code: int = None, event_user_id: int = None,
        event_folder_id: int = None
):
    return Event(
        tenant_id=tenant_id or uuid.uuid4(),
        event_op_code=operation_code or random.choice([10000, 20000, 30000]),
        event_user_id=event_user_id or random.choice([x for x in range(1, 50)]),
        event_folder_id=event_folder_id or random.choice([x for x in range(1, 100)]),
        event_time=datetime.combine(event_date or date.today(), datetime.min.time()),
        event_id=uuid.uuid4(),
    )


tenant_id = uuid.uuid4()

today = date.today()
week_start = today - timedelta(days=today.weekday())
weekdays = [week_start + timedelta(days=week_day_index) for week_day_index in range(0, 7)]
operation_codes = [x for x in range(3)]
user_ids = [x for x in range(5)]
folder_ids = [x for x in range(10)]

events = [
    generate_event(tenant_id=tenant_id, event_date=weekday, operation_code=operation_code, event_user_id=user_id,
                   event_folder_id=folder_id)
    for weekday in weekdays
    for operation_code in operation_codes
    for user_id in user_ids
    for folder_id in folder_ids
    for _ in range(5)
]

print(len(events), weekdays)

ids = [str(event.event_id) for event in events]
embeddings = [event.to_embedding() for event in events]
documents = [str(event) for event in events]

metadata = {"source": "events", "tenant_id": str(tenant_id)}
metadatas = [metadata for _ in range(len(events))]

collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

print(
    len(collection.get(where={"$and": [{"source": "events"}, {"tenant_id": str(tenant_id)}]}, include=[])["ids"])
)

today_event = generate_event(
    tenant_id=tenant_id,
    event_date=today,
    operation_code=operation_codes[0],
    event_user_id=user_ids[0],
    event_folder_id=folder_ids[0]
)

print(today_event)

results = collection.query(
    [today_event.to_embedding()],
    where={"$and": [{"source": "events"}, {"tenant_id": str(tenant_id)}]},
    n_results=50,
    include=["documents", "distances"]
)
for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(distance, doc)

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0].tolist()
