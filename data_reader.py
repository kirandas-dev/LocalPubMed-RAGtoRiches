import h5py
import numpy as np
import torch
import sqlite3
import heapq
from tqdm import tqdm

import torch

class DataReader:
    def __init__(self, hdf5_file, db_file):
        self.hdf5_file = hdf5_file
        self.db_file = db_file
        self._load_hdf5_data()
        self.dimension = self.doc_vectors.shape[1]  # Infer dimension directly from loaded data
        self.connection = self._connect_db()
    
    def _load_hdf5_data(self):
        self.h5file = h5py.File(self.hdf5_file, 'r')
        self.doc_vectors = self.h5file['doc_vectors']
        self.doc_ids = self.h5file['doc_ids']
        self.num_rows = self.doc_vectors.shape[0]
        # self.num_rows = 1000
    
    def _connect_db(self):
        connection = sqlite3.connect(self.db_file)
        return connection
    
    def fetch_document_info(self, pmids):
        cursor = self.connection.cursor()

        query = "SELECT pmid, title, authors, abstract, publication_year FROM articles WHERE pmid IN ("
        query += ",".join([str(id) for id in pmids])
        query += ")"

        cursor.execute(query)

        rows = cursor.fetchall()
        cursor.close()
        # Convert rows to list of dictionaries
        documents = []
        for row in rows:
            documents.append({
                'pmid': row[0],
                'title': row[1],
                'authors': row[2],
                'abstract': row[3],
                'publication_year': row[4]
            })
        return documents