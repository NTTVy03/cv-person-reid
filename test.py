import faiss

index = faiss.read_index('output.index')

print(index.ntotal)