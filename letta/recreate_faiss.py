import faiss

index = faiss.read_index("/data/IVF.index")
index.make_direct_map()             # builds the id→vector map
faiss.write_index(index, "/data/IVF.direct.index")
