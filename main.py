from fastapi import FastAPI

from search_api import NeuralSearcher
from index_builder import IndexBuilder

app = FastAPI()

neural_searcher = NeuralSearcher(collection_name="startups")

@app.get("/api/search")
def search_startup(q: str):
    return {"result": neural_searcher.search(q)}

if __name__ == "__main__":
    import uvicorn

    #start create index
    builder = IndexBuilder()
    builder.index_vectors("startups")

    uvicorn.run(app, host="0.0.0.0", port=4200)