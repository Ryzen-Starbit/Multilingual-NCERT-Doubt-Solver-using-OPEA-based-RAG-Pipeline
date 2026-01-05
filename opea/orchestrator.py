class Orchestrator:
    def run(self, query, retriever, generator):
        docs = retriever(query)
        return generator(query, docs)
