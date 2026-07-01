from knowledge_engine.pipeline.pipeline import Pipeline
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class TestComponent(PipelineComponent):

    def process(self, data):
        print(f"Ricevuto: {data}")
        return data + " -> processed"


def main():

    pipeline = Pipeline()

    pipeline.add_component(TestComponent())

    result = pipeline.run("KnowledgeEngine")

    print(result)


if __name__ == "__main__":
    main()