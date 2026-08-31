from prism_infer.engine.llm_engine import GenerationOutput, LLMEngine


def test_generate_annotation_matches_the_returned_schema():
    assert LLMEngine.generate.__annotations__["return"] == list[GenerationOutput]
