from runtime.capability_matching import CapabilityMatcher, CapabilityProfile, cosine_similarity


def test_cosine_vectors():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([1, 0], [-1, 0]) == -1.0
    assert cosine_similarity([0, 0], [1, 0]) == 0.0
    assert cosine_similarity([1], [1, 0]) == 0.0


def test_semantic_ranking_and_cache():
    matcher = CapabilityMatcher([
        CapabilityProfile("document_creation", "create Word DOCX documents and reports", "document_agent", "document", ("docx",)),
        CapabilityProfile("coding", "write and debug source code", "coding_agent"),
    ])
    ranked, metrics = matcher.rank("make a Microsoft Word file", output="docx")
    assert ranked[0]["capability"] == "document_creation"
    assert metrics["cache_size"] == 2
    matcher.rank("create a DOCX", output="docx")
    assert len(matcher._cache) == 2


def test_document_artifact_requests_use_semantic_document_capability():
    from runtime.agents import MasterAgent
    plan = MasterAgent(None)._capability_plan("create txt file about agentic AI")
    assert plan[0]["agent"] == "document_agent"
    assert plan[0]["capability"] == "document_creation"


def test_python_file_request_about_rb_tree_routes_to_coding():
    from runtime.agents import MasterAgent
    plan = MasterAgent(None)._capability_plan("create py file about rb trees")
    assert plan[0]["agent"] == "coding_agent"
    assert plan[0]["capability"] == "code_debugging"
