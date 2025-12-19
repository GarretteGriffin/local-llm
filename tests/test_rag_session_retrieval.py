from tools.rag import RAGTool


def test_rag_has_documents_tracks_session():
    rag = RAGTool()
    assert rag.has_documents() is False

    rag.add_document("a.txt", "hello world", is_structured=False)
    assert rag.has_documents() is True
