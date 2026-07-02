from src.DatabaseLayer.DatabasesManagers.tmp_invertedindex import InvertedIndex


class SectionToParagraphSearcher:
    def __init__(self, index_path, doc_to_segment_count, thresh=0.5, macro_B=0.75):
        # We hit the Paragraph Inverted Index because our target documents are mapped as paragraphs
        self.index = InvertedIndex(index_path)

        # CRITICAL: doc_to_segment_count must be the paragraph count of the target cases,
        # NOT the section count. We are normalizing against the target's length.
        self.doc_to_segment_count = doc_to_segment_count

        self.avg_segment_count = sum(self.doc_to_segment_count.values())
        if len(self.doc_to_segment_count) > 0:
            self.avg_segment_count = self.avg_segment_count / len(self.doc_to_segment_count)
        else:
            self.avg_segment_count = 1

        self.B = macro_B
        self.thresh = thresh

    def search(self, query_sections, top_k=5, pool_size_per_query=100):
        """
        Executes a BM25 search for a full case using a list of sections.
        Aggregates the target paragraph hits and returns the top-K scored documents.
        """
        aggregated_segment_scores = {}

        # 1. Execute search for every section in the query case
        for section in query_sections:
            # Feed the section string as a query to the paragraph index
            hits = self.index.search(section, pool_size_per_query, "paragraphs")

            # 2. Pool the scores at the target segment (paragraph) level
            for case_id, para_id, score in hits:
                key = (case_id, para_id)
                # Sum the score if multiple query sections hit this same target paragraph
                aggregated_segment_scores[key] = aggregated_segment_scores.get(key, 0.0) + score

        # Reconstruct the tuple format
        combined_segment_results = [
            (case_id, para_id, score)
            for (case_id, para_id), score in aggregated_segment_scores.items()
        ]

        # 3. Roll up to the document level using actual Macro-BM25 logic
        scored_documents = self.doc_to_macro_score(combined_segment_results)

        return scored_documents[:top_k]

    def doc_to_macro_score(self, segment_search_results):
        raw_document_scores = {}

        # 1. SUM all the paragraph segment scores for each document
        for case_id, paragraph_id, score in segment_search_results:
            raw_document_scores[case_id] = raw_document_scores.get(case_id, 0.0) + score

        final_documents = []

        # 2. Apply Length Normalization (The true Macro-BM25 formula)
        for doc_id, raw_score in raw_document_scores.items():
            # Get how many total paragraphs this target document has
            doc_length = self.doc_to_segment_count.get(doc_id, 1)

            # Calculate the length penalty.
            # If B=0.75, long docs are penalized heavily. If B=0, no penalty is applied.
            length_penalty = (1 - self.B) + self.B * (doc_length / self.avg_segment_count)

            # Normalize the score
            normalized_score = raw_score / length_penalty

            final_documents.append((doc_id, normalized_score))

        # 3. Sort by the new normalized score
        final_documents.sort(key=lambda x: x[1], reverse=True)

        return final_documents