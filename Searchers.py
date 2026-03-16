from DataSetHandler import load_gzipped_json
from Indexes import *

class BM25_searcher:
    def __init__(self,thresh=0.5,macro_B=0.75, micro_K=1.2,micro_B=0.75,segment_multiplier=10):
        self.index = InvertedIndexTantivy()
        self.segment_to_doc = load_gzipped_json("Data/segment_to_doc.json.gz")
        self.doc_to_segment_count = load_gzipped_json("Data/doc_to_segment_count.json.gz")
        self.avg_segment_count = 0
        for doc_id,segment_count in self.doc_to_segment_count.items():
            self.avg_segment_count += segment_count
        self.avg_segment_count = self.avg_segment_count / len(self.doc_to_segment_count)
        self.B = macro_B
        self.micro_B = micro_B
        self.micro_K = micro_K
        self.thresh = thresh
        self.segment_multiplier = segment_multiplier


    def search(self, query_string, top_k=5):
        """
        Executes a BM25 search on the 'text' field and returns top-K documents.
        """
        results = self.index.search(query_string, top_k*self.segment_multiplier,k1=self.micro_K,b=self.micro_B)
        return self.doc_to_B25_segment(results)[:top_k]

    def doc_to_max_segment(self, segment_search_results):
        document_scores = {}
        for segment_id,score in segment_search_results:
            prev_score = document_scores.get(self.segment_to_doc[segment_id],0)
            document_scores[self.segment_to_doc[segment_id]] = max(score,prev_score)

        retrieved_documents = [(doc_id,score) for doc_id, score in document_scores.items()]
        retrieved_documents = sorted(retrieved_documents, key=lambda x: x[1], reverse=True)

        thresh_score = self.thresh * retrieved_documents[0][1]
        final_results = []
        for doc_id,score in retrieved_documents:
            if score >= thresh_score:
                final_results.append(doc_id)
        return final_results

    def doc_to_B25_segment(self, segment_search_results):
        K = 1.2
        B = self.B
        document_scores = {}
        for segment_id,score in segment_search_results:
            prev_score = document_scores.get(self.segment_to_doc[segment_id],0)
            document_scores[self.segment_to_doc[segment_id]] = score + prev_score

        tmp = {}
        for doc_id, score in document_scores.items():
            seg_count = self.doc_to_segment_count[doc_id]
            res = (score*(K+1))/(score + K*(1-B+B*(seg_count/self.avg_segment_count)))
            tmp[doc_id] = res
        document_scores = tmp

        retrieved_documents = [(doc_id,score) for doc_id, score in document_scores.items()]
        retrieved_documents = sorted(retrieved_documents, key=lambda x: x[1], reverse=True)

        thresh_score = self.thresh * retrieved_documents[0][1]
        final_results = []
        for doc_id,score in retrieved_documents:
            if score >= thresh_score:
                final_results.append(doc_id)
        return final_results