import torch
import torch.nn as nn
import torch.nn.functional as F
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
from config.settings import DOMAINS, EMBEDDING_DIM
from src.models.domain_classifier import DomainClassifier
from src.models.domain_embedders import DomainEmbedderManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingComposer:
    """
    Simplified composer - removed failing approaches
    """

    def __init__(self):
        """Initialize with only necessary components"""
        logger.info("Initializing EmbeddingComposer...")

        # Keep domain classifier for analysis only
        self.classifier = DomainClassifier()
        self.embedder_manager = DomainEmbedderManager(load_all=False)  # Don't load all
        self.domains = DOMAINS

        # Add INSTRUCTOR for task-aware embeddings (this actually works)
        try:
            self.instructor = INSTRUCTOR('hkunlp/instructor-large')
            self.has_instructor = True
        except:
            logger.warning("INSTRUCTOR not available, using fallback")
            self.has_instructor = False

        logger.info("EmbeddingComposer initialized")

    def compose(self, text: str, method: str = 'instructor', task: str = 'classification') -> np.ndarray:
        """
        Simplified composition - only use what works
        """
        if method == 'instructor' and self.has_instructor:
            # Use INSTRUCTOR with task-specific instruction
            instruction = self._get_instruction(task)
            embedding = self.instructor.encode([[instruction, text]])[0]
            return embedding
        else:
            # Fallback to single best domain (news/MPNet performs best)
            return self.embedder_manager.get_embedding(text, 'news')

    def compose_batch(self, texts: List[str], method: str = 'instructor',
                     task: str = 'classification', batch_size: int = 32) -> np.ndarray:
        """
        Batch processing - simplified
        """
        if method == 'instructor' and self.has_instructor:
            instruction = self._get_instruction(task)
            input_pairs = [[instruction, text] for text in texts]
            return self.instructor.encode(input_pairs, batch_size=batch_size)
        else:
            # Use news domain (MPNet) as it performs best
            return self.embedder_manager.get_batch_embeddings(texts, 'news')

    def _get_instruction(self, task: str) -> str:
        """Get task-specific instruction"""
        instructions = {
            'classification': 'Represent the text for classification: ',
            'similarity': 'Represent the text for similarity comparison: ',
            'retrieval': 'Represent the document for retrieval: ',
            'clustering': 'Represent the text for clustering: '
        }
        return instructions.get(task, instructions['classification'])