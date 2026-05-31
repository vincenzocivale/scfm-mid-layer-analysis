# Vendored subset of the GeneCompass package (Chen et al., Cell 2024).
# Only the modules required for inference / embedding extraction are included.
# data_collator is intentionally not imported here — it pulls in a private
# transformers symbol (_torch_collate_batch) that conflicts across versions.
from . import output
from . import utils
from .output import MaskedLMOutputBoth
from .modeling_bert import BertForMaskedLM, BertForSequenceClassification, BertForTokenClassification
from .knowledge_embeddings import KnowledgeBertEmbeddings
