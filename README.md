## MDA-Dual
## A Literature-Guided Dual-Strategy Framework for Microbe–Disease Association Prediction

__Abstract__

__Motivation:__ Existing microbe–disease association studies commonly focus on whether an association exists, whereas the direction of the association is also biologically important. A microbe may be enriched or depleted in a disease state, and unlabeled microbe–disease pairs should not automatically be treated as confirmed negative associations. In addition, the relevant evidence is distributed across the biomedical literature and multiple association databases, making it necessary to integrate literature-derived knowledge, associations, and complementary node features in a reproducible framework.

__Results:__ We developed a literature-guided framework that combines large language model-based relation extraction with two complementary prediction strategies. The literature pipeline reads PDF, DOCX, or TXT articles and extracts structured microbe–disease relations and supporting evidence. The Two-step strategy first applies positive–unlabeled learning to estimate association existence and then distinguishes increase from decrease associations using fold-specific knowledge graph representations and conventional classifiers. The One-step strategy constructs 402-dimensional microbe features and 391-dimensional disease features by integrating a reproducible 128-dimensional latent representation, external descriptors, global Gaussian interaction profile features, and separately reduced Peryton, Disbiome, and HMDAD feature blocks. Their concatenation produces a 793-dimensional representation for three-class XGBoost prediction of decrease, pseudo-no-relation, and increase associations. The public repository provides 3,194 literature-derived knowledge-graph edges and an independent association space containing 11,400 labeled associations. Saved models, evaluation figures, and final candidate predictions are not distributed.



