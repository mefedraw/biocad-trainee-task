from sklearn.metrics.pairwise import cosine_similarity

from src.embedding.embedder import PritamdekaEmbedder, NeuMLEmbedder

text1 = "Recent advances in Alzheimer's disease research highlight the growing importance of fluid and imaging biomarkers for early diagnosis and monitoring disease progression. Plasma biomarkers such as phosphorylated tau (p-tau217 and p-tau231) have shown strong correlations with established cerebrospinal fluid markers and amyloid PET imaging, making them promising non-invasive tools. In clinical trials, these biomarkers are increasingly used both for patient selection and as outcome measures, allowing for more efficient evaluation of therapeutic effects. Additionally, tau PET imaging and measures of neuroinflammation, such as TSPO PET, provide complementary insights into disease mechanisms. The integration of these biomarkers into trial design has accelerated drug development and improved the ability to detect subtle changes over time. As a result, the use of multimodal biomarker strategies is becoming central to the development of disease-modifying therapies for Alzheimer's disease."
text2 = "Biomarker-driven approaches are transforming the landscape of Alzheimer's disease clinical research by enabling earlier detection and more precise tracking of disease progression. Blood-based indicators, including phosphorylated tau variants like p-tau217, have demonstrated high agreement with traditional cerebrospinal fluid analyses and amyloid imaging techniques. These markers are now widely incorporated into clinical trials to identify eligible participants and to assess treatment responses. Imaging modalities such as tau PET and translocator protein PET further contribute to understanding pathological changes, including tau accumulation and neuroinflammation. The combination of fluid and imaging biomarkers enhances the sensitivity of clinical studies and supports the evaluation of novel therapeutics. Consequently, biomarker integration is playing a critical role in advancing the development of effective interventions for Alzheimer's disease."
text3 = "Quantum computing uses qubits and superposition to perform complex calculations beyond classical computers."

models = {
    "pritamdeka": PritamdekaEmbedder(),
    "neuml": NeuMLEmbedder(),
}

for name, model in models.items():
    emb = model.encode([text1, text2, text3])
    sim = cosine_similarity(emb)
    print(name, emb.shape, sim)