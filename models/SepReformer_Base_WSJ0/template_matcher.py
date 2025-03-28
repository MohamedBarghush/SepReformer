import torch

from speechbrain.pretrained import EncoderClassifier
class SepFormerWithTemplateMatching:
    def __init__(self, original_model, device='cuda'):
        self.model = original_model
        self.device = device
        
        # Initialize speaker embedding model on correct device
        self.speaker_embedding = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}  # Critical for device placement
        ).eval()

    def match_template(self, separated_sources, reference_audio):
        """Compare separated sources to reference audio"""
        with torch.no_grad():
            # Ensure reference audio is on same device
            reference_audio = reference_audio.to(self.device)
            
            # Get reference embedding
            ref_embed = self.speaker_embedding.encode_batch(
                reference_audio.squeeze(1),  # Remove channel dimension
                wav_lens=torch.tensor([1.0]).to(self.device)
            ).squeeze(1)

            similarities = []
            for source in separated_sources:
                # Move source to same device and process
                source = source.to(self.device)
                clean_source = source[..., :reference_audio.shape[-1]]
                
                # Get source embedding
                source_embed = self.speaker_embedding.encode_batch(
                    clean_source.squeeze(1),  # Remove channel dimension
                    wav_lens=torch.tensor([1.0]).to(self.device)
                ).squeeze(1)
                
                similarities.append(
                    torch.cosine_similarity(ref_embed, source_embed, dim=-1)
                )

            return separated_sources[torch.argmax(torch.stack(similarities))]