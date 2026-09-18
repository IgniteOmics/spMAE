import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss as mse
import pytorch_lightning as pl
import numpy as np
from evaluate import evaluate, db_sil, pas_chaos
from loss_contrastive import CrossModalCLIPLoss
from util import mclust_R, dopca
import pandas as pd


class ContextGatedAttention(nn.Module):
    def __init__(self, in_feat, hidden_dim=128):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.LayerNorm(in_feat * 2),    
            nn.Linear(in_feat * 2, hidden_dim),
            nn.GELU(),                    
            nn.Linear(hidden_dim, in_feat * 2) 
        )
        self.post_fusion_norm = nn.LayerNorm(in_feat)
        
    def forward(self, emb1, emb2):
        # emb1 (RNA), emb2 (ATAC): (B, D)
        cat_emb = torch.cat([emb1, emb2], dim=1) # (B, 2*D)
        scores = self.attention_net(cat_emb)     # (B, 2*D)
        score1, score2 = torch.chunk(scores, 2, dim=1) # (B, D), (B, D)
        raw_scores = torch.stack([score1, score2], dim=0) # (2, B, D)
        alpha = torch.softmax(raw_scores, dim=0)          # (2, B, D)
        alpha1, alpha2 = alpha[0], alpha[1] # (B, D)
        emb_combined = alpha1 * emb1 + alpha2 * emb2
        emb_combined = self.post_fusion_norm(emb_combined)
        return emb_combined, alpha


class spMAE(nn.Module):
    def __init__(self, num_feat1, num_feat2, n_clusters, hidden_size=128, dropout=0, 
                 masked_data_weight=.75, mask_loss_weight=0.7):
        super().__init__()
        self.num_feat1 = num_feat1
        self.num_feat2 = num_feat2
        self.mddim = hidden_size
        self.masked_data_weight = masked_data_weight
        self.mask_loss_weight = mask_loss_weight
        self.n_clusters = n_clusters
        self.alpha = 1.0

        # RNA Encoder
        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_feat1, 256),
            nn.LayerNorm(256),
            nn.Mish(inplace=True),
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size)
        )
        # ATAC Encoder
        self.encoder1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_feat2, 256),
            nn.LayerNorm(256),
            nn.Mish(inplace=True),
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size)
        )

        # self.projection_head = nn.Linear(hidden_size, hidden_size)
        self.mask_predictor = nn.Linear(hidden_size, num_feat1)
        self.mask_predictor1 = nn.Linear(hidden_size, num_feat2)
        
        self.decoder = nn.Linear(in_features=hidden_size+num_feat1, out_features=num_feat1)
        self.decoder1 = nn.Linear(in_features=hidden_size+num_feat2, out_features=num_feat2)

        self.atten = ContextGatedAttention(hidden_size)


    def forward_mask(self, x, x1, x_raw, x1_raw):
        latent = self.encoder(x)
        latent1 = self.encoder1(x1)
        latent_fused, alpha = self.atten(latent, latent1)

        predicted_mask = self.mask_predictor(latent_fused)
        predicted_mask1 = self.mask_predictor1(latent_fused)
        reconstruction = self.decoder(torch.cat([latent_fused, torch.sigmoid(predicted_mask)], dim=1))
        reconstruction1 = self.decoder1(torch.cat([latent_fused, torch.sigmoid(predicted_mask1)], dim=1))
        return latent_fused, predicted_mask, reconstruction, predicted_mask1, reconstruction1, alpha, latent, latent1

    def feature(self, x, x1, x_smooth, x_smooth1):

        fea, _, _, _, _, alpha, latent, latent1 = self.forward_mask(x, x1, x, x1)
        return fea, alpha, latent, latent1


# --- Lightning Module ---
class spMAE_Lightning(pl.LightningModule):
    def __init__(self, 
                 num_feat1, 
                 num_feat2, 
                 n_classes, 
                 adj_matrix=None,
                 alpha=1.0,
                 hidden_size=128, 
                 lr=1e-3, 
                 epochs=100,
                 mask_prob_rna=0.4,
                 mask_prob_atac=0.4,
                 mask_loss_weight=0.7,
                 masked_data_weight=0.75,
                 save_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.num_feat1 = num_feat1
        self.num_feat2 = num_feat2
        self.model = spMAE(num_feat1, num_feat2, n_classes, hidden_size, 
                           mask_loss_weight=mask_loss_weight, masked_data_weight=masked_data_weight)
        self.alpha = alpha

        self.lr = lr
        self.max_epochs_config = epochs
        self.n_classes = n_classes
        self.mask_prob_rna = mask_prob_rna
        self.mask_prob_atac = mask_prob_atac
        self.dec_initialized = False
        self.val_history = []
        self.save_path = save_path
        
        self.validation_step_outputs = []
        self.clip_loss_fn = CrossModalCLIPLoss(initial_temperature=0.07)

    def apply_noise(self, X, p):
        """
        X: (batch_size, num_features)
        p_vec: (num_features,) 每一列的mask概率
        """
        num_features = X.shape[1]

        p = torch.as_tensor(p, device=X.device, dtype=X.dtype)
        if p.shape[0] > num_features:
            p = p[:num_features]
        p_mat = p.unsqueeze(0).expand_as(X)
        
        should_swap = torch.bernoulli(p_mat)
        
        idx = torch.randperm(X.shape[0], device=X.device)
        corrupted_X = torch.where(should_swap == 1, X[idx], X)
        
        masked = (corrupted_X != X).float()
        
        return corrupted_X, masked

    def loss_fn(self, x, y, alpha=2):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    def training_step(self, batch, batch_idx):
        x, x1, x_smooth, x_smooth1, spatial, y, indices = batch

        x_corrupted, mask = self.apply_noise(x, p=self.mask_prob_rna)
        x1_corrupted, mask1 = self.apply_noise(x1, p=self.mask_prob_atac)

        latent_fused, predicted_mask, reconstruction, predicted_mask1, reconstruction1, alpha, latent, latent1 = \
            self.model.forward_mask(x_corrupted, x1_corrupted, x, x_smooth1)

        # RNA Loss
        w_nums = mask * self.model.masked_data_weight + (1 - mask) * (1 - self.model.masked_data_weight)
        loss_rna_recon = (1 - self.model.mask_loss_weight) * torch.mul(w_nums, mse(reconstruction, x_smooth, reduction='none')).mean()
        loss_rna_mask = self.model.mask_loss_weight * bce_logits(predicted_mask, mask, reduction="mean")

        # ATAC Loss
        w_nums1 = mask1 * self.model.masked_data_weight + (1 - mask1) * (1 - self.model.masked_data_weight)
        loss_atac_recon = (1 - self.model.mask_loss_weight) * torch.mul(w_nums1, mse(reconstruction1, x_smooth1, reduction='none')).mean()
        loss_atac_mask = self.model.mask_loss_weight * bce_logits(predicted_mask1, mask1, reduction="mean")
        loss_clip = self.clip_loss_fn(latent, latent1)*0.1

        loss_total = loss_rna_recon + loss_rna_mask + loss_atac_recon + loss_atac_mask + loss_clip
        # WandB Logging
        log_dict = {
            "train/loss_total": loss_total,
            "train/loss_rna_recon": loss_rna_recon,
            "train/loss_rna_mask": loss_rna_mask,
            "train/loss_atac_recon": loss_atac_recon,
            "train/loss_atac_mask": loss_atac_mask,
            "train/loss_clip": loss_clip
        }
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.loss_1 = loss_total
        return loss_total
    
    @torch.no_grad()
    def get_latents(self, dataloader):
        was_training = self.training
        self.eval()

        device = self.device
        
        all_latent_fused = []
        all_latent = []
        all_latent1 = []
        all_rna_weight = []
        all_atac_weight = []
        
        for batch in dataloader:
            x, x1, x_smooth, x_smooth1, spatial, y, indices = batch
            x_smooth = x_smooth.to(device)
            x_smooth1 = x_smooth1.to(device)
            
            latent_fused, alpha, latent, latent1 = self.model.feature(x_smooth, x_smooth1, x_smooth, x_smooth1)
    
            all_latent_fused.append(latent_fused.detach().cpu())
            all_rna_weight.append(alpha[0].detach().cpu())
            all_atac_weight.append(alpha[1].detach().cpu())

        results = {
            'latent_fused': torch.cat(all_latent_fused, dim=0).numpy(),
            'rna_weight': torch.cat(all_rna_weight, dim=0).numpy(),
            'atac_weight': torch.cat(all_atac_weight, dim=0).numpy()
        }
        if was_training:
            self.train()
            
        return results

    # def validation_step(self, batch, batch_idx):
    #     x, x1, x_smooth, x_smooth1, spatial, y, indices = batch
    #     latent_fused, alpha, latent, latent1 = self.model.feature(x_smooth, x_smooth1, x_smooth, x_smooth1)
    #     self.validation_step_outputs.append({
    #         'latent_fused': latent_fused.detach().cpu(),
    #         'latent': latent.detach().cpu(),
    #         'latent1': latent1.detach().cpu(),
    #         'label': y.cpu(),
    #         'spatial': spatial.cpu(),
    #         'alpha': alpha.detach().cpu()
    #     })
    #     return latent

    # def on_validation_epoch_end(self):
    #     outputs = self.validation_step_outputs
    #     if not outputs:
    #         return
    #     all_latent_fused = torch.cat([x['latent_fused'] for x in outputs], dim=0).numpy()
    #     all_latent1 = torch.cat([x['latent1'] for x in outputs], dim=0).numpy()
    #     all_latent = torch.cat([x['latent'] for x in outputs], dim=0).numpy()
    #     all_labels = torch.cat([x['label'] for x in outputs], dim=0).numpy()
    #     all_spatial = torch.cat([x['spatial'] for x in outputs], dim=0).numpy()
    #     rna_weight = torch.cat([x['alpha'][0] for x in outputs], dim=0).numpy()
    #     atac_weight = torch.cat([x['alpha'][1] for x in outputs], dim=0).numpy()
    #     if self.save_path is not None:
    #         np.savetxt(self.save_path+f"/epoch_weight0_{self.current_epoch}.txt", rna_weight)
    #         np.savetxt(self.save_path+f"/epoch_weight1_{self.current_epoch}.txt", atac_weight)
    #         np.savetxt(self.save_path+f"/epoch_{self.current_epoch}.txt", all_latent_fused)
    #         np.savetxt(self.save_path+f"/epoch_latent0_{self.current_epoch}.txt", all_latent)
    #         np.savetxt(self.save_path+f"/epoch_latent1_{self.current_epoch}.txt", all_latent1)

        
    #     # PCA + mclust
    #     all_latents_pca = dopca(all_latent_fused, dim=20)
    #     pred_labels_mclust = mclust_R(all_latents_pca, self.n_classes)
        
    #     db1, sil1 = db_sil(all_latents_pca, pred_labels_mclust)
    #     metrics = {
    #         "val/db_mclust": db1,
    #         "val/sil_mclust": sil1,
    #     }
        
    #     if (all_labels >= 0).any():
    #         nmi1, ari1, hom1, ami1 = evaluate(all_labels, pred_labels_mclust)

    #         metrics.update({
    #             "val/ari_mclust": ari1,
    #             "val/nmi_mclust": nmi1,
    #             "val/ami_mclust": ami1,
    #             "val/hom_mclust": hom1,
    #         })
    #         print(f"\n[Epoch {self.current_epoch}]| ARI_mclust: {ari1:.4f}")
        
    #     if not np.all(all_spatial == 0):
    #         pas1, chaos1 = pas_chaos(pred_labels_mclust, all_spatial)
        
    #         metrics.update({
    #             "val/pas_mclust": pas1,
    #             "val/chaos_mclust": chaos1,
    #         })
            
    
    #     self.log_dict(metrics, prog_bar=True, logger=True)
    #     row = metrics
    #     row["epoch"] = self.current_epoch
    #     self.val_history.append(row)
    #     self.val_df = pd.DataFrame(self.val_history)
    #     self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
