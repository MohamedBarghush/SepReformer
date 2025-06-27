import os
import torch
import csv
import time
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from utils import util_engine, functions
from utils.decorators import *
from torch.utils.tensorboard import SummaryWriter

from .template_matcher import SepFormerWithTemplateMatching

from asteroid.metrics import get_metrics


# @logger_wraps()
class Engine(object):
    def __init__(self, args, config, model, dataloaders, criterions, optimizers, schedulers, gpuid, device):
        
        ''' Default setting '''
        self.engine_mode = args.engine_mode
        self.out_wav_dir = args.out_wav_dir
        self.config = config
        self.gpuid = gpuid
        self.device = device
        self.model = model.to(self.device)
        self.dataloaders = dataloaders # self.dataloaders['train'] or ['valid'] or ['test']
        self.PIT_SISNR_mag_loss, self.PIT_SISNR_time_loss, self.PIT_SISNRi_loss, self.PIT_SDRi_loss = criterions
        self.main_optimizer = optimizers[0]
        self.main_scheduler, self.warmup_scheduler = schedulers
        
        self.pretrain_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "pretrain_weights")
        os.makedirs(self.pretrain_weights_path, exist_ok=True)
        self.scratch_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "scratch_weights")
        os.makedirs(self.scratch_weights_path, exist_ok=True)
        
        self.checkpoint_path = self.pretrain_weights_path if any(file.endswith(('.pt', '.pt', '.pkl')) for file in os.listdir(self.pretrain_weights_path)) else self.scratch_weights_path
        self.start_epoch = util_engine.load_last_checkpoint_n_get_epoch(self.checkpoint_path, self.model, self.main_optimizer, location=self.device)
        
        # Logging 
        util_engine.model_params_mac_summary(
            model=self.model, 
            input=torch.randn(1, self.config['check_computations']['dummy_len']).to(self.device), 
            dummy_input=torch.rand(1, self.config['check_computations']['dummy_len']).to(self.device), 
            metrics=['ptflops', 'thop', 'torchinfo']
            # metrics=['ptflops']
        )
        
        logger.info(f"Clip gradient by 2-norm {self.config['engine']['clip_norm']}")
    
    # @logger_wraps()
    def _train(self, dataloader, epoch):
        self.model.train()
        tot_loss_freq = [0 for _ in range(self.model.num_stages)]
        tot_loss_time, num_batch = 0, 0
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:25}{r_bar}{bar:-10b}', colour="YELLOW", dynamic_ncols=True)
        for input_sizes, mixture, src, _ in dataloader:
            nnet_input = mixture
            nnet_input = functions.apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
            num_batch += 1
            pbar.update(1)
            # Scheduler learning rate for warm-up (Iteration-based update for transformers)
            if epoch == 1: self.warmup_scheduler.step()
            nnet_input = nnet_input.to(self.device)
            self.main_optimizer.zero_grad()
            estim_src, estim_src_bn = torch.nn.parallel.data_parallel(self.model, nnet_input, device_ids=self.gpuid)
            cur_loss_s_bn = 0
            cur_loss_s_bn = []
            for idx, estim_src_value in enumerate(estim_src_bn):
                cur_loss_s_bn.append(self.PIT_SISNR_mag_loss(estims=estim_src_value, idx=idx, input_sizes=input_sizes, target_attr=src))
                tot_loss_freq[idx] += cur_loss_s_bn[idx].item() / (self.config['model']['num_spks'])
            cur_loss_s = self.PIT_SISNR_time_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
            tot_loss_time += cur_loss_s.item() / self.config['model']['num_spks']
            alpha = 0.4 * 0.8**(1+(epoch-101)//5) if epoch > 100 else 0.4
            cur_loss = (1-alpha) * cur_loss_s + alpha * sum(cur_loss_s_bn) / len(cur_loss_s_bn)
            cur_loss = cur_loss / self.config['model']['num_spks']
            cur_loss.backward()
            if self.config['engine']['clip_norm']: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['engine']['clip_norm'])
            self.main_optimizer.step()
            dict_loss = {"T_Loss": tot_loss_time / num_batch}
            dict_loss.update({'F_Loss_' + str(idx): loss / num_batch for idx, loss in enumerate(tot_loss_freq)})
            pbar.set_postfix(dict_loss)
        pbar.close()
        tot_loss_freq = sum(tot_loss_freq) / len(tot_loss_freq)
        return tot_loss_time / num_batch, tot_loss_freq / num_batch, num_batch
    
    # @logger_wraps()
    def _validate(self, dataloader):
        self.model.eval()
        tot_loss_freq = [0 for _ in range(self.model.num_stages)]
        tot_loss_time, num_batch = 0, 0
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="RED", dynamic_ncols=True)
        with torch.inference_mode():
            for input_sizes, mixture, src, _ in dataloader:
                nnet_input = mixture
                nnet_input = functions.apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
                nnet_input = nnet_input.to(self.device)
                num_batch += 1
                pbar.update(1)
                estim_src, estim_src_bn = torch.nn.parallel.data_parallel(self.model, nnet_input, device_ids=self.gpuid)
                cur_loss_s_bn = []
                for idx, estim_src_value in enumerate(estim_src_bn):
                    cur_loss_s_bn.append(self.PIT_SISNR_mag_loss(estims=estim_src_value, idx=idx, input_sizes=input_sizes, target_attr=src))
                    tot_loss_freq[idx] += cur_loss_s_bn[idx].item() / (self.config['model']['num_spks'])
                cur_loss_s_SDR = self.PIT_SISNR_time_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
                tot_loss_time += cur_loss_s_SDR.item() / self.config['model']['num_spks']
                dict_loss = {"T_Loss":tot_loss_time / num_batch}
                dict_loss.update({'F_Loss_' + str(idx): loss / num_batch for idx, loss in enumerate(tot_loss_freq)})
                pbar.set_postfix(dict_loss)
        pbar.close()
        tot_loss_freq = sum(tot_loss_freq) / len(tot_loss_freq)
        return tot_loss_time / num_batch, tot_loss_freq / num_batch, num_batch
    
    # @logger_wraps()
    def _test(self, dataloader, wav_dir=None):
        self.model.eval()
        total_loss_SISNRi, total_loss_SDRi, num_batch = 0, 0, 0
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="grey", dynamic_ncols=True)
        with torch.inference_mode():
            csv_file_name_sisnr = os.path.join(os.path.dirname(__file__),'test_SISNRi_value.csv')
            csv_file_name_sdr = os.path.join(os.path.dirname(__file__),'test_SDRi_value.csv')
            with open(csv_file_name_sisnr, 'w', newline='') as csvfile_sisnr, open(csv_file_name_sdr, 'w', newline='') as csvfile_sdr:
                idx = 0
                writer_sisnr = csv.writer(csvfile_sisnr, quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer_sdr = csv.writer(csvfile_sdr, quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for input_sizes, mixture, src, key in dataloader:
                    if len(key) > 1:
                        raise("batch size is not one!!")
                    nnet_input = mixture.to(self.device)
                    num_batch += 1
                    pbar.update(1)
                    estim_src, _ = torch.nn.parallel.data_parallel(self.model, nnet_input, device_ids=self.gpuid)
                    cur_loss_SISNRi, cur_loss_SISNRi_src = self.PIT_SISNRi_loss(estims=estim_src, mixture=mixture, input_sizes=input_sizes, target_attr=src, eps=1.0e-15)
                    total_loss_SISNRi += cur_loss_SISNRi.item() / self.config['model']['num_spks']
                    cur_loss_SDRi, cur_loss_SDRi_src = self.PIT_SDRi_loss(estims=estim_src, mixture=mixture, input_sizes=input_sizes, target_attr=src)
                    total_loss_SDRi += cur_loss_SDRi.item() / self.config['model']['num_spks']
                    writer_sisnr.writerow([key[0][:-4]] + [cur_loss_SISNRi_src[i].item() for i in range(self.config['model']['num_spks'])])
                    writer_sdr.writerow([key[0][:-4]] + [cur_loss_SDRi_src[i].item() for i in range(self.config['model']['num_spks'])])
                    if self.engine_mode == "test_save":
                        if wav_dir == None: wav_dir = os.path.join(os.path.dirname(__file__),"wav_out")
                        if wav_dir and not os.path.exists(wav_dir): os.makedirs(wav_dir)
                        mixture = torch.squeeze(mixture).cpu().data.numpy()
                        sf.write(os.path.join(wav_dir,key[0][:-4]+str(idx)+'_mixture.wav'), 0.5*mixture/max(abs(mixture)), 8000)
                        for i in range(self.config['model']['num_spks']):
                            src = torch.squeeze(estim_src[i]).cpu().data.numpy()
                            sf.write(os.path.join(wav_dir,key[0][:-4]+str(idx)+'_out_'+str(i)+'.wav'), 0.5*src/max(abs(src)), 8000)
                    idx += 1
                    dict_loss = {"SiSNRi": total_loss_SISNRi/num_batch, "SDRi": total_loss_SDRi/num_batch}
                    pbar.set_postfix(dict_loss)
        pbar.close()
        return total_loss_SISNRi/num_batch, total_loss_SDRi/num_batch, num_batch
    
    # Helper functions for audio processing
    def normalize_audio(self, audio):
        """Normalize audio to have unit energy with safety checks"""
        audio = audio - np.mean(audio)
        energy = np.sum(audio**2)
        if energy > 0:
            scale = np.sqrt(energy / len(audio))
            return audio / (scale + 1e-10)
        return audio

    def pad_to_length(self, audio, target_length):
        """Pad audio to target length"""
        if len(audio) < target_length:
            return np.pad(audio, (0, target_length - len(audio)))
        return audio[:target_length]

    def compute_metrics_safely(self, mixture, source, estimate, sample_rate):
        """Compute metrics with safety checks and improvement metrics"""
        try:
            if not (np.all(np.isfinite(mixture)) and 
                    np.all(np.isfinite(source)) and 
                    np.all(np.isfinite(estimate))):
                print("Warning: Non-finite values detected in audio")
                return None

            if np.all(mixture == 0) or np.all(source == 0) or np.all(estimate == 0):
                print("Warning: Zero signal detected")
                return None

            # Compute metrics for the estimate
            est_metrics = get_metrics(
                mixture,
                source[None, :],  # Add speaker dimension
                estimate[None, :], 
                sample_rate=sample_rate,
                metrics_list=["si_sdr", "sdr", "sir", "sar", "stoi"],
                ignore_metrics_errors=True
            )

            # Compute metrics for the mixture (baseline)
            mix_metrics = get_metrics(
                mixture,
                source[None, :],
                mixture[None, :],  # Mixture as estimate
                sample_rate=sample_rate,
                metrics_list=["si_sdr", "sdr"],
                ignore_metrics_errors=True
            )

            # Calculate improvement metrics
            improvements = {}
            for metric in ["si_sdr", "sdr"]:
                est_val = est_metrics.get(metric)
                mix_val = mix_metrics.get(metric)
                if est_val is not None and mix_val is not None:
                    improvements[f"{metric}i"] = est_val - mix_val
                else:
                    improvements[f"{metric}i"] = None

            # Merge metrics
            merged_metrics = {**est_metrics, **improvements}

            # Check for non-finite values
            for key in merged_metrics:
                if not np.isfinite(merged_metrics[key]):
                    merged_metrics[key] = None
            
            return merged_metrics
        except Exception as e:
            print(f"Error computing metrics: {str(e)}")
            return None

    # @logger_wraps()
    def _inference_sample(self, sample, reference_sample=None, source_sample=None):
        self.model.eval()
        self.fs = self.config["dataset"]["sampling_rate"]
        
        # Load mixture audio
        mixture, _ = librosa.load(sample, sr=self.fs)
        mixture = torch.tensor(mixture, dtype=torch.float32)[None]
        
        # Handle padding
        self.stride = self.config["model"]["module_audio_enc"]["stride"]
        remains = mixture.shape[-1] % self.stride
        if remains != 0:
            padding = self.stride - remains
            mixture_padded = torch.nn.functional.pad(mixture, (0, padding), "constant", 0)
        else:
            mixture_padded = mixture

        with torch.inference_mode():
            nnet_input = mixture_padded.to(self.device)
            
            # Get separated sources from original model
            estim_src, _ = torch.nn.parallel.data_parallel(self.model, nnet_input, device_ids=self.gpuid)
            
            # Save original mixture
            # mixture_clean = torch.squeeze(mixture).cpu().numpy()
            # sf.write(sample[:-4]+'_mixture.wav', 0.9*mixture_clean/max(abs(mixture_clean)), self.fs)

            # Template matching if reference provided
            if reference_sample is not None:
                # Load reference audio
                reference, _ = librosa.load(reference_sample, sr=self.fs)
                reference = torch.tensor(reference, dtype=torch.float32)[None].to(self.device)
                
                # Initialize template matcher
                template_matcher = SepFormerWithTemplateMatching(self.model, self.device)
                
                # Select best matching source
                matched_audio = template_matcher.match_template(
                    separated_sources=estim_src,
                    reference_audio=reference
                )
                
                # Save matched audio
                src = torch.squeeze(matched_audio[..., :mixture.shape[-1]]).cpu().numpy()
                
                filename = os.path.splitext(os.path.basename(sample))[0] + '_matched.wav'
                sf.write(os.path.join("outputs",filename), 0.9*src/max(abs(src)), self.fs)
                
            else:
                # Save all separated sources (original behavior)
                for i in range(self.config['model']['num_spks']):
                    src = torch.squeeze(estim_src[i][..., :mixture.shape[-1]]).cpu().numpy()
                    sf.write(sample[:-4]+'_out_'+str(i)+'.wav', 0.9*src/max(abs(src)), self.fs)

            # If a source input audio is provided, load it and compute metrics
        if source_sample is not None:
            # Load source audio (ground truth)
            mixture, _ = librosa.load(sample, sr=self.fs)
            source, _ = librosa.load(source_sample, sr=self.fs)
            
            # Normalize the audio signals using your helper functions
            mixture_np = self.normalize_audio(mixture)
            source_np = self.normalize_audio(source)
            estimate_audio_norm = self.normalize_audio(src)
            
            # Ensure all signals are the same length (pad or trim as needed)
            # max_len = max(len(mixture_np), len(source_np), len(estimate_audio_norm))
            # mixture_np = self.pad_to_length(mixture_np, max_len)
            source_np = self.pad_to_length(source_np, len(mixture_np))
            estimate_audio_norm = self.pad_to_length(estimate_audio_norm, len(mixture_np))
            
            # Compute metrics using the safe function provided
            metrics = self.compute_metrics_safely(mixture_np, source_np, estimate_audio_norm, self.fs)
            if metrics is not None:
                # Add filename info to metrics
                metrics["filename"] = sample
                
                # Append metrics to CSV file
                csv_file = "metrics.csv"
                try:
                    if not os.path.exists(csv_file):
                        pd.DataFrame([metrics]).to_csv(csv_file, index=False)
                    else:
                        df_existing = pd.read_csv(csv_file)
                        df_new = pd.DataFrame([metrics])
                        df_all = pd.concat([df_existing, df_new], ignore_index=True)
                        df_all.to_csv(csv_file, index=False)
                    print(f"Metrics for sample {sample} appended to {csv_file}.")
                except Exception as e:
                    print(f"Error writing metrics to CSV: {e}")
            else:
                print(f"Metrics computation failed for sample {sample}.")

    
    # @logger_wraps()
    def run(self):
        with torch.cuda.device(self.device):
            writer_src = SummaryWriter(os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/tensorboard"))
            if "test" in self.engine_mode:
                on_test_start = time.time()
                test_loss_src_time_1, test_loss_src_time_2, test_num_batch = self._test(self.dataloaders['test'], self.out_wav_dir)
                on_test_end = time.time()
                logger.info(f"[TEST] Loss(time/mini-batch) \n - Epoch {self.start_epoch:2d}: SISNRi = {test_loss_src_time_1:.4f} dB | SDRi = {test_loss_src_time_2:.4f} dB | Speed = ({on_test_end - on_test_start:.2f}s/{test_num_batch:d})")
                logger.info(f"Testing done!")
            else:
                start_time = time.time()
                if self.start_epoch > 1:
                    init_loss_time, init_loss_freq, valid_num_batch = self._validate(self.dataloaders['valid'])
                else:
                    init_loss_time, init_loss_freq = 0, 0
                end_time = time.time()
                logger.info(f"[INIT] Loss(time/mini-batch) \n - Epoch {self.start_epoch:2d}: Loss_t = {init_loss_time:.4f} dB | Loss_f = {init_loss_freq:.4f} dB | Speed = ({end_time-start_time:.2f}s)")
                for epoch in range(self.start_epoch, self.config['engine']['max_epoch']):
                    valid_loss_best = init_loss_time
                    train_start_time = time.time()
                    train_loss_src_time, train_loss_src_freq, train_num_batch = self._train(self.dataloaders['train'], epoch)
                    train_end_time = time.time()
                    valid_start_time = time.time()
                    valid_loss_src_time, valid_loss_src_freq, valid_num_batch = self._validate(self.dataloaders['valid'])
                    valid_end_time = time.time()
                    if epoch > self.config['engine']['start_scheduling']: self.main_scheduler.step(valid_loss_src_time)
                    logger.info(f"[TRAIN] Loss(time/mini-batch) \n - Epoch {epoch:2d}: Loss_t = {train_loss_src_time:.4f} dB | Loss_f = {train_loss_src_freq:.4f} dB | Speed = ({train_end_time - train_start_time:.2f}s/{train_num_batch:d})")
                    logger.info(f"[VALID] Loss(time/mini-batch) \n - Epoch {epoch:2d}: Loss_t = {valid_loss_src_time:.4f} dB | Loss_f = {valid_loss_src_freq:.4f} dB | Speed = ({valid_end_time - valid_start_time:.2f}s/{valid_num_batch:d})")
                    if epoch in self.config['engine']['test_epochs']:
                        on_test_start = time.time()
                        test_loss_src_time_1, test_loss_src_time_2, test_num_batch = self._test(self.dataloaders['test'])
                        on_test_end = time.time()
                        logger.info(f"[TEST] Loss(time/mini-batch) \n - Epoch {epoch:2d}: SISNRi = {test_loss_src_time_1:.4f} dB | SDRi = {test_loss_src_time_2:.4f} dB | Speed = ({on_test_end - on_test_start:.2f}s/{test_num_batch:d})")
                    valid_loss_best = util_engine.save_checkpoint_per_best(valid_loss_best, valid_loss_src_time, train_loss_src_time, epoch, self.model, self.main_optimizer, self.checkpoint_path, self.wandb_run)
                    # Logging to monitoring tools (Tensorboard && Wandb)
                    writer_src.add_scalars("Metrics", {
                        'Loss_train_time': train_loss_src_time, 
                        'Loss_valid_time': valid_loss_src_time}, epoch)
                    writer_src.add_scalars("Learning Rate", self.main_optimizer.param_groups[0]['lr'], epoch)
                    writer_src.flush()
                logger.info(f"Training for {self.config['engine']['max_epoch']} epoches done!")
