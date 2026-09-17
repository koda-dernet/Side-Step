"""Epoch audio preview sampling.

Generates a short audio sample every N epochs during training using the
live in-memory model (current adapter weights, EMA-applied when active),
so you can hear the adapter converge instead of guessing from loss curves.


Design notes (mirrors the proven approach in ace-lora-trainer-custombuild):

- The full ``AceStepConditionGenerationModel`` is already resident during
  training and carries a complete flow-matching sampler
  (``generate_audio``), so no checkpoint reload is needed.
- Conditioning (caption/bpm/key from the first dataset sample's metadata
  + generic verse/chorus lyrics) is encoded ONCE on the first sample:
  the Qwen3 text encoder is loaded, used, and immediately unloaded.
  The resulting tensors are cached on CPU for the rest of the run.
- The VAE is loaded only for the latents->audio decode of each sample and
  unloaded straight after, returning the VRAM to training.
- A fixed noise seed (via ``generate_audio``'s own ``torch.Generator``)
  plus ``torch.random.fork_rng`` guarantees sampling NEVER perturbs
  training RNG state — resumed runs stay bit-identical whether or not
  sampling is enabled.
- Every failure is caught and reported as a warning; a broken sample must
  never kill a training run. After ``_MAX_FAILURES`` consecutive failures
  sampling disables itself for the rest of the run.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_LATENT_FPS = 25
_SAMPLE_RATE = 48000
_MAX_FAILURES = 3

DEFAULT_SAMPLE_LYRICS = """[Verse]
Walking down an empty street tonight
Shadows dancing underneath the light
Every step is taking me away
Searching for the words I could not say

[Chorus]
And I know we're never letting go
Hold on to what we found
Cause I know it's all we've ever known
No turning back now
"""


class EngineSampler:
    """Audio previews rendered by the external ``ace-synth`` C++ engine.

    Rather than re-implementing ACE-Step conditioning in Python (which
    proved fragile — see EpochSampler's history), this backend shells out
    to the exact inference stack used for real generations: GGUF DiT +
    Lua solver plugins + the engine's own prompt building.  The adapter
    is read from disk (the milestone/preview snapshot), so what you hear
    is literally the saved artifact.

    Requires: ``ace-synth`` binary (plugin-init fix of 2026-07-27), a
    ``<model_variant>-*.gguf`` DiT in the checkpoint dir, and the stock
    text-encoder/VAE GGUFs the engine registry auto-discovers.  Falls
    back to :class:`EpochSampler` via :func:`make_epoch_sampler` when
    anything is missing.
    """

    _GGUF_PREFERENCE = ("-Q8_0.gguf", "-Q6_K.gguf", "-Q5_K_M.gguf", "-BF16.gguf")

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.enabled: bool = int(getattr(cfg, "sample_every_n_epochs", 0) or 0) > 0
        self.last_sample_path: Optional[str] = None
        self._failures = 0
        self._meta: Optional[Dict[str, Any]] = None
        self._paths = self._resolve_paths(cfg)

    @property
    def available(self) -> bool:
        return self._paths is not None

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_paths(cfg: Any) -> Optional[Dict[str, Any]]:
        models_dir = Path(str(getattr(cfg, "checkpoint_dir", "") or ""))
        if not models_dir.is_dir():
            return None
        engine_dir = models_dir.parent / "engine"
        exe = None
        for cand in (
            engine_dir / "build" / "Release" / "ace-synth.exe",
            engine_dir / "build" / "ace-synth.exe",
            engine_dir / "ace-synth.exe",
            engine_dir / "build" / "ace-synth",
        ):
            if cand.is_file():
                exe = cand
                break
        if exe is None:
            return None

        variant = str(getattr(cfg, "model_variant", "") or "")
        gguf = None
        for suffix in EngineSampler._GGUF_PREFERENCE:
            cand = models_dir / f"{variant}{suffix}"
            if cand.is_file():
                gguf = cand
                break
        if gguf is None:
            for cand in sorted(models_dir.glob(f"{variant}-*.gguf")):
                if "convrot" not in cand.name:
                    gguf = cand
                    break
        if gguf is None:
            return None

        dll_dirs = [str(exe.parent)]
        trt_libs = engine_dir / "deps" / "tensorrt_libs"
        if trt_libs.is_dir():
            dll_dirs.append(str(trt_libs))
        return {
            "exe": exe,
            "engine_dir": engine_dir,
            "models_dir": models_dir,
            "gguf_name": gguf.name,
            "dll_dirs": dll_dirs,
        }

    # ------------------------------------------------------------------
    def _load_metadata(self) -> Dict[str, Any]:
        """Metadata from the first dataset sample (same source as EpochSampler)."""
        if self._meta is not None:
            return self._meta
        ds = Path(self.cfg.dataset_dir)
        pt_files = sorted(p for p in ds.glob("*.pt") if not p.name.endswith(".tmp.pt"))
        if not pt_files:
            raise RuntimeError(f"No .pt tensor files found in {ds}")
        data = torch.load(str(pt_files[0]), map_location="cpu", weights_only=False)
        self._meta = dict(data.get("metadata") or {})
        self._meta["_source"] = pt_files[0].name
        del data
        return self._meta

    # ------------------------------------------------------------------
    def should_sample(self, epoch: int) -> bool:
        n = int(getattr(self.cfg, "sample_every_n_epochs", 0) or 0)
        return self.enabled and n > 0 and epoch % n == 0

    # ------------------------------------------------------------------
    def generate(
        self,
        module: Any,  # unused — engine renders the on-disk adapter
        epoch: int,
        output_dir: Path,
        ema: Any = None,  # unused — snapshots are saved EMA-applied already
        tag: Optional[str] = None,
        force: bool = False,
        adapter_path: Optional[str] = None,
        save_adapter_fn: Any = None,
    ) -> Tuple[Optional[str], bool]:
        """Render one preview via ace-synth.  Never raises."""
        if not (self.enabled or force):
            return None, True
        if self._failures >= _MAX_FAILURES:
            return None, True

        import os
        import subprocess

        t0 = time.time()
        cfg = self.cfg
        p = self._paths
        # Resolve everything absolute HERE (in the training process, whose CWD
        # is the Side-Step root).  The ace-synth subprocess runs with a
        # different CWD, so relative run dirs like "trained_adapters/..."
        # would resolve against the wrong root and FATAL as "not found".
        samples_dir = (Path(output_dir) / "samples").resolve()
        samples_dir.mkdir(parents=True, exist_ok=True)
        stem = f"sample_{tag}_epoch_{epoch}" if tag else f"sample_epoch_{epoch}"

        try:
            # No snapshot on disk yet (periodic previews): save one now,
            # EMA-applied to match what the Python backend would render.
            if adapter_path is None and save_adapter_fn is not None:
                tmp_dir = samples_dir / ".preview_adapter"
                if ema is not None:
                    ema.apply()
                try:
                    save_adapter_fn(str(tmp_dir))
                finally:
                    if ema is not None:
                        ema.restore()
                adapter_path = str(tmp_dir)
            if adapter_path is not None:
                ap = Path(adapter_path).resolve()
                if ap.is_dir():
                    st = sorted(ap.glob("*.safetensors"))
                    if not st:
                        raise RuntimeError(f"No .safetensors found in {ap}")
                    ap = st[0]
                adapter_path = str(ap)

            meta = self._load_metadata()
            caption = str(meta.get("caption") or "").strip()
            trigger = str(meta.get("custom_tag") or "").strip()
            if trigger and not caption.lower().startswith(trigger.lower()):
                caption = f"{trigger}, {caption}" if caption else trigger
            lyrics = (getattr(cfg, "sample_lyrics", "") or "").strip() or DEFAULT_SAMPLE_LYRICS

            is_turbo = bool(getattr(cfg, "is_turbo", False))
            steps = (
                int(getattr(cfg, "sample_steps", 0) or 0)
                or int(getattr(cfg, "num_inference_steps", 0) or 0)
                or (8 if is_turbo else 30)
            )
            shift = float(getattr(cfg, "shift", 0.0) or 0.0) or (3.0 if is_turbo else 1.0)

            req: Dict[str, Any] = {
                "caption": caption,
                "lyrics": lyrics,
                "duration": int(round(float(cfg.sample_duration))),
                "vocal_language": "en",
                "inference_steps": steps,
                "guidance_scale": 1.0,
                "shift": shift,
                "seed": int(getattr(cfg, "sample_seed", 42)),
                "output_format": "wav16",
                "synth_model": p["gguf_name"],
            }
            for src_key, req_key in (("bpm", "bpm"), ("keyscale", "keyscale"),
                                     ("timesignature", "timesignature")):
                if meta.get(src_key):
                    req[req_key] = meta[src_key]
            if adapter_path is not None:
                req["adapters"] = [{"name": str(Path(adapter_path)).replace("\\", "/"),
                                    "scale": 1.0}]

            req_path = samples_dir / f".{stem}.json"
            req_path.write_text(json.dumps(req, indent=2), encoding="utf-8")

            # Release cached (but unused) VRAM so the one-shot ace-synth can
            # fit its DiT next to the resident training allocations.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            env = os.environ.copy()
            env["PATH"] = os.pathsep.join(p["dll_dirs"]) + os.pathsep + env.get("PATH", "")
            proc = subprocess.run(
                [str(p["exe"]), "--models", str(p["models_dir"]),
                 "--request", str(req_path)],
                cwd=str(p["engine_dir"]),
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            produced = req_path.with_name(req_path.name[:-len(".json")] + "0.wav")
            if proc.returncode != 0 or not produced.is_file():
                combined = (proc.stderr or "") + "\n" + (proc.stdout or "")
                lines = [ln for ln in combined.splitlines() if ln.strip()]
                tail = "\n".join(lines[-8:])
                raise RuntimeError(f"ace-synth exit={proc.returncode}: {tail}")

            final = samples_dir / f"{stem}.wav"
            if final.exists():
                final.unlink()
            produced.rename(final)
            req_path.unlink(missing_ok=True)

            self.last_sample_path = str(final)
            self._failures = 0
            msg = (
                f"[OK] Engine audio sample at epoch {epoch} -> samples/{final.name} "
                f"({steps} steps, shift {shift:g}, {time.time() - t0:.1f}s, ace-synth)"
            )
            logger.info("[Sample] %s", msg)
            return msg, True

        except Exception as exc:  # noqa: BLE001 — sampling must never kill training
            self._failures += 1
            logger.exception("[Sample] Engine preview failed (attempt %d): %s", self._failures, exc)
            suffix = ""
            if self._failures >= _MAX_FAILURES:
                suffix = f" — disabled after {self._failures} consecutive failures"
            return f"[WARN] Engine audio sample failed: {exc}{suffix}", False


def make_epoch_sampler(cfg: Any) -> Any:
    """Pick the preview backend: ace-synth engine when available, else Python.

    ``sample_backend`` config: 'auto' (default), 'engine', or 'python'.
    'engine' falls back to Python (with a log warning) when the binary or
    GGUF model cannot be found.
    """
    backend = str(getattr(cfg, "sample_backend", "auto") or "auto").lower()
    if backend != "python":
        eng = EngineSampler(cfg)
        if eng.available:
            logger.info("[Sample] Preview backend: ace-synth engine (%s)",
                        eng._paths["gguf_name"])
            return eng
        if backend == "engine":
            logger.warning("[Sample] sample_backend=engine but ace-synth/GGUF not found "
                           "— falling back to Python sampler")
    return EpochSampler(cfg)


def _precision_str(dtype: torch.dtype) -> str:
    """Map a torch dtype to the loader-module precision string."""
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return "fp32"


class EpochSampler:
    """Stateful per-run epoch preview generator.

    Instantiate once before the epoch loop; call :meth:`should_sample`
    and :meth:`generate` at each epoch boundary.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.enabled: bool = int(getattr(cfg, "sample_every_n_epochs", 0) or 0) > 0
        self.last_sample_path: Optional[str] = None
        self._cond: Optional[Dict[str, Any]] = None
        self._failures = 0

    # ------------------------------------------------------------------
    def should_sample(self, epoch: int) -> bool:
        n = int(getattr(self.cfg, "sample_every_n_epochs", 0) or 0)
        return self.enabled and n > 0 and epoch % n == 0

    # ------------------------------------------------------------------
    def _build_conditioning(self, device: Any, dtype: torch.dtype) -> Dict[str, Any]:
        """Encode the sample prompt once and cache the tensors on CPU.

        Loads the Qwen3 text encoder temporarily; it is deleted before
        this method returns.
        """
        from sidestep_engine.data.preprocess_prompt import build_simple_prompt
        from sidestep_engine.models.loader import load_silence_latent, load_text_encoder
        from sidestep_engine.vendor.preprocess_lyrics import encode_lyrics
        from sidestep_engine.vendor.preprocess_text import encode_text

        cfg = self.cfg
        ds = Path(cfg.dataset_dir)
        pt_files = sorted(p for p in ds.glob("*.pt") if not p.name.endswith(".tmp.pt"))
        if not pt_files:
            raise RuntimeError(f"No .pt tensor files found in {ds}")
        src = pt_files[0]
        data = torch.load(str(src), map_location="cpu", weights_only=False)
        meta = dict(data.get("metadata") or {})
        del data

        tag_position = "prepend"
        meta_json = ds / "preprocess_meta.json"
        if meta_json.is_file():
            try:
                tag_position = (
                    json.loads(meta_json.read_text(encoding="utf-8")).get("tag_position")
                    or "prepend"
                )
            except Exception:
                pass

        # Same caption/bpm/key/signature the adapter trains on, but with the
        # preview duration so the metas block matches what we generate.
        meta["duration"] = int(round(float(cfg.sample_duration)))
        prompt = build_simple_prompt(meta, tag_position=tag_position, use_genre=False)
        lyrics = (getattr(cfg, "sample_lyrics", "") or "").strip() or DEFAULT_SAMPLE_LYRICS

        logger.info("[Sample] Conditioning source: %s", src.name)
        logger.info("[Sample] Prompt: %s", prompt.replace("\n", " | ")[:400])

        precision = _precision_str(dtype)
        tokenizer, text_enc = load_text_encoder(cfg.checkpoint_dir, device, precision)
        try:
            with torch.no_grad():
                text_hs, text_mask = encode_text(text_enc, tokenizer, prompt, device, dtype)
                lyric_hs, lyric_mask = encode_lyrics(text_enc, tokenizer, lyrics, device, dtype)
        finally:
            del text_enc, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        silence_full = load_silence_latent(
            cfg.checkpoint_dir, device, precision, variant=cfg.model_variant
        )
        # Null-timbre reference: the model expects ~750 frames of the SILENCE
        # LATENT in the refer-audio slot when there is no reference audio
        # (upstream handler convention).  Feeding zeros(1,1,64) instead puts
        # garbage in the timbre encoder and wrecks every generation — this
        # rendered all previews as static tones / hissy noise.
        refer = silence_full[:, : min(750, silence_full.shape[1]), :]

        latent_length = max(1, int(round(float(cfg.sample_duration) * _LATENT_FPS)))
        silence = silence_full
        if silence.shape[1] < latent_length:
            reps = math.ceil(latent_length / silence.shape[1])
            silence = silence.repeat(1, reps, 1)
        silence = silence[:, :latent_length, :]

        return {
            "text_hs": text_hs.cpu(),
            "text_mask": text_mask.cpu(),
            "lyric_hs": lyric_hs.cpu(),
            "lyric_mask": lyric_mask.cpu(),
            "silence": silence.cpu(),
            "refer": refer.cpu(),
            "latent_length": latent_length,
            "prompt": prompt,
        }

    # ------------------------------------------------------------------
    def _decode_latents(self, latents: torch.Tensor, device: Any, dtype: torch.dtype) -> torch.Tensor:
        """Decode ``(1, T, 64)`` latents to a ``(channels, samples)`` float32 waveform.

        Loads the VAE for the duration of the decode only.
        """
        from sidestep_engine.models.loader import load_vae

        vae = load_vae(self.cfg.checkpoint_dir, device, _precision_str(dtype))
        try:
            with torch.no_grad():
                z = latents.transpose(1, 2).to(device=device, dtype=vae.dtype)
                audio = vae.decode(z).sample  # (1, C, N)
        finally:
            del vae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return audio.squeeze(0).to(torch.float32).cpu().clamp_(-1.0, 1.0)

    # ------------------------------------------------------------------
    def generate(
        self,
        module: Any,
        epoch: int,
        output_dir: Path,
        ema: Any = None,
        tag: Optional[str] = None,
        force: bool = False,
        adapter_path: Optional[str] = None,  # unused — samples live weights
        save_adapter_fn: Any = None,  # unused — samples live weights
    ) -> Tuple[Optional[str], bool]:
        """Generate one epoch preview. Returns ``(status_msg, ok)``.

        Never raises: failures are logged and returned as a warning
        message. ``module`` is the live ``FixedLoRAModule``.

        Args:
            tag: Optional label inserted into the output filename
                (``sample_<tag>_epoch_<N>.wav``) — used by loss-milestone
                previews.
            force: Generate even when periodic sampling is disabled
                (``sample_every_n_epochs == 0``).  The consecutive-failure
                breaker still applies.
        """
        if not (self.enabled or force):
            return None, True
        if self._failures >= _MAX_FAILURES:
            return None, True

        t0 = time.time()
        cfg = self.cfg
        model = module.model
        decoder = model.decoder
        device = module.device
        dtype = module.dtype
        was_training = bool(getattr(decoder, "training", False))
        ema_applied = False

        try:
            if self._cond is None:
                self._cond = self._build_conditioning(device, dtype)
            c = self._cond
            latent_length = c["latent_length"]

            text_hs = c["text_hs"].to(device)
            text_mask = c["text_mask"].to(device)
            lyric_hs = c["lyric_hs"].to(device)
            lyric_mask = c["lyric_mask"].to(device)
            silence = c["silence"].to(device=device, dtype=dtype)

            src_latents = silence  # text2music: generate over silence context
            chunk_masks = torch.ones(1, latent_length, 64, device=device, dtype=dtype)
            attention_mask = torch.ones(1, latent_length, device=device, dtype=dtype)
            refer_audio = c["refer"].to(device=device, dtype=dtype)
            refer_order = torch.zeros(1, device=device, dtype=torch.long)
            is_covers = torch.zeros(1, device=device, dtype=dtype)

            is_turbo = bool(getattr(cfg, "is_turbo", False))
            # Use the run's OWN inference recipe (cfg.shift / num_inference_steps,
            # e.g. shift=1 for the merge-base-sft-turbo-xl family) — hardcoding
            # the stock turbo defaults (shift 3) rendered every preview as a
            # washed-out hissy mess on shift-1 models, even with a near-identity
            # adapter.  Generic turbo/base values remain only as fallbacks.
            steps = (
                int(getattr(cfg, "sample_steps", 0) or 0)
                or int(getattr(cfg, "num_inference_steps", 0) or 0)
                or (8 if is_turbo else 30)
            )
            shift = float(getattr(cfg, "shift", 0.0) or 0.0) or (3.0 if is_turbo else 1.0)
            guidance = 1.0 if is_turbo else 7.0  # CFG engages only when > 1.0

            if ema is not None:
                ema.apply()  # no-op before EMA activates
                ema_applied = True
            decoder.eval()

            fork_devices = (
                list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            )
            with torch.random.fork_rng(devices=fork_devices):
                with torch.no_grad():
                    out = model.generate_audio(
                        text_hidden_states=text_hs,
                        text_attention_mask=text_mask,
                        lyric_hidden_states=lyric_hs,
                        lyric_attention_mask=lyric_mask,
                        refer_audio_acoustic_hidden_states_packed=refer_audio,
                        refer_audio_order_mask=refer_order,
                        src_latents=src_latents,
                        chunk_masks=chunk_masks,
                        is_covers=is_covers,
                        silence_latent=silence,
                        attention_mask=attention_mask,
                        seed=int(getattr(cfg, "sample_seed", 42)),
                        infer_steps=steps,
                        diffusion_guidance_scale=guidance,
                        shift=shift,
                        use_cache=False,  # training force-disables decoder KV cache
                        use_progress_bar=False,
                    )

            wav = self._decode_latents(out["target_latents"], device, dtype)

            samples_dir = Path(output_dir) / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            stem = f"sample_{tag}_epoch_{epoch}" if tag else f"sample_epoch_{epoch}"
            path = samples_dir / f"{stem}.wav"
            import torchaudio

            torchaudio.save(str(path), wav, _SAMPLE_RATE)
            self.last_sample_path = str(path)
            self._failures = 0
            msg = (
                f"[OK] Audio sample at epoch {epoch} -> samples/{path.name} "
                f"({steps} steps, shift {shift:g}, cfg {guidance:g}, {time.time() - t0:.1f}s)"
            )
            logger.info("[Sample] %s", msg)
            return msg, True

        except Exception as exc:  # noqa: BLE001 — sampling must never kill training
            self._failures += 1
            logger.exception("[Sample] Epoch preview failed (attempt %d): %s", self._failures, exc)
            suffix = ""
            if self._failures >= _MAX_FAILURES:
                suffix = f" — disabled after {self._failures} consecutive failures"
            return f"[WARN] Epoch audio sample failed: {exc}{suffix}", False

        finally:
            if ema_applied:
                try:
                    ema.restore()
                except Exception:
                    logger.exception("[Sample] EMA restore failed")
            if was_training:
                decoder.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
