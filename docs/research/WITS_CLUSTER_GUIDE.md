# Wits cluster discovery and operation

`cluster/wits/` holds generic Slurm templates: the Wits scheduler, account,
partition, GPU resource syntax, modules and scratch policy were unknown when they
were written. Take site values from the cluster itself or from Wits
research-computing documentation and support.

## Discover the scheduler and resources

Read-only, on the login node:

```bash
command -v sbatch
command -v qsub
sinfo -o "%P %a %l %D %G"
scontrol show partition
sacctmgr show associations user="$USER" format=Account,Partition,QOS 2>/dev/null
module avail 2>&1 | less
echo "$SCRATCH"
df -h "$SCRATCH"
nvidia-smi
```

Slurm (`sbatch` present): add the verified `#SBATCH --account=...` and
`#SBATCH --partition=...` lines to the templates; confirm `--gres=gpu:1`,
`--gpus=1` or a site-specific GPU type.

PBS/PBS Pro only:

- `#SBATCH --job-name` → `#PBS -N`;
- time and memory → Wits `select`/`walltime` syntax;
- `--gres` → site GPU resource;
- `SLURM_SUBMIT_DIR` → `PBS_O_WORKDIR`;
- `SLURM_JOB_ID` → `PBS_JOBID`;
- `afterok` → `qsub -W depend=afterok:<job-id>`.

## Environment and cache preparation

Before any compute-node preflight that needs the BART cache:

```bash
conda env create -f environment.reproduction.yml
conda activate ld4lg-reproduction
python -m spacy download en_core_web_sm
export HF_HOME="$SCRATCH/ld4lg-$USER/huggingface"
python tools/preflight.py --allow-download --load-model \
  --output-dir "$SCRATCH/ld4lg-$USER/outputs/login-preflight"
```

Compute nodes without internet: download on an approved networked login or
data-transfer node, following Wits policy on large downloads.

## First jobs and dependencies

With real scheduler values in place:

```bash
PREFLIGHT_JOB=$(sbatch --parsable cluster/wits/00_preflight.slurm)
AE_SMOKE_JOB=$(sbatch --parsable --dependency=afterok:${PREFLIGHT_JOB} \
  cluster/wits/01_autoencoder_smoke.slurm)
```

After inspecting the AE smoke output, submit the diffusion smoke with its exact
path:

```bash
sbatch --dependency=afterok:${AE_SMOKE_JOB} \
  --export=ALL,LATENT_MODEL_PATH=/absolute/path/to/validated/ae-output \
  cluster/wits/03_diffusion_smoke.slurm
```

Chain full jobs only after reviewing the corresponding pilot artefacts.

## Monitoring and failure diagnosis

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
scontrol show job JOB_ID
tail -f /path/to/slurm-JOB_ID.out
```

Templates run W&B offline; sync later only if permitted:

```bash
wandb sync "$WANDB_DIR/offline-run-*"
```

Default output: `$SCRATCH/ld4lg-$USER/outputs`. Scratch can be purged; copy
selected final artefacts to approved persistent storage.
