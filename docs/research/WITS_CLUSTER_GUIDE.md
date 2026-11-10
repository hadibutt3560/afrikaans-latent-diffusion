# Wits cluster discovery and operation

The Wits scheduler, account, partition, GPU resource syntax, modules and scratch
policy were not known when the repository was prepared, so the files in
`cluster/wits/` are generic Slurm templates. Site values come from the cluster
itself or from Wits research-computing documentation and support.

## Discover the scheduler and resources

Read-only commands for the login node:

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

If `sbatch` exists, use the templates in `cluster/wits/` with the verified
`#SBATCH --account=...` and `#SBATCH --partition=...` lines added, and confirm
whether the site uses `--gres=gpu:1`, `--gpus=1` or a site-specific GPU type.

If only PBS/PBS Pro is present, translate:

- `#SBATCH --job-name` → `#PBS -N`;
- time and memory to the Wits `select`/`walltime` syntax;
- `--gres` to the site GPU resource;
- `SLURM_SUBMIT_DIR` → `PBS_O_WORKDIR`;
- `SLURM_JOB_ID` → `PBS_JOBID`;
- `afterok` to `qsub -W depend=afterok:<job-id>`.

## Environment and cache preparation

Create the environment on the cluster and cache BART before a compute-node
preflight that requires the cache:

```bash
conda env create -f environment.reproduction.yml
conda activate ld4lg-reproduction
python -m spacy download en_core_web_sm
export HF_HOME="$SCRATCH/ld4lg-$USER/huggingface"
python tools/preflight.py --allow-download --load-model \
  --output-dir "$SCRATCH/ld4lg-$USER/outputs/login-preflight"
```

If compute nodes have no internet access, download on an approved networked login
or data-transfer node, following Wits policy on large downloads.

## First jobs and dependencies

With the real scheduler values added:

```bash
PREFLIGHT_JOB=$(sbatch --parsable cluster/wits/00_preflight.slurm)
AE_SMOKE_JOB=$(sbatch --parsable --dependency=afterok:${PREFLIGHT_JOB} \
  cluster/wits/01_autoencoder_smoke.slurm)
```

Once the AE smoke output has been inspected, submit the diffusion smoke with its
exact path:

```bash
sbatch --dependency=afterok:${AE_SMOKE_JOB} \
  --export=ALL,LATENT_MODEL_PATH=/absolute/path/to/validated/ae-output \
  cluster/wits/03_diffusion_smoke.slurm
```

Full jobs are chained only after the corresponding pilot artefacts have been
reviewed.

## Monitoring and failure diagnosis

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
scontrol show job JOB_ID
tail -f /path/to/slurm-JOB_ID.out
```

The templates set W&B offline mode; sync later only if permitted:

```bash
wandb sync "$WANDB_DIR/offline-run-*"
```

Checkpoints and outputs go to `$SCRATCH/ld4lg-$USER/outputs` by default. Scratch
can be purged, so selected final artefacts are copied to approved persistent
storage.
