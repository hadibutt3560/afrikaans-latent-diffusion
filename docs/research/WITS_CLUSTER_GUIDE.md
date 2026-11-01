# Wits cluster discovery and operation

The exact Wits scheduler, account, partition, GPU resource syntax, modules, and
scratch policy were not available during repository preparation. The supplied
files are generic Slurm templates. Do not invent values: discover them on the
cluster or obtain them from the Wits research-computing documentation/support.

## Discover the scheduler and resources

Run these read-only commands on the login node:

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

If `sbatch` exists, use the templates in `cluster/wits/`. Add the verified
`#SBATCH --account=...` and `#SBATCH --partition=...` lines. Confirm whether Wits
uses `--gres=gpu:1`, `--gpus=1`, or a site-specific GPU type.

If only PBS/PBS Pro is present, translate:

- `#SBATCH --job-name` → `#PBS -N`;
- time and memory to the Wits `select`/`walltime` syntax;
- `--gres` to the site GPU resource;
- `SLURM_SUBMIT_DIR` → `PBS_O_WORKDIR`;
- `SLURM_JOB_ID` → `PBS_JOBID`;
- `afterok` to `qsub -W depend=afterok:<job-id>`.

## Environment and cache preparation

Create the environment on the cluster and cache BART before requiring the cache
in a compute-node preflight:

```bash
conda env create -f environment.reproduction.yml
conda activate ld4lg-reproduction
python -m spacy download en_core_web_sm
export HF_HOME="$SCRATCH/ld4lg-$USER/huggingface"
python tools/preflight.py --allow-download --load-model \
  --output-dir "$SCRATCH/ld4lg-$USER/outputs/login-preflight"
```

If compute nodes have no internet, do the download on an approved networked
login/data-transfer node. Follow Wits policy; do not download large assets on a
login node if prohibited.

## First jobs and dependencies

After adding the real scheduler values:

```bash
PREFLIGHT_JOB=$(sbatch --parsable cluster/wits/00_preflight.slurm)
AE_SMOKE_JOB=$(sbatch --parsable --dependency=afterok:${PREFLIGHT_JOB} \
  cluster/wits/01_autoencoder_smoke.slurm)
```

After the AE smoke is inspected, submit diffusion smoke with its exact output:

```bash
sbatch --dependency=afterok:${AE_SMOKE_JOB} \
  --export=ALL,LATENT_MODEL_PATH=/absolute/path/to/validated/ae-output \
  cluster/wits/03_diffusion_smoke.slurm
```

Do not chain full jobs automatically until the corresponding pilot artifacts have
been reviewed.

## Monitoring and failure diagnosis

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
scontrol show job JOB_ID
tail -f /path/to/slurm-JOB_ID.out
```

The templates set W&B offline mode. Sync later only if permitted:

```bash
wandb sync "$WANDB_DIR/offline-run-*"
```

Checkpoints and outputs live under `$SCRATCH/ld4lg-$USER/outputs` by default.
Copy final, selected artifacts to approved persistent storage because scratch can
be purged.
