$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$bundle = Join-Path $root 'outputs/longrun-local-inputs'
$ar = Join-Path $root 'outputs/cluster-results/ar-baseline-57759'
$sources = [System.Collections.Generic.List[string]]::new()
function Read-Artifact($path) {
    $sources.Add($path)
    return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
}
$plan = Read-Artifact (Join-Path $bundle 'plan.json')
$protocol = Read-Artifact (Join-Path $ar 'protocol.json')
$summary = Read-Artifact (Join-Path $ar 'summary.json')
$selection = Read-Artifact (Join-Path $ar 'best_selection.json')
$logPath = Join-Path $ar 'training_metrics.jsonl'
$sources.Add($logPath)
$log = @(Get-Content -LiteralPath $logPath | ForEach-Object { $_ | ConvertFrom-Json })
$selectedLog = @($log | Where-Object step -eq $selection.step)
if ($selectedLog.Count -ne 1) { throw 'AR selection log ambiguous' }
$runs = @(foreach ($family in @('small','published')) {
    foreach ($seed in 42..46) {
        $name = "$family-seed$seed"
        $status = Read-Artifact (Join-Path $bundle "runs/$name/status.json")
        if ($status.status -ne 'early_stopped') { throw "Unexpected state: $name" }
        if (($status.best_step % $plan.steps_per_epoch) -ne 0 -or ($status.step % $plan.steps_per_epoch) -ne 0) { throw 'Nonintegral epochs require partial-batch accounting' }
        [pscustomobject]@{
            run=$name; selected_step=$status.best_step; final_step=$status.step
            selected_epochs=$status.best_step/$plan.steps_per_epoch
            final_epochs=$status.step/$plan.steps_per_epoch
            selected_passage_presentations=($status.best_step/$plan.steps_per_epoch)*$plan.train_records
            final_passage_presentations=($status.step/$plan.steps_per_epoch)*$plan.train_records
            recorded_loop_seconds=$status.cumulative_seconds
            recorded_loop_hours=$status.cumulative_seconds/3600
            selected_checkpoint_wall_time=$null
        }
    }
})
$old = Read-Artifact (Join-Path $root 'outputs/cluster-results/diffusion-50k-retry-20260920/run_metadata.json')
$report = [ordered]@{
    ar=[ordered]@{
        budget_seconds=$protocol.budget_seconds; timed_phase_seconds=$summary.elapsed_budget_seconds
        loop_and_final_evaluation_seconds=$summary.total_loop_and_final_evaluation_seconds
        selected_step=$selection.step; final_step=$summary.step
        selected_step_pre_validation_elapsed_seconds=$selectedLog[0].elapsed
        selected_passage_draws=$selection.step*$protocol.effective_batch
        final_passage_draws=$summary.step*$protocol.effective_batch
        selected_corpus_equivalent_draws=$selection.step*$protocol.effective_batch/$plan.train_records
        final_corpus_equivalent_draws=$summary.step*$protocol.effective_batch/$plan.train_records
        gpu=$protocol.gpu; precision=$protocol.precision
    }
    final_diffusion_gpu=$plan.hardware.gpu
    final_diffusion_runs=$runs
    older_diffusion_visible_gpu_names=@($old.hardware.gpus | ForEach-Object name)
    older_diffusion_configured_devices=$old.arguments.num_devices
    older_diffusion_resume_dir=$old.arguments.resume_dir
    older_diffusion_job_57231_accounting='Original scheduler accounting not located in bounded audit; 16266 seconds is recorded in AR protocol.'
    caveats=@('AR elapsed at best step is recorded before that validation/checkpoint operation, not exact selected-model cost.', 'Diffusion cumulative loop time includes validation, saves and any in-loop sampling; excludes initial AE/cache preparation and final evaluation. Last save duration can be excluded by timestamp placement.', 'No exact time to diffusion best checkpoint is recorded in these status files; do not scale whole-run time by step fraction.', 'Corpus-equivalent AR draws are replacement sampling, not complete dataset passes.', 'Visible GPUs do not establish multi-GPU use; older diffusion is configured for one device.')
}
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'budget_reconciliation.json') -Encoding utf8
$runs | Export-Csv -LiteralPath (Join-Path $PSScriptRoot 'diffusion_budgets.csv') -NoTypeInformation -Encoding utf8
$sources | Sort-Object -Unique | ForEach-Object { [pscustomobject]@{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLower()} } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'budget_sources.json') -Encoding utf8
$report.ar | ConvertTo-Json
$runs | Format-Table run,selected_epochs,final_epochs,recorded_loop_hours
