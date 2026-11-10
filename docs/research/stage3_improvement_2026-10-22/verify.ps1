$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$out=Join-Path $root 'outputs/stage3-decoder-20261022'
$parent=Join-Path $root 'outputs/reconstruction-audit-20261022'
$contract=Get-Content (Join-Path $out 'contract.json') -Raw | ConvertFrom-Json
$analysis=Get-Content (Join-Path $out 'analysis.json') -Raw | ConvertFrom-Json
$pairs=@(Get-Content (Join-Path $out 'pairs.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
$old=@(Get-Content (Join-Path $parent 'examples.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
function CheckHash($p,$expected) { if ((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw "Changed input: $p" } }
CheckHash (Join-Path $PSScriptRoot 'PROTOCOL.md') $contract.protocol_sha256
CheckHash (Join-Path $PSScriptRoot 'run_decoder_test.py') $contract.script_sha256
CheckHash (Join-Path $parent 'contract.json') $contract.parent_contract_sha256
CheckHash (Join-Path $parent 'manifest.json') $contract.parent_manifest_sha256
CheckHash (Join-Path $parent 'examples.jsonl') $contract.parent_examples_sha256
if ($pairs.Count -ne 81 -or @($pairs.id | Sort-Object -Unique).Count -ne 81) { throw 'Count mismatch' }
$changed=@(); $textChanged=0
for ($i=0; $i -lt 81; $i++) {
    $p=$pairs[$i]
    if ($p.id -ne $old[$i].id -or ($p.baseline.generated_ids -join ',') -ne ($old[$i].correct.generated_ids -join ',')) { throw 'Parent baseline mismatch' }
    if (($p.baseline.generated_ids -join ',') -ne ($p.treatment.generated_ids -join ',')) { $changed += $i }
    if ($p.baseline.text -ne $p.treatment.text) { $textChanged++ }
}
if (($changed -join ',') -ne ($analysis.changed_indices -join ',')) { throw 'Changed list mismatch' }
foreach ($arm in @('baseline','treatment')) {
    $values=@($pairs | ForEach-Object { $_.$arm })
    $mean=($values.chrf | Measure-Object -Average).Average
    if ([math]::Abs($mean-$analysis.arms.$arm.mean_chrf) -gt 1e-9) { throw 'Mean mismatch' }
    if (($values.number_matched | Measure-Object -Sum).Sum -ne $analysis.arms.$arm.number_matched) { throw 'Number counts mismatch' }
}
$lines=@('# Changed pairs', '', 'All token-changed pairs, in frozen manifest order.', '')
foreach ($i in $changed) { $p=$pairs[$i]; $lines+=@("## $i — $($p.id)", '', "Source: $($p.source)", '', "Baseline: $($p.baseline.text)", '', "Treatment: $($p.treatment.text)", '') }
$lines | Set-Content (Join-Path $PSScriptRoot 'CHANGED_PAIRS.md') -Encoding utf8
[ordered]@{status='passed';count=81;token_changed=$changed.Count;text_changed=$textChanged;parent_outputs_reproduced=$true;frozen_inputs_unchanged=$true;summary_recomputed=$true;automatic_gate_pass=$analysis.automatic_gate_pass} | ConvertTo-Json | Set-Content (Join-Path $out 'checks.json') -Encoding utf8
Get-Content (Join-Path $out 'checks.json')
