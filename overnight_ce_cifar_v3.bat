@echo on
setlocal EnableExtensions EnableDelayedExpansion

REM ========= Params =========
set "TAG=_v2"
set "DATA_DIR=.\data"
set "BATCH=128"
set "LR=1e-3"
set "WD=5e-4"
set "WORKERS=2"
set "SEEDS=0 1 2 3 4 5 6 7 8 9"

set "EPOCHS_C10=120"
set "EPOCHS_C100=160"

REM (optionnel) determinisme CUDA; inoffensif si ignoré
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"

REM ---------- CIFAR-10 (CE only) ----------
echo ===========================
echo  CIFAR-10  (CE only, lambda=0)
echo ===========================
for %%S in (%SEEDS%) do (
  set "RUN=runs\c10_smallcnn_ce_s%%S_full%TAG%"

  REM si un FICHIER (pas un dossier) existe à ce chemin, on le supprime
  if exist "!RUN!" if not exist "!RUN!\NUL" (
    echo [FIX] Stray file "!RUN!" detected. Removing...
    del /f /q "!RUN!"
  )

  REM crée le dossier si besoin
  if not exist "!RUN!\NUL" md "!RUN!" 2>nul

  REM skip si déjà fini (config + best.pt)
  if exist "!RUN!\config.json" if exist "!RUN!\best.pt" (
    echo [SKIP] !RUN! already trained.
  ) else (
    echo [CE][CIFAR-10][seed=%%S] Training -> !RUN!
    call python -m plr_hilbert.train ^
      --dataset cifar10 --model cnn_small ^
      --epochs %EPOCHS_C10% --batch_size %BATCH% --lr %LR% --weight_decay %WD% ^
      --plr_lambda 0.0 ^
      --data_dir %DATA_DIR% --run_dir "!RUN!" ^
      --seed %%S --use_cuda --num_workers %WORKERS%  1>>"!RUN!\train.log" 2>&1

    if errorlevel 1 (
      echo [ERR] Train returned errorlevel !ERRORLEVEL! for !RUN!
      echo [ERR] See !RUN!\train.log
    )
  )

  if exist "!RUN!\best.pt" (
    echo [CE][CIFAR-10][seed=%%S] Evaluating best.pt...
    call python -m plr_hilbert.eval --run "!RUN!" --ckpt best.pt --n_bins 15  1>>"!RUN!\eval.log" 2>&1
  ) else (
    echo [WARN] No best.pt in !RUN! ; skipping eval. See !RUN!\train.log
  )
)

REM ---------- CIFAR-100 (CE only) ----------
echo ===========================
echo  CIFAR-100 (CE only, lambda=0)
echo ===========================
for %%S in (%SEEDS%) do (
  set "RUN=runs\c100_smallcnn_ce_s%%S_full%TAG%"

  if exist "!RUN!" if not exist "!RUN!\NUL" (
    echo [FIX] Stray file "!RUN!" detected. Removing...
    del /f /q "!RUN!"
  )
  if not exist "!RUN!\NUL" md "!RUN!" 2>nul

  if exist "!RUN!\config.json" if exist "!RUN!\best.pt" (
    echo [SKIP] !RUN! already trained.
  ) else (
    echo [CE][CIFAR-100][seed=%%S] Training -> !RUN!
    call python -m plr_hilbert.train ^
      --dataset cifar100 --model cnn_small ^
      --epochs %EPOCHS_C100% --batch_size %BATCH% --lr %LR% --weight_decay %WD% ^
      --plr_lambda 0.0 ^
      --data_dir %DATA_DIR% --run_dir "!RUN!" ^
      --seed %%S --use_cuda --num_workers %WORKERS%  1>>"!RUN!\train.log" 2>&1

    if errorlevel 1 (
      echo [ERR] Train returned errorlevel !ERRORLEVEL! for !RUN!
      echo [ERR] See !RUN!\train.log
    )
  )

  if exist "!RUN!\best.pt" (
    echo [CE][CIFAR-100][seed=%%S] Evaluating best.pt...
    call python -m plr_hilbert.eval --run "!RUN!" --ckpt best.pt --n_bins 15  1>>"!RUN!\eval.log" 2>&1
  ) else (
    echo [WARN] No best.pt in !RUN! ; skipping eval. See !RUN!\train.log
  )
)

echo ===========================
echo  ALL CE QUEUED. tag=%TAG%
echo ===========================
endlocal
