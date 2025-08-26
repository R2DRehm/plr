@echo on
setlocal EnableExtensions EnableDelayedExpansion

REM ---------- Global params ----------
set "TAG=_v2"
set "DATA_DIR=.\data"
set "BATCH=128"
set "LR=1e-3"
set "WD=5e-4"
set "WORKERS=2"

REM Epochs
set "EPOCHS_C10=120"
set "EPOCHS_C100=160"

REM PLR hyperparams
set "PLR_LAMBDA=0.6"
set "PLR_TAU=0.35"
set "PLR_K=2"
set "PLR_SPACE=feature"

REM Determinism (optional)
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"

echo ===========================
echo  CIFAR-10 (CE then PLR)
echo ===========================
for %%S in (0 1 2 3 4 5 6 7 8 9) do (

  REM ---- CIFAR-10 CE (lambda=0) ----
  set "RUN=runs\c10_smallcnn_ce_s%%S_full%TAG%"

  REM Si un FICHIER (pas un dossier) occupe ce chemin, on le supprime
  if exist "!RUN!" if not exist "!RUN!\NUL" (
    echo [FIX] Stray file "!RUN!" detected. Removing...
    del /f /q "!RUN!"
  )

  REM Skip uniquement si run deja termine (config + best.pt)
  if exist "!RUN!\config.json" if exist "!RUN!\best.pt" (
    echo [SKIP] !RUN! already trained.
  ) else (
    echo [CE][CIFAR-10][seed=%%S] Training -> !RUN!
    python -m plr_hilbert.train ^
      --dataset cifar10 --model cnn_small ^
      --epochs %EPOCHS_C10% --batch_size %BATCH% --lr %LR% --weight_decay %WD% ^
      --plr_lambda 0.0 ^
      --data_dir %DATA_DIR% --run_dir "!RUN!" ^
      --seed %%S --use_cuda --num_workers %WORKERS%
  )

  if exist "!RUN!\config.json" (
    echo [CE][CIFAR-10][seed=%%S] Evaluating best.pt...
    python -m plr_hilbert.eval --run "!RUN!" --ckpt best.pt --n_bins 15
  ) else (
    echo [WARN] Missing config.json in !RUN! ; skipping eval.
  )

  REM ---- CIFAR-10 PLR ----
  set "RUN=runs\c10_smallcnn_plr_t0p35_s%%S_full%TAG%"

  if exist "!RUN!" if not exist "!RUN!\NUL" (
    echo [FIX] Stray file "!RUN!" detected. Removing...
    del /f /q "!RUN!"
  )

  if exist "!RUN!\config.json" if exist "!RUN!\best.pt" (
    echo [SKIP] !RUN! already trained.
  ) else (
    echo [PLR][CIFAR-10][seed=%%S] Training -> !RUN!
    python -m plr_hilbert.train ^
      --dataset cifar10 --model cnn_small ^
      --epochs %EPOCHS_C10% --batch_size %BATCH% --lr %LR% --weight_decay %WD% ^
      --plr_lambda %PLR_LAMBDA% --plr_tau %PLR_TAU% --plr_k %PLR_K% --plr_space %PLR_SPACE% ^
      --data_dir %DATA_DIR% --run_dir "!RUN!" ^
      --seed %%S --use_cuda --num_workers %WORKERS%
  )

  if exist "!RUN!\config.json" (
    echo [PLR][CIFAR-10][seed=%%S] Evaluating best.pt...
    python -m plr_hilbert.eval --run "!RUN!" --ckpt best.pt --n_bins 15
  ) else (
    echo [WARN] Missing config.json in !RUN! ; skipping eval.
  )
)

echo ===========================
echo  CIFAR-100 (CE then PLR)
echo ===========================
for %%S in (0 1 2 3 4 5 6 7 8 9) do (

  REM ---- CIFAR-100 CE (lambda=0) ----
  set "RUN=runs\c100_smallcnn_ce_s%%S_full%TAG%"

  if exist "!RUN!" if not exist "!RUN!\NUL" (
    echo [FIX] Stray file "!RUN!" detected. Removing...
    del /f /q "!RUN!"
  )

  if exist "!RUN!\config.json" if exist "!RUN!\best.pt" (
    echo [SKIP] !RUN! already trained.
  ) else (
    echo [CE][CIFAR-100][seed=%%S] Training -> !RUN!
    python -m plr_hilbert.train ^
      --dataset cifar100 --model cnn_small ^
      --epochs %EPOCHS_C100% --batch_size %BATCH% --lr %LR% --weight_decay %WD% ^
      --plr_lambda 0.0 ^
      --data_dir %DATA_DIR% --run_dir "!RUN!" ^
      --seed %%S --use_cuda --num_workers %WORKERS%
  )

  if exist "!RUN!\config.json" (
    echo [CE][CIFAR-100][seed=%%S] Evaluating best.pt...
    python -m plr_hilbert.eval --run "!RUN!" --ckpt best.pt --n_bins 15
  ) else (
    echo [WARN] Missing config.json in !RUN! ; skipping eval.
  )

  REM ---- CIFAR-100 PLR ----
  set "RUN=runs\c100_smallcnn_plr_t0p35_s%%S_full%TAG%"

  if exist "!RUN!" if not exist "!RUN!\NUL" (
    echo [FIX] Stray file "!RUN!" detected. Removing...
    del /f /q "!RUN!"
  )

  if exist "!RUN!\config.json" if exist "!RUN!\best.pt" (
    echo [SKIP] !RUN! already trained.
  ) else (
    echo [PLR][CIFAR-100][seed=%%S] Training -> !RUN!
    python -m plr_hilbert.train ^
      --dataset cifar100 --model cnn_small ^
      --epochs %EPOCHS_C100% --batch_size %BATCH% --lr %LR% --weight_decay %WD% ^
      --plr_lambda %PLR_LAMBDA% --plr_tau %PLR_TAU% --plr_k %PLR_K% --plr_space %PLR_SPACE% ^
      --data_dir %DATA_DIR% --run_dir "!RUN!" ^
      --seed %%S --use_cuda --num_workers %WORKERS%
  )

  if exist "!RUN!\config.json" (
    echo [PLR][CIFAR-100][seed=%%S] Evaluating best.pt...
    python -m plr_hilbert.eval --run "!RUN!" --ckpt best.pt --n_bins 15
  ) else (
    echo [WARN] Missing config.json in !RUN! ; skipping eval.
  )
)

echo ===========================
echo  ALL QUEUED. New runs carry tag %TAG%
echo ===========================
endlocal
