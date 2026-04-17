#!/usr/bin/env bash
set -u

LOG_DIR="./run_logs"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/master_run_$(TZ="America/Toronto" date +%Y%m%d_%H%M%S).txt"

run_exp() {
    local name="$1"
    shift

    local ts
    ts=$(TZ="America/Toronto" date +%Y%m%d_%H%M%S)
    local run_log="$LOG_DIR/${ts}_${name}.txt"

    {
        echo "============================================================"
        echo "START: $(TZ="America/Toronto" date)"
        echo "NAME : $name"
        echo "CMD  : python 2210.py $*"
        echo "PWD  : $(pwd)"
        echo "------------------------------------------------------------"
        echo "nvidia-smi BEFORE RUN"
        nvidia-smi
        echo "------------------------------------------------------------"
        python 2210.py "$@"
        status=$?
        echo "------------------------------------------------------------"
        echo "EXIT STATUS: $status"
        echo "END: $(TZ="America/Toronto" date)"
        echo "============================================================"
        echo
        exit $status
    } 2>&1 | tee "$run_log" | tee -a "$MASTER_LOG"

    local cmd_status=${PIPESTATUS[0]}
    if [ "$cmd_status" -ne 0 ]; then
        echo "Run failed: $name"
        echo "See log: $run_log"
        exit "$cmd_status"
    fi

    echo "Sleeping 120 seconds before next run..." | tee -a "$MASTER_LOG"
    sleep 120
}

run_exp "01_single_t1_fp32_full_bs2" \
    --fusion single --modalities t1 --dtype fp32 --input_size full --batch_size 2

run_exp "02_single_flair_fp32_full_bs2" \
    --fusion single --modalities flair --dtype fp32 --input_size full --batch_size 2

run_exp "03_early_t1flair_fp32_full_bs2" \
    --fusion early --modalities t1_flair --dtype fp32 --input_size full --batch_size 2

run_exp "04_late_t1flair_fp32_full_bs1" \
    --fusion late --modalities t1_flair --dtype fp32 --input_size full --batch_size 1

run_exp "05_lateopt_t1flair_fp32_full_bs2" \
    --fusion late_opt --modalities t1_flair --dtype fp32 --input_size full --batch_size 2 --use_checkpoint --freeze_until none

run_exp "06_early_t1flair_amp_full_bs2" \
    --fusion early --modalities t1_flair --dtype amp --input_size full --batch_size 2

run_exp "07_late_t1flair_amp_full_bs2" \
    --fusion late --modalities t1_flair --dtype amp --input_size full --batch_size 2

run_exp "08_early_t1flair_fp32_small_bs2" \
    --fusion early --modalities t1_flair --dtype fp32 --input_size small --batch_size 2

run_exp "09_late_t1flair_fp32_small_bs2" \
    --fusion late --modalities t1_flair --dtype fp32 --input_size small --batch_size 2

run_exp "10_early_t1flair_amp_small_bs2" \
    --fusion early --modalities t1_flair --dtype amp --input_size small --batch_size 2

run_exp "11_late_t1flair_amp_small_bs2" \
    --fusion late --modalities t1_flair --dtype amp --input_size small --batch_size 2

echo "All experiments completed successfully." | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG"