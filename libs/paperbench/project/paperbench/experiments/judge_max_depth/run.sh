
SUBMISSION_PATH="data/judge_eval/rice/0/submission"
PAPER_ID="rice"
JUDGE="simple"
MODEL="o3-mini-2025-01-31"
for DEPTH in 1 2 3 4 100; do
    for SEED in 1 2 3; do
        OUTPUT_PATH="experiments/judge_max_depth/results/judge_depth${DEPTH}_seed${SEED}"
        mkdir -p $OUTPUT_PATH
        python paperbench/scripts/run_judge.py --submission-path $SUBMISSION_PATH --paper-id $PAPER_ID --judge $JUDGE --model $MODEL --out-dir $OUTPUT_PATH -d $DEPTH > "$OUTPUT_PATH/stdout.log" 2> "$OUTPUT_PATH/run_judge.log" &
    done
done
wait
