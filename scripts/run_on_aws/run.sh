#!/bin/bash
# run.sh — Command execution, polling, artifact download, and termination
# Sourced by run_on_aws.sh

# ============================================================================
# Step 6: Run command via SSM (nohup for long-running jobs)
# ============================================================================

echo "Step 6: Running command via SSM (nohup for long-running jobs)..."
echo "  Command: $COMMAND"

# Base64 encode the command to avoid escaping issues with special chars like && | etc
COMMAND_B64=$(echo "$COMMAND" | base64 | tr -d '\n')

RUN_SCRIPT_CONTENT="#!/bin/bash
set -euxo pipefail

# Create output directory
mkdir -p /home/ubuntu/output

# Create the actual job script with embedded base64 command
cat > /home/ubuntu/run_job.sh << JOBSCRIPT
#!/bin/bash
set -exo pipefail

# Activate venv and run command
source /opt/wisent-venv/bin/activate

# Use device-optimized dtypes (bfloat16 on CUDA) for better GPU memory efficiency
export WISENT_DTYPE=auto

# HuggingFace token for gated models (Meta-Llama, etc.)
export HF_TOKEN=\"$HF_TOKEN\"
export HUGGING_FACE_HUB_TOKEN=\"$HF_TOKEN\"
export HF_HOME=\"/home/ubuntu/.cache/huggingface\"

# Supabase database for pre-existing contrastive pairs and activations
export DATABASE_URL=\"postgresql://postgres.rbqjqnouluslojmmnuqi:REDACTED_DB_PASSWORD@aws-0-eu-west-2.pooler.supabase.com:5432/postgres\"
# Login to HuggingFace
huggingface-cli login --token \"$HF_TOKEN\" --add-to-git-credential || true

# Decode the command from base64 (COMMAND_B64_PLACEHOLDER will be replaced)
DECODED_COMMAND=\\\$(echo 'COMMAND_B64_PLACEHOLDER' | base64 -d)

echo 'Running command...'
echo \"Command: \\\$DECODED_COMMAND\"
eval \\\$DECODED_COMMAND 2>&1 | tee /home/ubuntu/output/command_output.log
EXIT_CODE=\\\$?

# Mark completion with exit code
if [ \\\$EXIT_CODE -eq 0 ]; then
    echo \"Command completed successfully at \\\$(date)\" > /home/ubuntu/output/COMPLETED
else
    echo \"Command failed with exit code \\\$EXIT_CODE at \\\$(date)\" > /home/ubuntu/output/FAILED
fi
echo 'Done!'

# Upload artifacts to GCS immediately so they're not lost if local script dies
INSTANCE_ID=\\\$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
GCS_ARTIFACT_PATH=\"gs://$GCS_BUCKET/artifacts/\\\$INSTANCE_ID/output.tar.gz\"
echo \"Uploading artifacts to \\\$GCS_ARTIFACT_PATH\"
cd /home/ubuntu/output && tar -czf /tmp/output.tar.gz --exclude='nohup.log' . && gcloud storage cp /tmp/output.tar.gz \\\$GCS_ARTIFACT_PATH

# Wait for local script to download, then self-terminate
# If local script dies, artifacts are safe in GCS
echo 'Waiting 30 minutes for local script before self-terminating...'
sleep 1800
echo \"Self-terminating instance \\\$INSTANCE_ID\"
aws ec2 terminate-instances --instance-ids \\\$INSTANCE_ID --region us-east-1
JOBSCRIPT

chmod +x /home/ubuntu/run_job.sh

# Run the job with nohup in background - this survives SSM timeout
nohup /home/ubuntu/run_job.sh > /home/ubuntu/output/nohup.log 2>&1 &
JOB_PID=\$!
echo \$JOB_PID > /home/ubuntu/output/job.pid
echo \"Started background job with PID: \$JOB_PID\"
echo 'Job launched in background'
"

# Replace placeholder with actual base64-encoded command
RUN_SCRIPT_CONTENT="${RUN_SCRIPT_CONTENT//COMMAND_B64_PLACEHOLDER/$COMMAND_B64}"

# Base64 encode the script to avoid any escaping issues
SCRIPT_B64=$(echo "$RUN_SCRIPT_CONTENT" | base64 | tr -d '\n')

# Create a JSON parameters file to avoid shell quoting issues
PARAMS_FILE=$(mktemp)
cat > "$PARAMS_FILE" << PARAMS_EOF
{
    "commands": ["echo $SCRIPT_B64 | base64 -d | bash"]
}
PARAMS_EOF

# Send command via SSM - this just starts the nohup job and returns quickly
RUN_CMD_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "file://$PARAMS_FILE" \
    --timeout-seconds 300 \
    --query 'Command.CommandId' \
    --output text)

rm -f "$PARAMS_FILE"

echo "SSM Command ID: $RUN_CMD_ID"

# Wait for the launch command to complete (should be quick)
echo "Waiting for job to launch..."
for i in {1..30}; do
    LAUNCH_STATUS=$(aws ssm get-command-invocation \
        --command-id "$RUN_CMD_ID" \
        --instance-id "$INSTANCE_ID" \
        --query 'Status' \
        --output text 2>/dev/null || echo "Pending")

    if [[ "$LAUNCH_STATUS" == "Success" ]]; then
        echo "Job launched successfully!"
        break
    elif [[ "$LAUNCH_STATUS" == "Failed" ]]; then
        echo "Failed to launch job!"
        aws ssm get-command-invocation \
            --command-id "$RUN_CMD_ID" \
            --instance-id "$INSTANCE_ID" \
            --query 'StandardErrorContent' \
            --output text 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# ============================================================================
# Step 7: Poll for job completion
# ============================================================================

echo "Step 7: Polling for job completion (checking COMPLETED/FAILED marker file)..."
START_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))

    CHECK_CMD_ID=$(aws ssm send-command \
        --instance-ids "$INSTANCE_ID" \
        --document-name "AWS-RunShellScript" \
        --parameters 'commands=["if [ -f /home/ubuntu/output/COMPLETED ]; then echo COMPLETED; cat /home/ubuntu/output/COMPLETED; elif [ -f /home/ubuntu/output/FAILED ]; then echo FAILED; cat /home/ubuntu/output/FAILED; else echo RUNNING; fi"]' \
        --timeout-seconds 30 \
        --query 'Command.CommandId' \
        --output text 2>/dev/null || echo "")

    if [[ -n "$CHECK_CMD_ID" ]]; then
        sleep 3
        CHECK_OUTPUT=$(aws ssm get-command-invocation \
            --command-id "$CHECK_CMD_ID" \
            --instance-id "$INSTANCE_ID" \
            --query 'StandardOutputContent' \
            --output text 2>/dev/null || echo "RUNNING")

        if [[ "$CHECK_OUTPUT" == COMPLETED* ]]; then
            echo "Command completed successfully!"
            break
        elif [[ "$CHECK_OUTPUT" == FAILED* ]]; then
            echo "Command failed!"
            echo "$CHECK_OUTPUT"
            break
        fi
    fi

    echo "Still running... (${ELAPSED_MIN}m elapsed)"
    sleep 30
done

# ============================================================================
# Step 8: Download artifacts via GCS
# ============================================================================

echo "Step 8: Downloading artifacts via GCS..."

GCS_KEY="runs/$(date +%Y%m%d-%H%M%S)-$$"

UPLOAD_CMD="cd $REMOTE_OUTPUT_DIR && tar -czf /tmp/output.tar.gz --exclude='nohup.log' . && gcloud storage cp /tmp/output.tar.gz gs://$GCS_BUCKET/$GCS_KEY/output.tar.gz"

SSM_UPLOAD_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[\"$UPLOAD_CMD\"]" \
    --timeout-seconds 3600 \
    --query 'Command.CommandId' \
    --output text 2>/dev/null || echo "")

if [[ -n "$SSM_UPLOAD_ID" ]]; then
    echo "Upload SSM Command ID: $SSM_UPLOAD_ID"
    echo "Uploading artifacts to GCS..."

    for i in {1..180}; do
        UL_STATUS=$(aws ssm get-command-invocation \
            --command-id "$SSM_UPLOAD_ID" \
            --instance-id "$INSTANCE_ID" \
            --query 'Status' \
            --output text 2>/dev/null || echo "Pending")

        if [[ "$UL_STATUS" == "Success" ]]; then
            echo "Upload complete. Downloading from GCS..."
            gcloud storage cp "gs://$GCS_BUCKET/$GCS_KEY/output.tar.gz" "$LOCAL_DOWNLOAD_DIR/output.tar.gz"
            cd "$LOCAL_DOWNLOAD_DIR"
            tar -xzf output.tar.gz
            rm output.tar.gz
            gcloud storage rm "gs://$GCS_BUCKET/$GCS_KEY/output.tar.gz"
            echo "Artifacts downloaded to: $LOCAL_DOWNLOAD_DIR"
            break
        elif [[ "$UL_STATUS" == "Failed" ]]; then
            echo "Upload command failed"
            aws ssm get-command-invocation \
                --command-id "$SSM_UPLOAD_ID" \
                --instance-id "$INSTANCE_ID" \
                --query 'StandardErrorContent' \
                --output text 2>/dev/null || true
            break
        fi

        if (( i % 10 == 0 )); then
            echo "Still uploading... ($((i * 10))s elapsed)"
        fi
        sleep 10
    done
else
    echo "Failed to start upload command"
fi

# ============================================================================
# Termination
# ============================================================================

echo ""
echo "=========================================="
echo "Run completed!"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Artifacts downloaded to: $LOCAL_DOWNLOAD_DIR"
echo ""
echo "Terminating instance..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" > /dev/null 2>&1 || true
echo "Instance terminated."
