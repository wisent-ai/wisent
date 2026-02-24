#!/bin/bash
# setup.sh — Argument parsing, instance launch, and SSM wait
# Sourced by run_on_aws.sh

# ============================================================================
# Argument parsing
# ============================================================================

show_usage() {
    echo "Usage: $0 --model <model> [--wisent-cmd <cmd>] \"command\" [local_download_dir]"
    echo "   OR: $0 --instance-type <type> \"command\" [local_download_dir]"
    echo ""
    echo "Options:"
    echo "  --model <model>         - Model name (e.g., JunHowie/Qwen3-8B-GPTQ-Int4)"
    echo "  --wisent-cmd <cmd>      - Wisent CLI command (for memory estimation)"
    echo "  --instance-type <type>  - Override automatic selection"
    echo ""
    echo "Instance types: g6e.xlarge (1x L40S 46GB), g6e.12xlarge (4x L40S 184GB), g6e.48xlarge (8x L40S 368GB)"
    echo ""
    echo "Examples:"
    echo "  $0 --model JunHowie/Qwen3-8B-GPTQ-Int4 --wisent-cmd generate-vector-from-task \\"
    echo "     \"wisent generate-vector-from-task --task mmlu_abstract_algebra --model JunHowie/Qwen3-8B-GPTQ-Int4\""
    echo ""
    echo "  $0 --instance-type g6e.xlarge \\"
    echo "     \"wisent tasks mmlu_abstract_algebra --model Qwen/Qwen2.5-0.5B-Instruct\""
    exit 1
}

MODEL=""
WISENT_CMD=""
INSTANCE_TYPE=""
COMMAND=""
LOCAL_DOWNLOAD_DIR="./aws_output"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --wisent-cmd) WISENT_CMD="$2"; shift 2 ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --help|-h) show_usage ;;
        -*) echo "ERROR: Unknown option $1"; show_usage ;;
        *)
            if [[ -z "$COMMAND" ]]; then COMMAND="$1"
            else LOCAL_DOWNLOAD_DIR="$1"; fi
            shift ;;
    esac
done

if [[ -z "$COMMAND" ]]; then
    echo "ERROR: Command is required"
    show_usage
fi

if [[ -z "$INSTANCE_TYPE" ]]; then
    if [[ -z "$MODEL" ]]; then
        echo "ERROR: Either --instance-type OR --model must be specified"
        show_usage
    fi

    if [[ -z "$WISENT_CMD" && "$COMMAND" =~ wisent[[:space:]]+([a-z-]+) ]]; then
        WISENT_CMD="${BASH_REMATCH[1]}"
        echo "Auto-detected wisent command: $WISENT_CMD"
    fi

    REQUIRED_MEMORY=$(get_model_memory_requirement "$MODEL" "$WISENT_CMD")
    echo "Model: $MODEL"
    [[ -n "$WISENT_CMD" ]] && echo "Wisent command: $WISENT_CMD"
    echo "Memory requirement: ${REQUIRED_MEMORY}GB"
    echo ""

    INSTANCE_TYPE=$(select_instance_type "$REQUIRED_MEMORY")
    if [[ "$INSTANCE_TYPE" == ERROR* ]]; then echo "$INSTANCE_TYPE"; exit 1; fi
    echo "Selected instance type: $INSTANCE_TYPE"
    echo ""
fi

echo "=========================================="
echo "AWS Run Script"
echo "=========================================="
echo "Command: $COMMAND"
echo "Instance type: $INSTANCE_TYPE"
echo "AMI: $AMI_ID"
echo "Remote output dir: $REMOTE_OUTPUT_DIR"
echo "Local download dir: $LOCAL_DOWNLOAD_DIR"
echo ""

mkdir -p "$LOCAL_DOWNLOAD_DIR"

# ============================================================================
# USER_DATA script (runs on EC2 instance during boot)
# ============================================================================

USER_DATA="#!/bin/bash
set -euxo pipefail
exec > /var/log/user-data.log 2>&1

echo \"Starting user data script at \$(date)\"

echo \"Waiting for apt locks to be released...\"
while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do
    echo \"Waiting for apt lock...\"
    sleep 5
done
echo \"Apt locks released, proceeding with installation\"

apt-get update
apt-get install -y python3-venv python3-pip git ca-certificates curl gnupg

# Install Docker
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \$(. /etc/os-release && echo \$VERSION_CODENAME) stable\" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu
chmod 666 /var/run/docker.sock

# Install gcloud CLI for GCS artifact transfer
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo \"deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main\" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update && apt-get install -y google-cloud-cli

python3 -m venv /opt/wisent-venv
source /opt/wisent-venv/bin/activate
pip install --upgrade pip

pip install --no-cache-dir wisent==${WISENT_VERSION} lm-eval optuna pytest psycopg2-binary

# Build the coding sandbox Docker image
WISENT_PATH=\$(python3 -c \"import wisent; print(wisent.__path__[0])\")
SAFE_DOCKER_DIR=\$(find \"\$WISENT_PATH\" -type d -name safe_docker -path '*/coding/safe_docker' 2>/dev/null | head -1)
if [ -n \"\$SAFE_DOCKER_DIR\" ] && [ -f \"\$SAFE_DOCKER_DIR/Dockerfile\" ]; then
    cd \"\$SAFE_DOCKER_DIR\"
    docker build -t coding/sandbox:polyglot-1.0 .
    echo \"Docker sandbox image built successfully from pip package\"
else
    echo \"WARNING: Could not find Dockerfile in wisent package at \$WISENT_PATH (skipping Docker build)\"
    exit 1
fi

pip install git+https://github.com/hendrycks/math.git
pip install auto-gptq optimum

mkdir -p /home/ubuntu/output
chown -R ubuntu:ubuntu /home/ubuntu/output
touch /home/ubuntu/.wisent_ready
chown ubuntu:ubuntu /home/ubuntu/.wisent_ready

echo \"User data script completed at \$(date)\"
"

USER_DATA_B64=$(echo "$USER_DATA" | base64)

# ============================================================================
# Instance launch with AZ retry logic
# ============================================================================

echo "Step 1: Launching EC2 instance..."

AVAILABILITY_ZONES=("us-east-1a" "us-east-1c" "us-east-1d" "us-east-1b" "us-east-1e" "us-east-1f")
MAX_AZ_RETRIES=${#AVAILABILITY_ZONES[@]}
MAX_CAPACITY_RETRIES=60
CAPACITY_RETRY_DELAY=30
INSTANCE_ID=""

get_subnet_for_az() {
    local az="$1"
    aws ec2 describe-subnets \
        --filters "Name=availability-zone,Values=$az" \
        --query 'Subnets[0].SubnetId' \
        --output text 2>/dev/null || echo ""
}

for ((capacity_retry=0; capacity_retry<MAX_CAPACITY_RETRIES; capacity_retry++)); do
    if [[ $capacity_retry -gt 0 ]]; then
        echo ""
        echo "All zones had insufficient capacity. Waiting ${CAPACITY_RETRY_DELAY}s before retry... (attempt $((capacity_retry+1))/$MAX_CAPACITY_RETRIES)"
        sleep $CAPACITY_RETRY_DELAY
    fi

    for ((az_retry=0; az_retry<MAX_AZ_RETRIES; az_retry++)); do
        AZ="${AVAILABILITY_ZONES[$az_retry]}"
        SUBNET_ID=$(get_subnet_for_az "$AZ")

        if [[ -z "$SUBNET_ID" || "$SUBNET_ID" == "None" ]]; then
            echo "No subnet found in $AZ, skipping..."
            continue
        fi

        echo "Attempting launch in $AZ (subnet: $SUBNET_ID)..."

        LAUNCH_OUTPUT=$(aws ec2 run-instances \
            --image-id "$AMI_ID" \
            --instance-type "$INSTANCE_TYPE" \
            --security-group-ids "$SECURITY_GROUP" \
            --subnet-id "$SUBNET_ID" \
            --iam-instance-profile Name="$IAM_INSTANCE_PROFILE" \
            --user-data "$USER_DATA_B64" \
            --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wisent-run-script},{Key=Purpose,Value=one-time-run}]" \
            2>&1) || true

        if echo "$LAUNCH_OUTPUT" | grep -q "InsufficientInstanceCapacity"; then
            echo "Insufficient capacity in $AZ, trying next zone..."
            sleep 2
            continue
        fi

        INSTANCE_ID=$(echo "$LAUNCH_OUTPUT" | python3 -c "import sys, json; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])" 2>/dev/null || echo "")
        if [[ -n "$INSTANCE_ID" && "$INSTANCE_ID" =~ ^i-[a-f0-9]{17}$ ]]; then
            echo "Instance launched in $AZ: $INSTANCE_ID"
            break 2
        fi

        INSTANCE_ID=$(echo "$LAUNCH_OUTPUT" | grep -oE 'i-[a-f0-9]{17}' | head -1 || true)
        if [[ -n "$INSTANCE_ID" ]]; then
            echo "Instance launched in $AZ: $INSTANCE_ID"
            break 2
        fi

        echo "Launch failed in $AZ: $LAUNCH_OUTPUT"
    done
done

if [[ -z "$INSTANCE_ID" ]]; then
    echo "ERROR: Failed to launch instance after $MAX_CAPACITY_RETRIES capacity retries"
    exit 1
fi

echo "Instance launched: $INSTANCE_ID"

echo "Step 2: Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
echo "Instance is running"

echo "Step 3: Getting instance public IP..."
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)
echo "Public IP: $PUBLIC_IP"

echo "Step 4: Waiting for SSM agent to be ready..."
for i in {1..60}; do
    STATUS=$(aws ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
        --query 'InstanceInformationList[0].PingStatus' \
        --output text 2>/dev/null || echo "None")
    if [[ "$STATUS" == "Online" ]]; then echo "SSM agent is online"; break; fi
    echo "Waiting for SSM agent... ($i/60)"
    if [[ $i -eq 60 ]]; then echo "ERROR: SSM agent did not come online"; exit 1; fi
done

echo "Step 5: Waiting for wisent installation to complete..."
for i in {1..360}; do
    CHECK_CMD_ID=$(aws ssm send-command \
        --instance-ids "$INSTANCE_ID" \
        --document-name "AWS-RunShellScript" \
        --parameters "commands=[\"test -f /home/ubuntu/.wisent_ready && echo READY || echo WAITING\"]" \
        --query 'Command.CommandId' \
        --output text 2>/dev/null || echo "")

    if [[ -n "$CHECK_CMD_ID" ]]; then
        for j in {1..10}; do
            CHECK_STATUS=$(aws ssm get-command-invocation \
                --command-id "$CHECK_CMD_ID" \
                --instance-id "$INSTANCE_ID" \
                --query 'Status' \
                --output text 2>/dev/null || echo "Pending")
            if [[ "$CHECK_STATUS" == "Success" ]]; then break; fi
        done

        CHECK_RESULT=$(aws ssm get-command-invocation \
            --command-id "$CHECK_CMD_ID" \
            --instance-id "$INSTANCE_ID" \
            --query 'StandardOutputContent' \
            --output text 2>/dev/null | tr -d '[:space:]' || echo "WAITING")

        if [[ "$CHECK_RESULT" == "READY" ]]; then echo "Wisent installation complete"; break; fi
    fi

    echo "Waiting for wisent installation... ($i/360)"
    sleep 15
    if [[ $i -eq 360 ]]; then
        echo "ERROR: Wisent installation did not complete"
        aws ssm send-command \
            --instance-ids "$INSTANCE_ID" \
            --document-name "AWS-RunShellScript" \
            --parameters "commands=[\"cat /var/log/user-data.log\"]" \
            --query 'Command.CommandId' \
            --output text 2>/dev/null || true
        exit 1
    fi
done
