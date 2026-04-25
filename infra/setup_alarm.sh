#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="i-08b088fd29cbd5434"
REGION="us-east-2"

echo "Creating CloudWatch alarm to stop instance after 30 min idle..."

aws cloudwatch put-metric-alarm \
  --region "$REGION" \
  --alarm-name poet-idle-stop \
  --alarm-description "Stop EC2 when CPU < 5% for 30 minutes" \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions "Name=InstanceId,Value=$INSTANCE_ID" \
  --statistic Average \
  --period 300 \
  --evaluation-periods 6 \
  --threshold 5 \
  --comparison-operator LessThanThreshold \
  --alarm-actions "arn:aws:automate:${REGION}:ec2:stop"

echo "Done. Alarm 'poet-idle-stop' created."
